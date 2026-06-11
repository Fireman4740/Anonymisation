from __future__ import annotations

import io
import glob
import json
import logging
import os
import threading
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from eval.api import build_eval_config

from .components.feedback import show_empty_state, show_error
from .components.sidebar import render_sidebar_ui
from .core.errors import AppError
from .core.state import (
    get_comparison_mode,
    get_comparison_report,
    get_comparison_report_meta,
    get_current_report,
    get_current_report_meta,
    init_state,
    update_current_report,
)
from .core.models import BenchmarkEvalConfig
from .paths import AppPaths, ensure_import_paths, get_paths, resolve_eval_path
from .run_store_adapter import load_run_store
from .services.pipegraph_service import run_pipegraph_eval
from .services.ratbench_service import run_ratbench_eval
from .services.report_loader import load_report_from_file, load_report_from_run
from .services.run_store_service import list_runs
from .views.compare import render_comparison
from .views.dashboard import render_dashboard
from .views.ratbench_dashboard import render_ratbench_dashboard
from .views.history_compare import render_history_compare_view
from .views.ablation_dashboard import render_ablation_dashboard


class _LiveTextStream(io.TextIOBase):
    def __init__(self, on_line) -> None:
        super().__init__()
        self._on_line = on_line
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._on_line(line)
        return len(s)

    def flush(self) -> None:
        if self._buffer.strip():
            self._on_line(self._buffer.strip())
        self._buffer = ""


class _StreamlitLogHandler(logging.Handler):
    def __init__(self, on_line) -> None:
        super().__init__(level=logging.INFO)
        self._on_line = on_line

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        if msg:
            self._on_line(msg)


class _LiveConsole:
    def __init__(self, title: str = "Logs en direct") -> None:
        st.markdown(f"### {title}")
        self._log_box = st.empty()
        self._lines: deque[str] = deque(maxlen=250)
        self._owner_thread_id = threading.get_ident()
        self._lock = threading.Lock()

    def _render_locked(self) -> None:
        self._log_box.code("\n".join(self._lines), language="text")

    def flush(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            return
        with self._lock:
            self._render_locked()

    def push(self, msg: str) -> None:
        text = str(msg).strip()
        if not text:
            return
        with self._lock:
            self._lines.append(text)
            if threading.get_ident() == self._owner_thread_id:
                self._render_locked()

    @contextmanager
    def capture(self):
        root_logger = logging.getLogger()
        handler = _StreamlitLogHandler(self.push)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        root_logger.addHandler(handler)

        out_stream = _LiveTextStream(self.push)
        err_stream = _LiveTextStream(lambda m: self.push(f"ERR | {m}"))
        try:
            with redirect_stdout(out_stream), redirect_stderr(err_stream):
                yield
        finally:
            out_stream.flush()
            err_stream.flush()
            self.flush()
            root_logger.removeHandler(handler)

def _run_benchmark(
    *,
    cfg: BenchmarkEvalConfig,
    runs_dir: str,
    datasets_dir: str,
    run_store,
    paths: "AppPaths",
) -> None:
    if cfg.type == "RAT-Bench" and cfg.ratbench_config:
        _run_ratbench_eval_ui(cfg=cfg.ratbench_config, runs_dir=runs_dir, run_store=run_store, paths=paths)
    elif cfg.local_config:
        res = _run_local_eval(cfg=cfg.local_config, runs_dir=runs_dir, datasets_dir=datasets_dir, run_store=run_store)
        if res:
            report, label, run_config = res
            meta = {
                "title": "Evaluation PipeGraph",
                "subtitle": label,
                "source": "local_eval",
                "config": run_config,
                "dataset": {
                    "name": cfg.local_config.dataset_label,
                    "path": cfg.local_config.dataset_path,
                    "split": cfg.local_config.split,
                    "limit": None if cfg.local_config.run_full_dataset else cfg.local_config.limit,
                },
                "run_name": cfg.local_config.run_name or None,
                "profile": cfg.local_config.profile,
                "eval_mode": cfg.local_config.eval_mode,
                "masking_mode": cfg.local_config.masking_mode,
                "doc_workers": cfg.local_config.doc_workers,
            }
            render_dashboard(report, meta)

# Local Eval implementation — uses evaluate.py as the central backend
def _run_local_eval(
    *,
    cfg,
    runs_dir: str,
    datasets_dir: str,
    run_store,
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    no_llm = not (cfg.llm_detection_enabled or cfg.llm_audit_enabled or cfg.llm_paraphrase_enabled)
    config: Dict[str, Any] = build_eval_config(
        dataset=str(cfg.dataset_kind),
        profile=str(cfg.profile),
        eval_mode=str(cfg.eval_mode),
        masking_mode=str(cfg.masking_mode),
        no_llm=no_llm,
        detection_mode=str(cfg.detection_mode),
        llm_provider=str(cfg.llm_provider) if cfg.llm_provider else None,
        llm_model=str(cfg.llm_model) if cfg.llm_model else None,
        detection_threshold=float(cfg.detection_threshold),
        paraphrase_intensity=int(cfg.paraphrase_intensity),
        extra={
            "enable_detection": bool(cfg.enable_detection),
            "enable_deterministic": bool(cfg.enable_deterministic),
            "enable_ai": bool(cfg.enable_ai),
            "enable_anonymization": bool(cfg.enable_anonymization),
            "llm_detection": bool(cfg.llm_detection_enabled),
            "llm_audit": bool(cfg.llm_audit_enabled),
            "llm_paraphrase": bool(cfg.llm_paraphrase_enabled),
            "rupta_enabled": bool(cfg.rupta_enabled),
            "rupta_max_iterations": int(cfg.rupta_max_iterations),
            "rupta_p_threshold": int(cfg.rupta_p_threshold),
        },
    )

    st.subheader("Configuration effective")
    with st.expander("Détails"):
        st.code(json.dumps(config, ensure_ascii=False, indent=2), language="json")
    st.caption(f"Dataset: {cfg.dataset_path}")
    st.caption(f"Workers documents demandés: {cfg.doc_workers if cfg.doc_workers is not None else 'Auto'}")

    launch_label = "▶ Lancer l'évaluation (complet)" if cfg.run_full_dataset else "▶ Lancer l'évaluation"
    if not st.button(launch_label, type="primary"):
        return None

    progress = st.progress(0)
    status = st.empty()
    live = _LiveConsole(title="Logs d'évaluation en direct")
    live.push("Démarrage de l'évaluation...")

    def _progress_cb(i: int, total: int, doc_id: str) -> None:
        if total <= 0:
            return
        progress.progress(min(1.0, max(0.0, i / total)))
        status.write(f"Doc {i}/{total}: {doc_id}")
        live.push(f"Progression {i}/{total} | doc={doc_id}")

    try:
        safe_dataset_path = resolve_eval_path(
            cfg.dataset_path,
            datasets_dir,
            allowed_exts=(".json", ".jsonl", ".csv", ".txt", ".train", ".dev", ".test"),
        )
    except ValueError as exc:
        show_error(AppError("Chemin de dataset invalide", details=str(exc)), title="Echec chargement dataset")
        return None

    try:
        with live.capture():
            report = run_pipegraph_eval(
                dataset_kind=cfg.dataset_kind,
                dataset_path=safe_dataset_path,
                split=cfg.split,
                limit=None if cfg.run_full_dataset else int(cfg.limit),
                config=config,
                progress_cb=_progress_cb,
                doc_workers=cfg.doc_workers,
            )
        live.push("Évaluation terminée avec succès.")
    except Exception as exc:
        live.push(f"Échec de l'évaluation: {exc}")
        show_error(exc, title="Echec de l'evaluation")
        return None
    finally:
        progress.progress(1.0)
        status.write("Terminé")

    st.download_button(
        label="⬇ Telecharger le report (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"pipegraph_report_{cfg.dataset_label.replace('/', '_')}.json",
        mime="application/json",
    )

    effective_workers = (
        ((report[0].get("effective_config") or {}).get("doc_workers"))
        if report
        else (cfg.doc_workers if cfg.doc_workers is not None else "Auto")
    )
    st.caption(
        "Stratégie effective: "
        f"documents={len(report)} | workers={effective_workers} | "
        f"provider={config.get('llm_provider') or 'config.json'} | detection={cfg.detection_mode}"
    )

    if cfg.save_run:
        if run_store.save_run is None or run_store.utc_now_iso is None:
            st.warning("Run store indisponible; sauvegarde impossible.")
        else:
            try:
                os.makedirs(runs_dir, exist_ok=True)
                meta = {
                    "created_at": run_store.utc_now_iso(),
                    "pipeline": "pipegraph",
                    "run_name": (cfg.run_name or None),
                    "dataset": {
                        "name": cfg.dataset_label,
                        "path": safe_dataset_path,
                    },
                    "limit": (None if cfg.run_full_dataset else cfg.limit),
                    "config": config,
                    "doc_workers": effective_workers,
                }
                saved_path = run_store.save_run(
                    runs_dir,
                    meta=meta,
                    data=report,
                    run_name=(cfg.run_name or None),
                )
                st.success(f"✅ Run sauvegardée: {saved_path}")
            except Exception as exc:
                show_error(exc, title="Echec sauvegarde run")

    return report, f"pipegraph_{cfg.dataset_label}", config


def _run_ratbench_eval_ui(
    *,
    cfg,
    runs_dir: str,
    run_store,
    paths: "AppPaths",
) -> None:
    lvl_str = f"Level {cfg.level}" if cfg.level else "Tous niveaux"
    st.subheader(f"RAT-Bench — {cfg.language.capitalize()} / {lvl_str}")

    no_llm = not (cfg.llm_detection_enabled or cfg.llm_audit_enabled or cfg.llm_paraphrase_enabled)
    config: Dict[str, Any] = build_eval_config(
        dataset="ratbench",
        profile=str(cfg.profile),
        eval_mode=str(cfg.eval_mode),
        masking_mode=str(cfg.masking_mode),
        no_llm=no_llm,
        detection_mode=str(cfg.detection_mode),
        llm_provider=str(cfg.llm_provider) if cfg.llm_provider else None,
        llm_model=str(cfg.llm_model) if cfg.llm_model else None,
        detection_threshold=float(cfg.detection_threshold),
        paraphrase_intensity=int(cfg.paraphrase_intensity),
        extra={
            "enable_detection": bool(cfg.enable_detection),
            "enable_deterministic": bool(cfg.enable_deterministic),
            "enable_ai": bool(cfg.enable_ai),
            "enable_anonymization": bool(cfg.enable_anonymization),
            "llm_detection": bool(cfg.llm_detection_enabled),
            "llm_audit": bool(cfg.llm_audit_enabled),
            "llm_paraphrase": bool(cfg.llm_paraphrase_enabled),
            "rupta_enabled": bool(cfg.rupta_enabled),
            "rupta_max_iterations": int(cfg.rupta_max_iterations),
            "rupta_p_threshold": int(cfg.rupta_p_threshold),
        },
    )

    with st.expander("Configuration effective", expanded=False):
        st.code(json.dumps(config, ensure_ascii=False, indent=2), language="json")
    st.caption(f"Workers documents demandés: {cfg.doc_workers if cfg.doc_workers is not None else 'Auto'}")

    session_key = f"ratbench_result_{cfg.language}_{cfg.level}_{cfg.limit}"
    cached_result = st.session_state.get(session_key)

    col_btn, col_dl = st.columns([1, 3])
    launch_label = "▶ Lancer l'évaluation (complet)" if cfg.run_full_dataset else "▶ Lancer l'évaluation"
    launch = col_btn.button(launch_label, type="primary", key="rb_launch")

    if launch or cached_result is None:
        if not launch and cached_result is None:
            st.info("Sélectionnez les paramètres dans la barre latérale puis cliquez sur Lancer.")
            return

        progress = st.progress(0)
        status = st.empty()
        live = _LiveConsole(title="Logs RAT-Bench en direct")
        live.push("Démarrage de l'évaluation RAT-Bench...")

        def _progress_cb(i: int, total: int, doc_id: str) -> None:
            if total <= 0:
                return
            progress.progress(min(1.0, max(0.0, i / total)))
            status.write(f"Doc {i}/{total}: {doc_id}")
            live.push(f"Progression {i}/{total} | doc={doc_id}")

        try:
            with live.capture():
                result = run_ratbench_eval(
                    language=cfg.language,
                    level=cfg.level,
                    limit=(None if cfg.run_full_dataset else cfg.limit),
                    config=config,
                    progress_cb=_progress_cb,
                    enable_risk_eval=bool(getattr(cfg, 'enable_risk_eval', False)),
                    doc_workers=cfg.doc_workers,
                )
            live.push("Évaluation RAT-Bench terminée avec succès.")
        except Exception as exc:
            live.push(f"Échec de l'évaluation RAT-Bench: {exc}")
            show_error(exc, title="Échec de l'évaluation RAT-Bench")
            return
        finally:
            progress.progress(1.0)
            status.write("Terminé")

        st.session_state[session_key] = result
        cached_result = result

        col_dl.download_button(
            label="⬇ Télécharger le report (JSON)",
            data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"ratbench_{cfg.language}_L{cfg.level or 'all'}.json",
            mime="application/json",
        )

        details = result.get("details", [])
        effective_workers = (
            ((details[0].get("effective_config") or {}).get("doc_workers"))
            if details
            else (cfg.doc_workers if cfg.doc_workers is not None else "Auto")
        )
        st.caption(
            "Stratégie effective: "
            f"documents={len(details)} | workers={effective_workers} | "
            f"provider={config.get('llm_provider') or 'config.json'} | detection={cfg.detection_mode}"
        )

        if cfg.save_run:
            if run_store.save_run is None:
                st.warning("Run store indisponible ; sauvegarde impossible.")
            else:
                try:
                    os.makedirs(runs_dir, exist_ok=True)
                    meta_run = {
                        "created_at": run_store.utc_now_iso(),
                        "pipeline": "pipegraph",
                        "run_name": cfg.run_name or None,
                        "dataset": {
                            "name": f"RAT-Bench/{cfg.language}/L{cfg.level or 'all'}",
                            "benchmark": "RAT-Bench",
                        },
                        "limit": None if cfg.run_full_dataset else cfg.limit,
                        "config": config,
                        "doc_workers": effective_workers,
                        "aggregate_metrics": result.get("summary", {}),
                        "by_difficulty": result.get("by_difficulty", {}),
                        "by_scenario": result.get("by_scenario", {}),
                        "direct_id_detection_rates": result.get("direct_id_detection_rates", {}),
                        "reid_risk": result.get("reid_risk"),
                    }
                    saved_path = run_store.save_run(
                        runs_dir,
                        meta=meta_run,
                        data=result.get("details", []),
                        run_name=cfg.run_name or None,
                    )
                    st.success(f"✅ Run sauvegardé : {saved_path}")
                except Exception as exc:
                    show_error(exc, title="Échec sauvegarde run")

    if cached_result:
        st.markdown("---")
        meta_display = {
            "title": "RAT-Bench — Résultats",
            "subtitle": f"{cfg.language.capitalize()} | L{cfg.level or 'all'} | limit={cfg.limit}",
            "source": "ratbench_eval"
        }
        render_ratbench_dashboard(cached_result, meta=meta_display)


def run_app(*, app_file: str) -> None:
    st.set_page_config(layout="wide", page_title="Analyse Pipeline Anonymisation")

    paths: AppPaths = get_paths(app_file)
    ensure_import_paths(paths)
    run_store = load_run_store()

    init_state()

    report_files = sorted(glob.glob(os.path.join(paths.reports_dir, "*_details.json")))
    try:
        all_runs = list_runs(paths.runs_dir, run_store)
    except Exception as exc:
        show_error(exc, title="Run store indisponible")
        all_runs = []

    active_mode, config_obj, _ = render_sidebar_ui(
        reports=report_files,
        runs=all_runs,
        eval_dir=paths.eval_dir,
        run_store=run_store,
        reports_dir=paths.reports_dir,
        runs_dir=paths.runs_dir,
    )

    if active_mode == "benchmark" and config_obj:
        _run_benchmark(
            cfg=config_obj,
            runs_dir=paths.runs_dir,
            datasets_dir=paths.datasets_dir,
            run_store=run_store,
            paths=paths
        )
    elif active_mode == "history":
        render_history_compare_view(
            runs=all_runs,
            filters=config_obj,
            run_store=run_store,
            runs_dir=paths.runs_dir,
            report_files=report_files,
            reports_dir=paths.reports_dir,
        )
    elif active_mode == "ablation":
        render_ablation_dashboard(
            cfg=config_obj,
            reports_dir=paths.reports_dir
        )
    else:
        show_empty_state("Mode non implémenté ou configuration manquante.")


if __name__ == "__main__":
    import sys
    run_app(app_file=sys.argv[0] if len(sys.argv) > 0 else __file__)
