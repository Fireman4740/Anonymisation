from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

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
from .paths import AppPaths, ensure_import_paths, get_paths, resolve_eval_path
from .run_store_adapter import load_run_store
from .services.pipegraph_service import run_pipegraph_eval
from .services.report_loader import load_report_from_file, load_report_from_run
from .services.run_store_service import list_runs
from .views.compare import render_comparison
from .views.dashboard import render_dashboard


def _run_local_eval(
    *,
    cfg,
    runs_dir: str,
    datasets_dir: str,
    run_store,
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    config: Dict[str, Any] = {
        "enable_detection": bool(cfg.enable_detection),
        "enable_deterministic": bool(cfg.enable_deterministic),
        "enable_ai": bool(cfg.enable_ai),
        "enable_anonymization": bool(cfg.enable_anonymization),
        "detection_mode": str(cfg.detection_mode),
    }

    st.subheader("Config effective")
    st.code(json.dumps(config, ensure_ascii=False, indent=2), language="json")
    st.caption(f"Dataset: {cfg.dataset_path}")

    if not st.button("Lancer l'evaluation", type="primary"):
        return None

    progress = st.progress(0)
    status = st.empty()

    def _progress_cb(i: int, total: int, doc_id: str) -> None:
        if total <= 0:
            return
        progress.progress(min(1.0, max(0.0, i / total)))
        status.write(f"Doc {i}/{total}: {doc_id}")

    try:
        safe_dataset_path = resolve_eval_path(
            cfg.dataset_path,
            datasets_dir,
            allowed_exts=(".json", ".jsonl"),
        )
    except ValueError as exc:
        show_error(
            AppError("Chemin de dataset invalide", details=str(exc)),
            title="Echec chargement dataset",
        )
        return None

    try:
        report = run_pipegraph_eval(
            dataset_kind=cfg.dataset_kind,
            dataset_path=safe_dataset_path,
            limit=None if cfg.run_full_dataset else int(cfg.limit),
            config=config,
            progress_cb=_progress_cb,
        )
    except Exception as exc:
        show_error(exc, title="Echec de l'evaluation")
        return None
    finally:
        progress.progress(1.0)

    st.download_button(
        label="Telecharger le report (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"pipegraph_report_{cfg.dataset_label.replace('/', '_')}.json",
        mime="application/json",
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
                }
                saved_path = run_store.save_run(
                    runs_dir,
                    meta=meta,
                    data=report,
                    run_name=(cfg.run_name or None),
                )
                st.success(f"Run sauvegardee: {saved_path}")
            except Exception as exc:
                show_error(exc, title="Echec sauvegarde run")

    return report, f"pipegraph_{cfg.dataset_label}"


def _build_report_meta(
    *,
    title: str,
    subtitle: Optional[str] = None,
    source: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"title": title}
    if subtitle:
        meta["subtitle"] = subtitle
    if source:
        meta["source"] = source
    if path:
        meta["path"] = path
    return meta


def run_app(*, app_file: str) -> None:
    st.set_page_config(layout="wide", page_title="Analyse d'anonymisation")

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

    active_mode, config_obj, extra_data = render_sidebar_ui(
        reports=report_files,
        runs=all_runs,
        eval_dir=paths.eval_dir,
        run_store=run_store,
        reports_dir=paths.reports_dir,
        runs_dir=paths.runs_dir,
    )

    st.session_state["last_source"] = active_mode

    if active_mode == "history":
        sub_source = extra_data
        selection = config_obj

        if sub_source == "existing_report" and selection.selected_path:
            current_meta = get_current_report_meta()
            if not current_meta or current_meta.get("path") != selection.selected_path:
                try:
                    report = load_report_from_file(
                        selection.selected_path,
                        base_dir=paths.reports_dir,
                    )
                except Exception as exc:
                    show_error(exc, title="Echec chargement report")
                else:
                    name = os.path.basename(selection.selected_path)
                    meta = _build_report_meta(
                        title="Analyse de report",
                        subtitle=name,
                        source="existing_report",
                        path=selection.selected_path,
                    )
                    update_current_report(report, meta)

        elif sub_source == "saved_runs" and selection.selected_path:
            current_meta = get_current_report_meta()
            if not current_meta or current_meta.get("path") != selection.selected_path:
                try:
                    meta, report = load_report_from_run(
                        selection.selected_path,
                        run_store,
                        base_dir=paths.runs_dir,
                    )
                except Exception as exc:
                    show_error(exc, title="Echec chargement run")
                else:
                    title = "Run sauvegarde"
                    subtitle = f"{meta.get('pipeline')} | {meta.get('created_at')}"
                    meta_info = _build_report_meta(
                        title=title,
                        subtitle=subtitle,
                        source="saved_runs",
                        path=selection.selected_path,
                    )
                    update_current_report(report, meta_info)
                    with st.expander("Metadonnees du run"):
                        st.json(meta)

    elif active_mode == "local_eval":
        cfg = config_obj
        if cfg is not None:
            result = _run_local_eval(
                cfg=cfg,
                runs_dir=paths.runs_dir,
                datasets_dir=paths.datasets_dir,
                run_store=run_store,
            )
            if result is not None:
                report, safe_out_path = result
                subtitle = f"{cfg.dataset_label} | mode={cfg.detection_mode}"
                meta = _build_report_meta(
                    title="Evaluation PipeGraph (local)",
                    subtitle=subtitle,
                    source="local_eval",
                    path=safe_out_path,
                )
                update_current_report(report, meta)

    report = get_current_report()
    meta = get_current_report_meta()

    if not report:
        if active_mode == "comparison":
            st.info(
                "Veuillez d'abord charger un run principal (Historique ou Nouveau Run) avant de comparer."
            )
        else:
            show_empty_state("Selectionnez une source pour demarrer l'analyse.")
        return

    if get_comparison_mode():
        report_b = get_comparison_report()
        meta_b = get_comparison_report_meta()

        if report_b:
            render_comparison(report, report_b, meta, meta_b)
            return

        st.warning(
            "Mode Comparaison active, mais aucun rapport de reference (B) charge. Utilisez la barre laterale pour charger B."
        )

    render_dashboard(report, meta)
