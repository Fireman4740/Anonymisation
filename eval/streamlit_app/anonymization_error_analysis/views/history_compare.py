from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ..core.models import RunsFilter, RunSummary
from ..metrics import compute_dataset_metrics
from ..services.report_loader import load_report_from_file, load_report_from_run
from .compare import render_comparison
from .dashboard import render_dashboard
from .ratbench_dashboard import render_ratbench_dashboard


@dataclass(frozen=True)
class _Candidate:
    row_id: str
    label: str
    source: str
    path: str
    created_at: Optional[str]
    dataset: str
    avg_prec: float
    avg_rec: float
    avg_f2: float
    leaky_docs: int
    total_docs: int


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _build_candidates_from_runs(runs: List[RunSummary], filters: RunsFilter) -> List[_Candidate]:
    allowed_paths = set(filters.run_paths or [])
    candidates: List[_Candidate] = []

    for run in runs:
        if allowed_paths and run.path not in allowed_paths:
            continue

        dataset_name = str(run.dataset or "")
        if filters.dataset_filter and filters.dataset_filter.lower() not in dataset_name.lower():
            continue

        created_display = run.created_at or ""
        label = run.run_name or os.path.basename(str(run.path))

        candidates.append(
            _Candidate(
                row_id=f"run::{run.path}",
                label=label,
                source="run",
                path=run.path,
                created_at=created_display,
                dataset=dataset_name,
                avg_prec=_safe_float(run.avg_prec),
                avg_rec=_safe_float(run.avg_rec),
                avg_f2=_safe_float(run.avg_f2),
                leaky_docs=_safe_int(run.leaky_docs),
                total_docs=_safe_int(run.total_docs),
            )
        )

    return candidates


def _build_candidates_from_reports(report_files: List[str]) -> List[_Candidate]:
    candidates: List[_Candidate] = []
    for path in report_files:
        label = os.path.basename(path)
        candidates.append(
            _Candidate(
                row_id=f"report::{path}",
                label=label,
                source="report",
                path=path,
                created_at="",
                dataset="report détaillé",
                avg_prec=0.0,
                avg_rec=0.0,
                avg_f2=0.0,
                leaky_docs=0,
                total_docs=0,
            )
        )
    return candidates


def _load_candidate_report(
    candidate: _Candidate,
    *,
    run_store: Any,
    runs_dir: str,
    reports_dir: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if candidate.source == "run":
        meta_run, report = load_report_from_run(candidate.path, run_store, base_dir=runs_dir)
        is_ratbench = meta_run.get("dataset", {}).get("benchmark") == "RAT-Bench"
        meta = {
            "title": "Run RAT-Bench sauvegardé" if is_ratbench else "Run sauvegardé",
            "subtitle": f"{meta_run.get('created_at', '')} | {meta_run.get('dataset', {}).get('name', '')}",
            "source": "run",
            "path": candidate.path,
        }
        if is_ratbench:
            meta["_ratbench_result"] = {
                "summary": meta_run.get("aggregate_metrics", {}),
                "by_difficulty": meta_run.get("by_difficulty", {}),
                "by_scenario": meta_run.get("by_scenario", {}),
                "direct_id_detection_rates": meta_run.get("direct_id_detection_rates", {}),
                "details": report,
                "reid_risk": meta_run.get("reid_risk"),
            }
        return report, meta

    report = load_report_from_file(candidate.path, base_dir=reports_dir)
    if not isinstance(report, list):
        raise ValueError("Format de rapport non supporté.")

    metrics = compute_dataset_metrics(report)
    meta = {
        "title": "Rapport existant",
        "subtitle": (
            f"{os.path.basename(candidate.path)} | "
            f"P={metrics.get('avg_prec', 0.0):.1%} "
            f"R={metrics.get('avg_rec', 0.0):.1%} "
            f"F2={metrics.get('avg_f2', 0.0):.1%}"
        ),
        "source": "file",
        "path": candidate.path,
    }
    return report, meta


def render_history_compare_view(
    *,
    runs: List[RunSummary],
    filters: RunsFilter,
    run_store: Any,
    runs_dir: str,
    report_files: List[str],
    reports_dir: str,
) -> None:
    st.title("Historique & Comparaison")
    st.caption("Sélectionnez 1 run/rapport pour analyser, ou 2 pour comparer A/B.")

    candidates = _build_candidates_from_runs(runs, filters)
    candidates.extend(_build_candidates_from_reports(report_files))

    if not candidates:
        st.info("Aucun élément à afficher avec les filtres actuels.")
        return

    rows = []
    for c in candidates:
        rows.append(
            {
                "Sélection": False,
                "Nom": c.label,
                "Source": c.source,
                "Dataset": c.dataset,
                "Créé le": c.created_at,
                "Précision": c.avg_prec,
                "Rappel": c.avg_rec,
                "F2": c.avg_f2,
                "Fuites": c.leaky_docs,
                "Docs": c.total_docs,
                "_row_id": c.row_id,
            }
        )

    table = pd.DataFrame(rows)

    edited = st.data_editor(
        table,
        hide_index=True,
        use_container_width=True,
        disabled=["Nom", "Source", "Dataset", "Créé le", "Précision", "Rappel", "F2", "Fuites", "Docs", "_row_id"],
        column_config={
            "Sélection": st.column_config.CheckboxColumn("Sélection", help="Cochez 1 ou 2 lignes"),
            "Précision": st.column_config.NumberColumn(format="%.3f"),
            "Rappel": st.column_config.NumberColumn(format="%.3f"),
            "F2": st.column_config.NumberColumn(format="%.3f"),
            "_row_id": None,
        },
        key="history_compare_table",
    )

    selected_rows = edited[edited["Sélection"] == True]
    selected_ids = selected_rows["_row_id"].tolist()

    if not selected_ids:
        st.info("Cochez une ou deux lignes pour lancer l'analyse.")
        return

    if len(selected_ids) > 2:
        st.warning("Sélectionnez au maximum 2 lignes.")
        return

    candidate_by_id = {c.row_id: c for c in candidates}

    if len(selected_ids) == 1:
        candidate = candidate_by_id[selected_ids[0]]
        try:
            report, meta = _load_candidate_report(
                candidate,
                run_store=run_store,
                runs_dir=runs_dir,
                reports_dir=reports_dir,
            )
        except Exception as exc:
            st.error(f"Chargement impossible: {exc}")
            return

        st.markdown("---")
        ratbench_result = meta.pop("_ratbench_result", None)
        if ratbench_result is not None:
            render_ratbench_dashboard(ratbench_result, meta)
        else:
            render_dashboard(report, meta)
        return

    a = candidate_by_id[selected_ids[0]]
    b = candidate_by_id[selected_ids[1]]

    try:
        report_a, meta_a = _load_candidate_report(a, run_store=run_store, runs_dir=runs_dir, reports_dir=reports_dir)
        report_b, meta_b = _load_candidate_report(b, run_store=run_store, runs_dir=runs_dir, reports_dir=reports_dir)
    except Exception as exc:
        st.error(f"Comparaison impossible: {exc}")
        return

    st.markdown("---")
    render_comparison(report_a, report_b, meta_a, meta_b)
