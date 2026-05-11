from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from ..components.charts import render_charts, render_llm_export_view
from ..components.doc_inspector import render_doc_inspector
from ..components.doc_table import render_doc_table
from ..components.filters import render_doc_filters
from ..components.metrics_cards import render_metrics
from ..metrics import compute_dataset_metrics, compute_label_metrics


def render_dashboard(report: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> None:
    if meta:
        st.title(meta.get("title", "Analyse d'anonymisation"))
        subtitle = meta.get("subtitle")
        if subtitle:
            st.caption(subtitle)
    else:
        st.title("Analyse d'anonymisation")

    metrics = compute_dataset_metrics(report)
    label_metrics = compute_label_metrics(report)

    render_metrics(metrics)
    _render_profile_and_dual_metrics(report)

    st.subheader("Performance overview")

    tab_charts, tab_export = st.tabs(["Graphiques", "Export Texte (LLM)"])

    with tab_charts:
        render_charts(report, label_metrics=label_metrics)

    with tab_export:
        render_llm_export_view(report, label_metrics=label_metrics, meta=meta)

    st.subheader("Documents")
    filters = render_doc_filters()
    selection = render_doc_table(report, filters)

    if selection.selected_doc_id:
        doc = next(
            (d for d in report if str(d.get("doc_id")) == str(selection.selected_doc_id)), None
        )
        if doc:
            st.subheader("Document inspector")
            render_doc_inspector(doc)


def _avg_nested_metric(report: List[Dict[str, Any]], key: str, metric: str) -> float:
    vals = [float(doc.get(key, {}).get(metric, 0.0)) for doc in report if isinstance(doc.get(key), dict)]
    return sum(vals) / len(vals) if vals else 0.0


def _render_profile_and_dual_metrics(report: List[Dict[str, Any]]) -> None:
    if not report:
        return

    first = report[0]
    profile = first.get("masking_profile") or first.get("profile_diagnostics", {}).get("name")
    eval_mode = first.get("eval_mode")
    masking_mode = first.get("masking_mode")
    if profile or eval_mode or masking_mode:
        st.caption(
            f"Profil: {profile or 'n/a'} | "
            f"Mode évaluation: {eval_mode or 'n/a'} | "
            f"Mode masquage: {masking_mode or 'n/a'}"
        )

    if any("canonical_metrics" in doc or "benchmark_metrics" in doc for doc in report):
        st.subheader("Métriques canonique vs benchmark")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Canonique P", f"{_avg_nested_metric(report, 'canonical_metrics', 'precision'):.2%}")
        c2.metric("Canonique R", f"{_avg_nested_metric(report, 'canonical_metrics', 'recall'):.2%}")
        c3.metric("Canonique F2", f"{_avg_nested_metric(report, 'canonical_metrics', 'f2'):.2%}")
        c4.metric("Benchmark P", f"{_avg_nested_metric(report, 'benchmark_metrics', 'precision'):.2%}")
        c5.metric("Benchmark R", f"{_avg_nested_metric(report, 'benchmark_metrics', 'recall'):.2%}")
        c6.metric("Benchmark F2", f"{_avg_nested_metric(report, 'benchmark_metrics', 'f2'):.2%}")

    missed: Dict[str, int] = {}
    out_of_scope: Dict[str, int] = {}
    allowed = first.get("profile_diagnostics", {}).get("allowed_labels")
    allowed_set = set(allowed or [])
    for doc in report:
        for label, count in doc.get("fn_by_label", {}).items():
            missed[str(label)] = missed.get(str(label), 0) + int(count)
        if allowed_set:
            for _, _, label in doc.get("benchmark_predictions", doc.get("predictions", [])):
                if str(label) not in allowed_set:
                    out_of_scope[str(label)] = out_of_scope.get(str(label), 0) + 1

    if missed or out_of_scope:
        col_miss, col_scope = st.columns(2)
        if missed:
            col_miss.markdown("**Top labels ratés**")
            col_miss.dataframe(
                [{"label": k, "fn": v} for k, v in sorted(missed.items(), key=lambda item: -item[1])[:10]],
                use_container_width=True,
                hide_index=True,
            )
        if out_of_scope:
            col_scope.markdown("**Détectés hors scope dataset**")
            col_scope.dataframe(
                [{"label": k, "count": v} for k, v in sorted(out_of_scope.items(), key=lambda item: -item[1])[:10]],
                use_container_width=True,
                hide_index=True,
            )
