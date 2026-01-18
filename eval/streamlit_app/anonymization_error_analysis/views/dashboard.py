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

    st.subheader("Performance overview")

    tab_charts, tab_export = st.tabs(["Graphiques", "Export Texte (LLM)"])

    with tab_charts:
        render_charts(report, label_metrics=label_metrics)

    with tab_export:
        render_llm_export_view(report, label_metrics=label_metrics)

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
