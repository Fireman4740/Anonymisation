from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from ..core.models import DocFilter, DocSelection


def _apply_filters(report: List[Dict[str, Any]], filters: DocFilter) -> List[Dict[str, Any]]:
    return [
        doc
        for doc in report
        if filters.recall_range[0] <= doc.get("recall", 0.0) <= filters.recall_range[1]
        and (not filters.show_leaks_only or doc.get("leaks_count", 0) > 0)
    ]


def render_doc_table(report: List[Dict[str, Any]], filters: DocFilter) -> DocSelection:
    filtered = _apply_filters(report, filters)
    if not filtered:
        st.warning("Aucun document ne correspond aux filtres.")
        return DocSelection(selected_doc_id=None)

    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is None:
        st.info("Pandas indisponible. Tableau desactive.")
        return DocSelection(selected_doc_id=None)

    df = pd.DataFrame(filtered)
    cols = ["doc_id", "recall", "precision", "leaks_count", "text_snippet"]
    display_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

    doc_ids = [str(d.get("doc_id")) for d in filtered if "doc_id" in d]
    if not doc_ids:
        return DocSelection(selected_doc_id=None)

    selected_doc_id = st.selectbox("Inspecter un document", doc_ids, key="dashboard_doc_select")
    return DocSelection(selected_doc_id=str(selected_doc_id))
