from __future__ import annotations

from typing import Tuple

import streamlit as st

from ..core.models import DocFilter


def render_doc_filters() -> DocFilter:
    col1, col2 = st.columns(2)
    with col1:
        recall_range: Tuple[float, float] = st.slider(
            "Plage de recall",
            0.0,
            1.0,
            (0.0, 1.0),
            key="dashboard_recall_range",
        )
    with col2:
        show_leaks_only = st.checkbox(
            "Afficher uniquement les documents avec fuites",
            key="dashboard_leaks_only",
        )
    return DocFilter(recall_range=recall_range, show_leaks_only=show_leaks_only)
