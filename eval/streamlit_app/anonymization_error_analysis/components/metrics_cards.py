from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_metrics(metrics: Dict[str, Any]) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision moyenne", f"{metrics.get('avg_prec', 0.0):.2%}")
    m2.metric("Recall moyen", f"{metrics.get('avg_rec', 0.0):.2%}")
    m3.metric("F2 moyen", f"{metrics.get('avg_f2', 0.0):.2%}")
    m4.metric(
        "Docs avec fuites", f"{metrics.get('leaky_docs', 0)} / {metrics.get('total_docs', 0)}"
    )
