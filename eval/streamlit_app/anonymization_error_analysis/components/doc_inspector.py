from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from ..spans import classify_spans, render_text_with_spans


def render_doc_inspector(doc: Dict[str, Any]) -> None:
    text = doc.get("full_text", doc.get("text_snippet", ""))
    gt = [tuple(x) if isinstance(x, list) else x for x in doc.get("ground_truth", [])]
    preds = [tuple(x) if isinstance(x, list) else x for x in doc.get("predictions", [])]
    spans = classify_spans(text, gt, preds)
    span_html = render_text_with_spans(text, spans)

    st.markdown(
        f"""
    <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6; font-family: monospace; white-space: pre-wrap;">
        {span_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    **Legende:**
    <span style="background-color: #d4edda; border: 1px solid #28a745; padding: 2px 5px; border-radius: 3px;">Vrai positif</span>
    <span style="background-color: #f8d7da; border: 1px solid #dc3545; padding: 2px 5px; border-radius: 3px;">Faux negatif</span>
    <span style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 2px 5px; border-radius: 3px;">Faux positif</span>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.write("### Ground Truth")
        st.json(gt)
    with c2:
        st.write("### Predictions")
        st.json(preds)

    if doc.get("leaks"):
        st.error(f"### Fuites detectees: {len(doc['leaks'])}")
        st.write(doc["leaks"])
