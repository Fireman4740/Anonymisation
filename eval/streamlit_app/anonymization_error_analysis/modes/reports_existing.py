from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from ..metrics import compute_dataset_metrics
from ..spans import classify_spans, render_text_with_spans


def load_report(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_reports_existing_mode(*, reports_dir: str) -> None:
    if not os.path.exists(reports_dir):
        st.error(f"Reports directory not found: {reports_dir}")
        return

    report_files = glob.glob(os.path.join(reports_dir, "*_details.json"))
    if not report_files:
        st.error(f"No reports found in {reports_dir}")
        return

    report_names = [
        os.path.basename(f).replace("report_", "").replace("_details.json", "") for f in report_files
    ]
    selected_report_name = st.sidebar.selectbox("Select Dataset Report", report_names)

    selected_file = os.path.join(reports_dir, f"report_{selected_report_name}_details.json")

    try:
        data = load_report(selected_file)
    except Exception as e:
        st.error(f"Error loading report: {e}")
        return

    st.title(f"Analysis: {selected_report_name}")

    metrics = compute_dataset_metrics(data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Precision", f"{metrics['avg_prec']:.2%}")
    c2.metric("Avg Recall", f"{metrics['avg_rec']:.2%}")
    c3.metric("Avg F2", f"{metrics['avg_f2']:.2%}")
    c4.metric("Leaky Docs", f"{metrics['leaky_docs']} / {metrics['total_docs']}")

    st.subheader("Filter Documents")
    col1, col2 = st.columns(2)
    with col1:
        recall_range = st.slider("Recall Range", 0.0, 1.0, (0.0, 1.0))
    with col2:
        show_leaks_only = st.checkbox("Show only documents with leaks")

    filtered_data = [
        d
        for d in data
        if recall_range[0] <= d["recall"] <= recall_range[1]
        and (not show_leaks_only or d["leaks_count"] > 0)
    ]

    st.write(f"Showing {len(filtered_data)} documents")

    if not filtered_data:
        st.warning("No documents match filters.")
        return

    df = pd.DataFrame(filtered_data)
    cols = ["doc_id", "recall", "precision", "leaks_count", "text_snippet"]
    display_cols = [c for c in cols if c in df.columns]

    st.dataframe(df[display_cols], use_container_width=True)

    st.subheader("Document Inspector")

    doc_ids = [d["doc_id"] for d in filtered_data]
    selected_doc_id = st.selectbox("Select Document ID to inspect", doc_ids)

    doc = next((d for d in filtered_data if d["doc_id"] == selected_doc_id), None)

    if doc:
        text = doc.get("full_text", doc.get("text_snippet", ""))
        if "full_text" not in doc:
            st.warning(
                "⚠️ 'full_text' not found in report. Showing truncated snippet. Re-run evaluation pipeline to get full text."
            )

        gt = doc.get("ground_truth", [])
        preds = doc.get("predictions", [])

        gt = [tuple(x) if isinstance(x, list) else x for x in gt]
        preds = [tuple(x) if isinstance(x, list) else x for x in preds]

        spans = classify_spans(text, gt, preds)

        html = render_text_with_spans(text, spans)

        st.markdown(
            f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6; font-family: monospace; white-space: pre-wrap;">
            {html}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        **Legend:**
        <span style="background-color: #d4edda; border: 1px solid #28a745; padding: 2px 5px; border-radius: 3px;">True Positive (Correct)</span>
        <span style="background-color: #f8d7da; border: 1px solid #dc3545; padding: 2px 5px; border-radius: 3px;">False Negative (Missed)</span>
        <span style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 2px 5px; border-radius: 3px;">False Positive (Over-redacted)</span>
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
            st.error(f"### Leaks Detected: {len(doc['leaks'])}")
            st.write(doc["leaks"])
