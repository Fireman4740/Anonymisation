from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ..metrics import compute_dataset_metrics
from ..spans import classify_spans, render_text_with_spans
from ..run_store_adapter import RunStore


def render_history_mode(*, runs_dir: str, run_store: RunStore) -> None:
    if run_store.list_run_files is None or run_store.load_run is None or run_store.get_created_at is None:
        st.error("Le module `eval/run_store.py` est requis pour gérer l'historique.")
        return

    st.title("Historique des runs")

    if not os.path.isdir(runs_dir):
        st.warning(f"Dossier runs introuvable: {runs_dir}")
        return

    all_files = run_store.list_run_files(runs_dir)
    if not all_files:
        st.info("Aucun run sauvegardé pour le moment.")
        return

    runs: List[Dict[str, Any]] = []
    for p in all_files:
        try:
            meta, data = run_store.load_run(p)
            dt = run_store.get_created_at(meta)
            metrics = compute_dataset_metrics(data)
            runs.append(
                {
                    "path": p,
                    "created_at": meta.get("created_at"),
                    "created_dt": dt,
                    "pipeline": meta.get("pipeline"),
                    "dataset": (meta.get("dataset") or {}).get("name"),
                    "run_name": meta.get("run_name"),
                    "limit": meta.get("limit"),
                    "config": meta.get("config"),
                    "avg_prec": metrics["avg_prec"],
                    "avg_rec": metrics["avg_rec"],
                    "avg_f2": metrics["avg_f2"],
                    "leaky_docs": metrics["leaky_docs"],
                    "total_docs": metrics["total_docs"],
                }
            )
        except Exception:
            continue

    if not runs:
        st.info("Aucun run lisible.")
        return

    st.sidebar.subheader("Filtres")
    min_dt = min((r["created_dt"] for r in runs if r["created_dt"] is not None), default=None)
    max_dt = max((r["created_dt"] for r in runs if r["created_dt"] is not None), default=None)

    if min_dt and max_dt:
        start_date = st.sidebar.date_input("Du", value=min_dt.date())
        end_date = st.sidebar.date_input("Au", value=max_dt.date())
    else:
        start_date = st.sidebar.date_input("Du", value=date.today())
        end_date = st.sidebar.date_input("Au", value=date.today())

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    dataset_filter = st.sidebar.text_input("Dataset contient", value="")
    config_contains = st.sidebar.text_input("Config contient", value="")
    config_key = st.sidebar.text_input("Config key", value="")
    config_value = st.sidebar.text_input("Config value (texte)", value="")

    def _in_range(r: Dict[str, Any]) -> bool:
        dt = r.get("created_dt")
        if dt is None:
            return True
        d = dt.date()
        return (d >= start_date) and (d <= end_date)

    def _config_match(r: Dict[str, Any]) -> bool:
        if config_contains:
            cfg = r.get("config")
            try:
                cfg_s = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
            except Exception:
                cfg_s = str(cfg)
            if config_contains.lower() not in cfg_s.lower():
                return False
        if not config_key:
            return True
        cfg = r.get("config")
        if not isinstance(cfg, dict):
            return False
        if config_key not in cfg:
            return False
        if not config_value:
            return True
        return str(cfg.get(config_key)) == str(config_value)

    filtered_runs = [
        r
        for r in runs
        if _in_range(r)
        and (not dataset_filter or dataset_filter.lower() in str(r.get("dataset", "")).lower())
        and _config_match(r)
    ]

    st.write(f"Runs: {len(filtered_runs)} / {len(runs)}")

    df_runs = pd.DataFrame(
        [
            {
                "created_at": r.get("created_at"),
                "dataset": r.get("dataset"),
                "run_name": r.get("run_name"),
                "avg_rec": r.get("avg_rec"),
                "avg_prec": r.get("avg_prec"),
                "avg_f2": r.get("avg_f2"),
                "leaky_docs": r.get("leaky_docs"),
                "total_docs": r.get("total_docs"),
                "path": os.path.basename(r.get("path", "")),
            }
            for r in filtered_runs
        ]
    )
    st.dataframe(df_runs, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Run")

    run_paths = [r["path"] for r in filtered_runs]
    selected_path = str(
        st.sidebar.selectbox(
            "Sélectionner un run",
            run_paths,
            key="history_selected_run",
            format_func=lambda p: os.path.basename(str(p)),
        )
    )
    meta, data = run_store.load_run(selected_path)

    st.subheader("Meta")
    st.json(meta)

    st.download_button(
        label="Télécharger le run (JSON)",
        data=json.dumps({"meta": meta, "data": data}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=os.path.basename(str(selected_path)),
        mime="application/json",
    )

    st.subheader("Documents")
    metrics = compute_dataset_metrics(data)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Precision", f"{metrics['avg_prec']:.2%}")
    c2.metric("Avg Recall", f"{metrics['avg_rec']:.2%}")
    c3.metric("Avg F2", f"{metrics['avg_f2']:.2%}")
    c4.metric("Leaky Docs", f"{metrics['leaky_docs']} / {metrics['total_docs']}")

    col1, col2 = st.columns(2)
    with col1:
        recall_range = st.slider("Recall Range", 0.0, 1.0, (0.0, 1.0), key="runs_recall")
    with col2:
        show_leaks_only = st.checkbox("Show only documents with leaks", key="runs_leaks")

    filtered_data = [
        d
        for d in data
        if recall_range[0] <= d.get("recall", 0.0) <= recall_range[1]
        and (not show_leaks_only or d.get("leaks_count", 0) > 0)
    ]

    if not filtered_data:
        st.warning("No documents match filters.")
        return

    df = pd.DataFrame(filtered_data)
    cols = ["doc_id", "recall", "precision", "leaks_count", "text_snippet"]
    display_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

    doc_ids = [d["doc_id"] for d in filtered_data]
    selected_doc_id = st.selectbox("Select Document ID to inspect", doc_ids, key="runs_doc")
    doc = next((d for d in filtered_data if d["doc_id"] == selected_doc_id), None)
    if not doc:
        return

    text = doc.get("full_text", doc.get("text_snippet", ""))
    gt = [tuple(x) if isinstance(x, list) else x for x in doc.get("ground_truth", [])]
    preds = [tuple(x) if isinstance(x, list) else x for x in doc.get("predictions", [])]
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
