from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ..metrics import compute_dataset_metrics
from ..run_store_adapter import RunStore
from ..spans import classify_spans, render_text_with_spans


def _try_import_pipegraph_eval_local():
    try:
        import pipegraph_eval_local as pgel  # type: ignore

        return pgel
    except Exception:
        return None


@st.cache_resource
def _get_pipegraph_runner():
    pgel = _try_import_pipegraph_eval_local()
    if pgel is None:
        raise RuntimeError("Impossible d'importer eval/pipegraph_eval_local.py")
    create_pipeline_graph, create_initial_state = pgel.load_pipegraph()
    pipeline = create_pipeline_graph()
    return pgel, pipeline, create_initial_state


def render_pipegraph_local_mode(*, eval_dir: str, runs_dir: str, run_store: RunStore) -> None:
    pgel, _, _ = (None, None, None)
    try:
        pgel, pipeline, create_initial_state = _get_pipegraph_runner()
    except Exception:
        st.error("Le module d'évaluation local est introuvable: `eval/pipegraph_eval_local.py`.")
        return

    st.title("Analyse: PipeGraph (local)")

    ds = st.sidebar.selectbox("Dataset", ["TAB", "anonymization_dataset", "DB-bio"], index=0)

    if ds == "TAB":
        split = st.sidebar.selectbox("Split (TAB)", ["test", "dev", "train"], index=0)
        dataset_path = os.path.join(eval_dir, "datasets", "TAB", f"{split}.jsonl")
        dataset_meta: Dict[str, Any] = {"name": "TAB", "split": split, "path": dataset_path}
    elif ds == "DB-bio":
        split = st.sidebar.selectbox("Split (DB-bio)", ["test", "val", "train"], index=0)
        dataset_path = os.path.join(eval_dir, "datasets", "DB-bio", f"{split}.jsonl")
        dataset_meta = {"name": "DB-bio", "split": split, "path": dataset_path}
    else:
        data_dir = os.path.join(eval_dir, "datasets", "data")
        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        default_idx = 0
        if "anonymization_dataset.json" in json_files:
            default_idx = json_files.index("anonymization_dataset.json")
        selected_json = st.sidebar.selectbox("Fichier JSON", json_files, index=default_idx)
        dataset_path = os.path.join(data_dir, selected_json)
        dataset_meta = {"name": selected_json.replace(".json", ""), "path": dataset_path}

    st.sidebar.markdown("---")
    st.sidebar.subheader("Échantillonnage")
    run_all_dataset = st.sidebar.checkbox("Tout le dataset", value=False)
    limit: int | None
    if run_all_dataset:
        limit = None
        st.sidebar.caption("Mode: évaluation sur l'intégralité du dataset")
    else:
        limit = int(st.sidebar.number_input("Limit", min_value=1, max_value=5000, value=50, step=10))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres PipeGraph")

    enable_detection = st.sidebar.checkbox("Activer detection node", value=True)
    enable_deterministic = st.sidebar.checkbox("Activer deterministic", value=True)
    enable_ai = st.sidebar.checkbox("Activer AI NER", value=True)
    enable_anonymization = st.sidebar.checkbox("Activer anonymisation", value=True)
    detection_mode = st.sidebar.selectbox("Detection mode", ["serial", "parallel"], index=0)

    config: Dict[str, Any] = {
        "enable_detection": bool(enable_detection),
        "enable_deterministic": bool(enable_deterministic),
        "enable_ai": bool(enable_ai),
        "enable_anonymization": bool(enable_anonymization),
        "detection_mode": str(detection_mode),
    }

    run = st.sidebar.button("Lancer l'évaluation")

    if "pg_data" not in st.session_state:
        st.session_state.pg_data = None
    if "pg_meta" not in st.session_state:
        st.session_state.pg_meta = None

    if run:
        st.session_state.pg_data = None
        st.session_state.pg_meta = None

        try:
            progress = st.progress(0, text="Évaluation en cours…")
            current_doc = st.empty()

            def _progress_cb(done: int, total: int, doc_id: str) -> None:
                if total <= 0:
                    return
                pct = int((done / total) * 100)
                pct = 0 if pct < 0 else 100 if pct > 100 else pct
                progress.progress(pct, text=f"Évaluation… {done}/{total}")
                if doc_id:
                    current_doc.caption(f"Doc en cours: {doc_id}")

            if ds == "TAB":
                docs = pgel.build_docs_from_tab(dataset_path, limit=limit)
            elif ds == "DB-bio":
                docs = pgel.build_docs_from_db_bio(dataset_path, limit=limit)
            else:
                docs = pgel.build_docs_from_anonymization_dataset(dataset_path, limit=limit)

            data = pgel.build_report(docs, pipeline, create_initial_state, config=config, progress_cb=_progress_cb)
            progress.progress(100, text="Évaluation terminée")

            created_at = run_store.utc_now_iso() if run_store.utc_now_iso is not None else None
            st.session_state.pg_data = data
            st.session_state.pg_meta = {
                "created_at": created_at,
                "pipeline": "pipegraph",
                "dataset": dataset_meta,
                "limit": limit,
                "all_dataset": bool(run_all_dataset),
                "config": config,
            }
        except Exception as e:
            st.error(f"Erreur pendant l'évaluation: {e}")
            return

    if st.session_state.pg_data is None:
        st.info("Choisis un dataset puis clique sur 'Lancer l\'évaluation'.")
        return

    data = st.session_state.pg_data
    meta: Dict[str, Any] = st.session_state.pg_meta or {}

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sauvegarde")
    run_name = st.sidebar.text_input("Nom du run (optionnel)", value=meta.get("run_name", ""))
    save_click = st.sidebar.button("Sauvegarder le run")

    meta["run_name"] = run_name or None

    st.download_button(
        label="Télécharger le run (JSON)",
        data=json.dumps({"meta": meta, "data": data}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="pipegraph_run.json",
        mime="application/json",
    )

    if save_click:
        if run_store.save_run is None:
            st.error("Impossible d'enregistrer: `eval/run_store.py` manquant.")
        else:
            try:
                saved_path = run_store.save_run(runs_dir, meta=meta, data=data, run_name=run_name or None)
                st.sidebar.success(f"Run sauvegardé: {os.path.basename(saved_path)}")
            except Exception as e:
                st.sidebar.error(f"Erreur sauvegarde: {e}")

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
        if recall_range[0] <= d.get("recall", 0.0) <= recall_range[1]
        and (not show_leaks_only or d.get("leaks_count", 0) > 0)
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

    if not doc:
        return

    text = doc.get("full_text", doc.get("text_snippet", ""))
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
