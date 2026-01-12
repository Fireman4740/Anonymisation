from __future__ import annotations

import streamlit as st

from .paths import AppPaths, ensure_import_paths, get_paths
from .run_store_adapter import load_run_store
from .modes.history_runs import render_history_mode
from .modes.pipegraph_local import render_pipegraph_local_mode
from .modes.reports_existing import render_reports_existing_mode


def run_app(*, app_file: str) -> None:
    st.set_page_config(layout="wide", page_title="Anonymization Error Analysis")

    paths: AppPaths = get_paths(app_file)
    ensure_import_paths(paths)
    run_store = load_run_store()

    st.sidebar.title("Configuration")
    mode = st.sidebar.radio(
        "Mode",
        ["Rapports existants", "Évaluer PipeGraph (local)", "Historique (runs sauvegardés)"],
        index=0,
    )

    if mode == "Historique (runs sauvegardés)":
        render_history_mode(runs_dir=paths.runs_dir, run_store=run_store)
        return

    if mode == "Évaluer PipeGraph (local)":
        render_pipegraph_local_mode(eval_dir=paths.eval_dir, runs_dir=paths.runs_dir, run_store=run_store)
        return

    render_reports_existing_mode(reports_dir=paths.reports_dir)
