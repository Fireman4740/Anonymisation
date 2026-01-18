from __future__ import annotations

from typing import List, Tuple

import streamlit as st

from ..run_store_adapter import RunStore


@st.cache_data(show_spinner="Loading runs list...")
def list_run_files_cached(runs_dir: str, run_store: RunStore) -> List[str]:
    if run_store.list_run_files is None:
        return []
    return run_store.list_run_files(runs_dir)


@st.cache_data(show_spinner="Loading run...")
def load_run_cached(path: str, run_store: RunStore) -> Tuple[dict, list[dict]]:
    if run_store.load_run is None:
        return {}, []
    return run_store.load_run(path)
