from __future__ import annotations

import streamlit as st


@st.cache_resource(show_spinner="Loading PipeGraph pipeline...")
def load_pipegraph_cached():
    from eval.core.pipeline import load_pipegraph

    return load_pipegraph()
