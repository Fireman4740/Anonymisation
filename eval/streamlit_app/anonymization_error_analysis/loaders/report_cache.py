from __future__ import annotations

import json
from typing import Any, Dict, List

import streamlit as st


@st.cache_data(show_spinner="Loading report...")
def load_report_cached(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
