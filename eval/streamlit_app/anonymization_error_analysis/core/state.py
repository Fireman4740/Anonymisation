from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from .constants import DEFAULT_SOURCE


DEFAULT_DOC_FILTERS: Dict[str, Any] = {
    "recall_range": (0.0, 1.0),
    "show_leaks_only": False,
}


def init_state() -> None:
    if "source" not in st.session_state:
        st.session_state["source"] = DEFAULT_SOURCE
    if "current_report" not in st.session_state:
        st.session_state["current_report"] = None
    if "current_report_meta" not in st.session_state:
        st.session_state["current_report_meta"] = None
    if "selected_doc_id" not in st.session_state:
        st.session_state["selected_doc_id"] = None
    if "doc_filters" not in st.session_state:
        st.session_state["doc_filters"] = dict(DEFAULT_DOC_FILTERS)
    if "comparison_report" not in st.session_state:
        st.session_state["comparison_report"] = None
    if "comparison_report_meta" not in st.session_state:
        st.session_state["comparison_report_meta"] = None
    if "comparison_mode" not in st.session_state:
        st.session_state["comparison_mode"] = False


def update_current_report(
    report: Optional[list[dict[str, Any]]], meta: Optional[Dict[str, Any]]
) -> None:
    st.session_state["current_report"] = report
    st.session_state["current_report_meta"] = meta
    if report is None:
        st.session_state["selected_doc_id"] = None
        # Reset comparison if main report is cleared
        st.session_state["comparison_report"] = None
        st.session_state["comparison_report_meta"] = None
        st.session_state["comparison_mode"] = False


def update_comparison_report(
    report: Optional[list[dict[str, Any]]], meta: Optional[Dict[str, Any]]
) -> None:
    st.session_state["comparison_report"] = report
    st.session_state["comparison_report_meta"] = meta


def set_comparison_mode(enabled: bool) -> None:
    st.session_state["comparison_mode"] = enabled


def update_doc_filters(filters: Dict[str, Any]) -> None:
    st.session_state["doc_filters"] = filters


def get_current_report() -> Optional[list[dict[str, Any]]]:
    return st.session_state.get("current_report")


def get_current_report_meta() -> Optional[Dict[str, Any]]:
    return st.session_state.get("current_report_meta")


def get_comparison_report() -> Optional[list[dict[str, Any]]]:
    return st.session_state.get("comparison_report")


def get_comparison_report_meta() -> Optional[Dict[str, Any]]:
    return st.session_state.get("comparison_report_meta")


def get_comparison_mode() -> bool:
    return st.session_state.get("comparison_mode", False)


def get_doc_filters() -> Dict[str, Any]:
    return dict(st.session_state.get("doc_filters", DEFAULT_DOC_FILTERS))


def set_selected_doc_id(doc_id: Optional[str]) -> None:
    st.session_state["selected_doc_id"] = doc_id


def get_selected_doc_id() -> Optional[str]:
    return st.session_state.get("selected_doc_id")
