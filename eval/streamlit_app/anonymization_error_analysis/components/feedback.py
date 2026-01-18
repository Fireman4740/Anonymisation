from __future__ import annotations

from typing import Optional

import streamlit as st

from ..core.errors import AppError


def show_error(err: Exception, *, title: Optional[str] = None) -> None:
    message = getattr(err, "message", None) or str(err)
    if title:
        st.error(f"{title}: {message}")
    else:
        st.error(message)
    details = getattr(err, "details", None)
    if details:
        with st.expander("Details"):
            st.code(details)


def show_app_error(err: AppError) -> None:
    show_error(err)


def show_empty_state(message: str) -> None:
    st.info(message)
