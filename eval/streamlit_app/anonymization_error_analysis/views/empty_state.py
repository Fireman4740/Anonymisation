from __future__ import annotations

import streamlit as st


def render_empty_state() -> None:
    st.info("Selectionnez une source et chargez un report pour demarrer l'analyse.")
