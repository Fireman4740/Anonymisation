"""Streamlit entrypoint.

Le code de l'app est organisé dans `eval/streamlit_app/anonymization_error_analysis/`.
Ce wrapper maintient `streamlit run app.py` compatible.
"""

from __future__ import annotations

import os
import sys


def _ensure_repo_root_on_syspath() -> None:
    # This file lives in <repo>/eval/streamlit_app/app.py
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_syspath()

from eval.streamlit_app.anonymization_error_analysis.main import run_app


if __name__ == "__main__":
    run_app(app_file=__file__)
