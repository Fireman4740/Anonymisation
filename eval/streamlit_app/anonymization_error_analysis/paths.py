from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class AppPaths:
    reports_dir: str
    runs_dir: str
    eval_dir: str
    project_root: str


def get_paths(app_file: str) -> AppPaths:
    """Compute absolute paths used by the Streamlit app.

    `app_file` should be __file__ from the Streamlit entrypoint.
    """

    streamlit_app_dir = os.path.dirname(os.path.abspath(app_file))
    eval_dir = os.path.abspath(os.path.join(streamlit_app_dir, ".."))
    project_root = os.path.abspath(os.path.join(eval_dir, ".."))

    reports_dir = os.path.join(eval_dir, "evaluation", "reports")
    runs_dir = os.path.join(eval_dir, "evaluation", "runs")

    return AppPaths(
        reports_dir=reports_dir,
        runs_dir=runs_dir,
        eval_dir=eval_dir,
        project_root=project_root,
    )


def ensure_import_paths(paths: AppPaths) -> None:
    """Ensure `eval/` and repo root are importable."""

    if paths.eval_dir not in sys.path:
        sys.path.insert(0, paths.eval_dir)
    if paths.project_root not in sys.path:
        sys.path.insert(0, paths.project_root)
