
import os
import sys
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AppPaths:
    reports_dir: str
    runs_dir: str
    eval_dir: str
    project_root: str
    datasets_dir: str


def get_paths(app_file: str) -> AppPaths:
    """Compute absolute paths used by the Streamlit app.

    `app_file` should be __file__ from the Streamlit entrypoint.
    """

    streamlit_app_dir = os.path.dirname(os.path.abspath(app_file))
    eval_dir = os.path.abspath(os.path.join(streamlit_app_dir, ".."))
    project_root = os.path.abspath(os.path.join(eval_dir, ".."))

    reports_dir = os.path.join(eval_dir, "evaluation", "reports")
    runs_dir = os.path.join(eval_dir, "evaluation", "runs")
    datasets_dir = os.path.join(eval_dir, "datasets")

    return AppPaths(
        reports_dir=reports_dir,
        runs_dir=runs_dir,
        eval_dir=eval_dir,
        project_root=project_root,
        datasets_dir=datasets_dir,
    )


def ensure_import_paths(paths: AppPaths) -> None:
    if paths.eval_dir not in sys.path:
        sys.path.insert(0, paths.eval_dir)
    if paths.project_root not in sys.path:
        sys.path.insert(0, paths.project_root)


def resolve_eval_path(
    given_path: str,
    base_dir: str,
    *,
    allowed_exts: Tuple[str, ...] | None = None,
) -> str:
    """
    Validate and resolve an evaluation file path.

    - If `given_path` is absolute and within `base_dir`, return it.
    - If relative, join with `base_dir`.
    - Reject paths that escape `base_dir`.
    - Optionally validate file extension.
    """
    if os.path.isabs(given_path):
        resolved = os.path.normpath(given_path)
    else:
        resolved = os.path.normpath(os.path.join(base_dir, given_path))

    # Ensure the resolved path is within base_dir
    real_base = os.path.normpath(base_dir)
    if not resolved.startswith(real_base + os.sep) and resolved != real_base:
        raise ValueError(
            f"Path '{given_path}' resolves to '{resolved}' which is outside the allowed base '{base_dir}'"
        )

    # Validate extension if specified
    if allowed_exts:
        _, ext = os.path.splitext(resolved)
        if ext.lower() not in allowed_exts:
            raise ValueError(f"File '{resolved}' has extension '{ext}'; allowed: {allowed_exts}")

    return resolved