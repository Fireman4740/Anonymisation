from __future__ import annotations

import os
import sys
from typing import Any, Tuple


def eval_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def project_root() -> str:
    return os.path.abspath(os.path.join(eval_root(), ".."))


def ensure_on_sys_path(path: str) -> str:
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def ensure_eval_importable() -> str:
    return ensure_on_sys_path(eval_root())


def ensure_project_importable() -> str:
    return ensure_on_sys_path(project_root())


def ensure_pipegraph_importable() -> str:
    ensure_project_importable()
    pipegraph_dir = os.path.join(project_root(), "pipegraph")
    return ensure_on_sys_path(pipegraph_dir)


def load_pipegraph() -> Tuple[Any, Any]:
    """Returns ``(create_pipeline_graph, create_initial_state)``."""
    ensure_pipegraph_importable()
    from src.graph import create_pipeline_graph  # type: ignore
    from src.state import create_initial_state  # type: ignore

    return create_pipeline_graph, create_initial_state
