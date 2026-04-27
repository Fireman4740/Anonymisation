from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(relative_path: str) -> Path:
    return project_root() / relative_path

