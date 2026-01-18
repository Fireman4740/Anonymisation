from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..core.errors import AppError
from ..run_store_adapter import RunStore


def load_report_from_file(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise AppError(f"Failed to load report: {path}", details=str(exc)) from exc


def load_report_from_run(
    path: str, run_store: RunStore
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if run_store.load_run is None:
        raise AppError("Run store not available")
    try:
        meta, data = run_store.load_run(path)
        return meta, data
    except Exception as exc:
        raise AppError(f"Failed to load run: {path}", details=str(exc)) from exc
