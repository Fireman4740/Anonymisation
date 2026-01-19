from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(value: str, max_len: int = 80) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        return "run"
    return value[:max_len]


def default_runs_dir(project_root: str) -> str:
    return os.path.join(project_root, "eval", "evaluation", "runs")


def build_run_filename(
    created_at_iso: str,
    pipeline_name: str,
    dataset_name: str,
    run_name: Optional[str] = None,
) -> str:
    # 2025-12-12T10:11:12+00:00 -> 20251212_101112
    safe_ts = re.sub(r"[^0-9]", "", created_at_iso)[:14]
    # run_name at the beginning if provided
    base = (
        f"{_slugify(run_name)}_{safe_ts}_{_slugify(pipeline_name)}_{_slugify(dataset_name)}"
        if run_name
        else f"{safe_ts}_{_slugify(pipeline_name)}_{_slugify(dataset_name)}"
    )
    return f"{base}.json"


def save_run(
    runs_dir: str,
    *,
    meta: Dict[str, Any],
    data: List[Dict[str, Any]],
    run_name: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    os.makedirs(runs_dir, exist_ok=True)

    created_at = str(meta.get("created_at") or utc_now_iso())
    pipeline_name = str(meta.get("pipeline") or "pipegraph")
    dataset_name = str((meta.get("dataset") or {}).get("name") or "dataset")

    filename = build_run_filename(created_at, pipeline_name, dataset_name, run_name=run_name)
    path = os.path.join(runs_dir, filename)

    if (not overwrite) and os.path.exists(path):
        # Avoid clobber: add suffix
        root, ext = os.path.splitext(path)
        i = 2
        while os.path.exists(f"{root}_{i}{ext}"):
            i += 1
        path = f"{root}_{i}{ext}"

    payload = {"meta": {**meta, "created_at": created_at}, "data": data}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


def list_run_files(runs_dir: str) -> List[str]:
    if not os.path.isdir(runs_dir):
        return []
    files = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) if f.endswith(".json")]
    files.sort(reverse=True)
    return files


def load_run(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "meta" in payload and "data" in payload:
        meta = payload.get("meta") or {}
        data = payload.get("data") or []
        if not isinstance(meta, dict) or not isinstance(data, list):
            raise ValueError("Run JSON invalide: meta doit être dict et data doit être list")
        return meta, data

    # Back-compat: si on charge un ancien report 'data' directement
    if isinstance(payload, list):
        return {"created_at": None, "pipeline": None, "dataset": {"name": None}}, payload

    raise ValueError("Run JSON invalide")


def get_created_at(meta: Dict[str, Any]) -> Optional[datetime]:
    raw = meta.get("created_at")
    if not raw:
        return None
    try:
        # datetime.fromisoformat supports +00:00
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
