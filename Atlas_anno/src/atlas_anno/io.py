from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from atlas_anno.paths import project_root


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def project_file(relative_path: str) -> Path:
    return project_root() / relative_path


def serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> Path:
    ensure_parent(path)
    path.write_text(json.dumps(serialize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_json_atomic(path: Path, payload: Any) -> Path:
    ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(serialize(payload), ensure_ascii=False, indent=2))
        tmp_name = handle.name
    os.replace(tmp_name, path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Any) -> Path:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize(payload), ensure_ascii=False) + "\n")
    return path


def write_jsonl(path: Path, records: Iterable[Any]) -> Path:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(serialize(record), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[Any]:
    if not path.exists():
        return []
    rows: List[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_text(path: Path, content: str) -> Path:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")
    return path


def flatten_record(record: Any, prefix: str = "") -> Dict[str, Any]:
    serialized = serialize(record)
    flat: Dict[str, Any] = {}
    if isinstance(serialized, dict):
        for key, value in serialized.items():
            nested_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(flatten_record(value, nested_key))
            elif isinstance(value, list):
                flat[nested_key] = json.dumps(value, ensure_ascii=False)
            else:
                flat[nested_key] = value
    else:
        flat[prefix or "value"] = serialized
    return flat
