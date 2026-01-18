from __future__ import annotations

import os
from typing import List, Tuple


def list_tab_splits(tab_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(tab_dir):
        return []
    files = sorted([p for p in os.listdir(tab_dir) if p.endswith(".jsonl")])
    out = [(name.replace(".jsonl", ""), os.path.join(tab_dir, name)) for name in files]
    preferred = ["test", "dev", "train"]
    out.sort(key=lambda x: (preferred.index(x[0]) if x[0] in preferred else 999, x[0]))
    return out


def list_json_datasets(data_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(data_dir):
        return []
    files = sorted([p for p in os.listdir(data_dir) if p.endswith(".json")])
    return [(name, os.path.join(data_dir, name)) for name in files]


def list_db_bio_files(db_dir: str) -> List[str]:
    if not os.path.isdir(db_dir):
        return []
    return sorted([os.path.join(db_dir, p) for p in os.listdir(db_dir) if p.endswith(".jsonl")])
