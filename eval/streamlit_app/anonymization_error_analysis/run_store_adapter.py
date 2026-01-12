from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


LoadRunFn = Callable[[str], Tuple[Dict[str, Any], List[Dict[str, Any]]]]
ListRunFilesFn = Callable[[str], List[str]]
SaveRunFn = Callable[..., str]
GetCreatedAtFn = Callable[[Dict[str, Any]], Any]
UtcNowIsoFn = Callable[[], str]


@dataclass(frozen=True)
class RunStore:
    default_runs_dir: Optional[Callable[[str], str]]
    get_created_at: Optional[GetCreatedAtFn]
    list_run_files: Optional[ListRunFilesFn]
    load_run: Optional[LoadRunFn]
    save_run: Optional[SaveRunFn]
    utc_now_iso: Optional[UtcNowIsoFn]


def _load_run_store_module():
    for name in ("eval.run_store", "run_store"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def load_run_store() -> RunStore:
    mod = _load_run_store_module()
    return RunStore(
        default_runs_dir=getattr(mod, "default_runs_dir", None),
        get_created_at=getattr(mod, "get_created_at", None),
        list_run_files=getattr(mod, "list_run_files", None),
        load_run=getattr(mod, "load_run", None),
        save_run=getattr(mod, "save_run", None),
        utc_now_iso=getattr(mod, "utc_now_iso", None),
    )
