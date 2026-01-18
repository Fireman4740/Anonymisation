from __future__ import annotations

from typing import Any, Dict, List

from ..core.errors import AppError
from ..core.models import RunSummary
from ..metrics import compute_dataset_metrics
from ..run_store_adapter import RunStore


def list_runs(runs_dir: str, run_store: RunStore) -> List[RunSummary]:
    if (
        run_store.list_run_files is None
        or run_store.load_run is None
        or run_store.get_created_at is None
    ):
        raise AppError("Run store module unavailable")

    try:
        run_files = run_store.list_run_files(runs_dir)
    except Exception as exc:
        raise AppError("Unable to list runs", details=str(exc)) from exc

    runs: List[RunSummary] = []
    for path in run_files:
        try:
            meta, data = run_store.load_run(path)
            dt = run_store.get_created_at(meta)
            metrics = compute_dataset_metrics(data)
            runs.append(
                RunSummary(
                    path=path,
                    created_at=meta.get("created_at"),
                    created_dt=dt,
                    pipeline=meta.get("pipeline"),
                    dataset=(meta.get("dataset") or {}).get("name"),
                    run_name=meta.get("run_name"),
                    limit=meta.get("limit"),
                    config=meta.get("config"),
                    avg_prec=metrics.get("avg_prec", 0.0),
                    avg_rec=metrics.get("avg_rec", 0.0),
                    avg_f2=metrics.get("avg_f2", 0.0),
                    leaky_docs=metrics.get("leaky_docs", 0),
                    total_docs=metrics.get("total_docs", 0),
                )
            )
        except Exception:
            continue

    return runs
