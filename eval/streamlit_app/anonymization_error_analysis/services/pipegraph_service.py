from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.errors import AppError
from ..loaders.pipeline_cache import load_pipegraph_cached


def run_pipegraph_eval(
    *,
    dataset_kind: str,
    dataset_path: str,
    limit: Optional[int],
    config: Dict[str, Any],
    progress_cb: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    try:
        from eval.pipegraph_eval_local import (
            build_docs_from_anonymization_dataset,
            build_docs_from_db_bio,
            build_docs_from_tab,
            build_report,
        )
    except Exception as exc:
        raise AppError("Unable to import pipegraph eval modules", details=str(exc)) from exc

    try:
        create_pipeline_graph, create_initial_state = load_pipegraph_cached()
        pipeline = create_pipeline_graph()
    except Exception as exc:
        raise AppError("Unable to load PipeGraph", details=str(exc)) from exc

    try:
        if dataset_kind == "TAB":
            docs = build_docs_from_tab(dataset_path, limit=limit)
        elif dataset_kind == "DB-bio":
            docs = build_docs_from_db_bio(dataset_path, limit=limit)
        else:
            docs = build_docs_from_anonymization_dataset(dataset_path, limit=limit)
    except Exception as exc:
        raise AppError("Unable to load dataset", details=str(exc)) from exc

    try:
        report = build_report(
            docs, pipeline, create_initial_state, config=config, progress_cb=progress_cb
        )
    except Exception as exc:
        raise AppError("PipeGraph evaluation failed", details=str(exc)) from exc

    return report
