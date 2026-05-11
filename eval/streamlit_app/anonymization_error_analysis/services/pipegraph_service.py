from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.errors import AppError
from ..loaders.pipeline_cache import load_pipegraph_cached


def run_pipegraph_eval(
    *,
    dataset_kind: str,
    dataset_path: str,
    split: Optional[str],
    limit: Optional[int],
    config: Dict[str, Any],
    progress_cb: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    try:
        from eval.core.datasets import get_allowed_labels, load_local_dataset_docs, normalize_dataset_key
        from eval.pipegraph_eval_local import build_report
    except Exception as exc:
        raise AppError("Unable to import pipegraph eval modules", details=str(exc)) from exc

    try:
        create_pipeline_graph, create_initial_state = load_pipegraph_cached()
        pipeline = create_pipeline_graph()
    except Exception as exc:
        raise AppError("Unable to load PipeGraph", details=str(exc)) from exc

    try:
        dataset_key = normalize_dataset_key(dataset_kind)
        docs = load_local_dataset_docs(
            dataset_kind=dataset_kind,
            dataset_path=dataset_path,
            limit=limit,
            split=split,
        )
    except Exception as exc:
        raise AppError("Unable to load dataset", details=str(exc)) from exc

    try:
        report = build_report(
            docs,
            pipeline,
            create_initial_state,
            config=config,
            progress_cb=progress_cb,
            allowed_labels=get_allowed_labels(dataset_key, profile=str(config.get("eval_profile") or config.get("profile") or "auto")),
        )
    except Exception as exc:
        raise AppError("PipeGraph evaluation failed", details=str(exc)) from exc

    return report
