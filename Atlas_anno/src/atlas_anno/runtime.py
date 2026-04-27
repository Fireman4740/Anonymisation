from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from atlas_anno.config import load_config
from atlas_anno.console import ProgressBar, log
from atlas_anno.io import serialize
from atlas_anno.records import llm_run_meta_from_dict
from atlas_anno.schemas import LLMRunMeta
from atlas_anno.storage import append_stage_checkpoint, load_stage_checkpoints


@dataclass
class _StageWorkerState:
    active_workers: int = 0
    peak_workers: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def enter(self) -> None:
        with self.lock:
            self.active_workers += 1
            self.peak_workers = max(self.peak_workers, self.active_workers)

    def exit(self) -> None:
        with self.lock:
            self.active_workers = max(0, self.active_workers - 1)

    def snapshot(self) -> tuple[int, int]:
        with self.lock:
            return self.active_workers, self.peak_workers


def build_runtime_options(
    *,
    batch_name: str,
    reasoning_workers: int | None,
    creative_workers: int | None,
    resume_enabled: bool | None,
    cache_enabled: bool | None,
) -> Dict[str, Any]:
    runtime_defaults = load_config().defaults.get("llm", {}).get("runtime", {})
    return {
        "batch_name": batch_name,
        "reasoning_workers": reasoning_workers if reasoning_workers is not None else int(runtime_defaults.get("reasoning_workers", 12)),
        "creative_workers": creative_workers if creative_workers is not None else int(runtime_defaults.get("creative_workers", 8)),
        "checkpoint_every": int(runtime_defaults.get("checkpoint_every", 1)),
        "resume_enabled": bool(runtime_defaults.get("resume_enabled", True)) if resume_enabled is None else resume_enabled,
        "cache_enabled": bool(runtime_defaults.get("cache_enabled", True)) if cache_enabled is None else cache_enabled,
        "backoff_initial_seconds": float(runtime_defaults.get("backoff_initial_seconds", 2)),
        "backoff_max_seconds": float(runtime_defaults.get("backoff_max_seconds", 30)),
    }


def _p95(values: List[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return int(ordered[index])


def _progress_extra(
    *,
    total_items: int,
    resumed_items: int,
    processed_items: int,
    cache_hits: int,
    fallback_items: int,
    active_workers: int,
    started: float,
) -> str:
    elapsed = max(0.0, time.perf_counter() - started)
    throughput = (processed_items / elapsed) * 60.0 if processed_items and elapsed > 0 else 0.0
    completed = resumed_items + processed_items
    remaining = max(0, total_items - completed)
    if throughput > 0:
        eta_seconds = int((remaining / throughput) * 60.0)
        eta_text = f"{eta_seconds // 60:02d}:{eta_seconds % 60:02d}"
    else:
        eta_text = "--:--"
    return (
        f"active={active_workers} "
        f"cache={cache_hits} "
        f"fallback={fallback_items} "
        f"{throughput:.1f}/m "
        f"eta={eta_text}"
    )


def _worker_call(
    item: Any,
    worker_state: _StageWorkerState,
    worker_fn: Callable[[Any], tuple[Any, LLMRunMeta]],
) -> tuple[Any, LLMRunMeta]:
    worker_state.enter()
    try:
        return worker_fn(item)
    finally:
        worker_state.exit()


def _load_checkpoint_results(
    *,
    batch_name: str,
    step_name: str,
    prompt_version: str,
    model: str,
    result_from_dict: Callable[[Dict[str, Any]], Any],
) -> Dict[str, tuple[Any, LLMRunMeta]]:
    loaded: Dict[str, tuple[Any, LLMRunMeta]] = {}
    for row in load_stage_checkpoints(batch_name, step_name):
        try:
            meta = llm_run_meta_from_dict(row["llm_run"])
            if meta.prompt_version != prompt_version or meta.model != model:
                continue
            loaded[str(row["item_id"])] = (result_from_dict(row["result"]), meta)
        except Exception:
            continue
    return loaded


def _flush_checkpoints(batch_name: str, step_name: str, rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        append_stage_checkpoint(batch_name, step_name, row)


def run_parallel_stage(
    *,
    items: List[Any],
    stage_name: str,
    label: str,
    batch_name: str,
    prompt_version: str,
    model: str,
    max_workers: int,
    resume_enabled: bool,
    checkpoint_every: int,
    item_id_fn: Callable[[Any], str],
    worker_fn: Callable[[Any], tuple[Any, LLMRunMeta]],
    result_from_dict: Callable[[Dict[str, Any]], Any],
) -> tuple[List[Any], Dict[str, LLMRunMeta], Dict[str, Any]]:
    total_items = len(items)
    results: List[Any | None] = [None] * total_items
    run_map: Dict[str, LLMRunMeta] = {}
    resumed_items = 0
    processed_items = 0
    cache_hits = 0
    fallback_items = 0
    llm_used_items = 0
    retry_total = 0
    attempt_total = 0
    latencies: List[int] = []
    worker_state = _StageWorkerState()
    checkpoint_buffer: List[Dict[str, Any]] = []
    progress = ProgressBar(total=total_items, label=label)
    stage_started = time.perf_counter()

    checkpoint_results: Dict[str, tuple[Any, LLMRunMeta]] = {}
    if resume_enabled:
        checkpoint_results = _load_checkpoint_results(
            batch_name=batch_name,
            step_name=stage_name,
            prompt_version=prompt_version,
            model=model,
            result_from_dict=result_from_dict,
        )

    pending_items: List[tuple[int, Any, str]] = []
    for index, item in enumerate(items):
        item_id = item_id_fn(item)
        if item_id in checkpoint_results:
            result, meta = checkpoint_results[item_id]
            results[index] = result
            run_map[item_id] = meta
            resumed_items += 1
            continue
        pending_items.append((index, item, item_id))

    if resumed_items:
        progress.update(
            resumed_items,
            extra=_progress_extra(
                total_items=total_items,
                resumed_items=resumed_items,
                processed_items=processed_items,
                cache_hits=cache_hits,
                fallback_items=fallback_items,
                active_workers=0,
                started=stage_started,
            ),
        )

    if pending_items:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures: Dict[Future[tuple[Any, LLMRunMeta]], tuple[int, str]] = {
                executor.submit(_worker_call, item, worker_state, worker_fn): (index, item_id)
                for index, item, item_id in pending_items
            }
            for future in as_completed(futures):
                index, item_id = futures[future]
                result, meta = future.result()
                results[index] = result
                run_map[item_id] = meta
                processed_items += 1
                cache_hits += int(meta.cache_hit)
                fallback_items += int(meta.fallback_used)
                llm_used_items += int(meta.llm_used)
                retry_total += meta.retry_count
                attempt_total += meta.attempt_count
                latencies.append(meta.latency_ms)
                checkpoint_buffer.append(
                    {
                        "item_id": item_id,
                        "step_name": stage_name,
                        "result": serialize(result),
                        "llm_run": serialize(meta),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                if len(checkpoint_buffer) >= max(1, checkpoint_every):
                    _flush_checkpoints(batch_name, stage_name, checkpoint_buffer)
                    checkpoint_buffer = []
                active_workers, peak_workers = worker_state.snapshot()
                progress.update(
                    resumed_items + processed_items,
                    extra=_progress_extra(
                        total_items=total_items,
                        resumed_items=resumed_items,
                        processed_items=processed_items,
                        cache_hits=cache_hits,
                        fallback_items=fallback_items,
                        active_workers=active_workers,
                        started=stage_started,
                    ),
                )
            if checkpoint_buffer:
                _flush_checkpoints(batch_name, stage_name, checkpoint_buffer)

        _, peak_workers = worker_state.snapshot()
    else:
        peak_workers = 0

    progress.close(
        extra=_progress_extra(
            total_items=total_items,
            resumed_items=resumed_items,
            processed_items=processed_items,
            cache_hits=cache_hits,
            fallback_items=fallback_items,
            active_workers=0,
            started=stage_started,
        )
    )
    elapsed_seconds = round(time.perf_counter() - stage_started, 3)
    stage_stats = {
        "total_items": total_items,
        "processed_items": processed_items,
        "resumed_items": resumed_items,
        "cache_hits": cache_hits,
        "fallback_items": fallback_items,
        "llm_used_items": llm_used_items,
        "retry_total": retry_total,
        "attempt_total": attempt_total,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": _p95(latencies),
        "elapsed_seconds": elapsed_seconds,
        "peak_concurrency": max(1 if processed_items else 0, peak_workers),
    }
    log(
        f"{label} summary elapsed={elapsed_seconds:.1f}s "
        f"processed={processed_items} resumed={resumed_items} "
        f"cache={cache_hits} fallback={fallback_items} "
        f"avg={stage_stats['avg_latency_ms']}ms p95={stage_stats['p95_latency_ms']}ms "
        f"peak_workers={stage_stats['peak_concurrency']}"
    )
    return [item for item in results if item is not None], run_map, stage_stats
