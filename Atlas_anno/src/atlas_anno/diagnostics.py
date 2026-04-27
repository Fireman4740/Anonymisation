from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from atlas_anno.console import log
from atlas_anno.io import read_jsonl
from atlas_anno.storage import llm_runs_path


def run_inspect_llm_runs_command(limit: int = 20) -> None:
    path = llm_runs_path()
    if not path.exists():
        log("No llm run log found")
        return

    rows = read_jsonl(path)
    if not rows:
        log("No llm run entries found")
        return

    by_step: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_step[str(row.get("step_name", "unknown"))].append(row)

    log(f"LLM runs total={len(rows)} file={path}")
    for step_name in sorted(by_step):
        items = by_step[step_name]
        latencies = [int(item.get("latency_ms", 0)) for item in items]
        retries = [int(item.get("retry_count", 0)) for item in items]
        cache_hits = sum(1 for item in items if item.get("cache_hit"))
        fallbacks = sum(1 for item in items if item.get("fallback_used"))
        llm_used = sum(1 for item in items if item.get("llm_used"))
        log(
            f"{step_name}: count={len(items)} avg_ms={round(sum(latencies) / len(latencies), 1)} "
            f"max_ms={max(latencies)} llm_used={llm_used} fallback={fallbacks} "
            f"cache_hit={cache_hits} avg_retry={round(sum(retries) / len(retries), 2)}"
        )

    log(f"Last {min(limit, len(rows))} runs:")
    for row in rows[-max(1, limit) :]:
        log(
            f"{row.get('step_name')} model={row.get('model')} latency_ms={row.get('latency_ms')} "
            f"retry={row.get('retry_count')} cache_hit={row.get('cache_hit')} "
            f"fallback={row.get('fallback_used')}"
        )
