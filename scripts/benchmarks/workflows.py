"""High-level workflows that combine multiple benchmarks."""

from __future__ import annotations

import logging
from typing import Optional

from . import ner, pipeline

logger = logging.getLogger(__name__)


def run_quick(samples: int, policy: str) -> None:
    """Run a compact workflow covering NER and pipeline benchmarks."""

    print("=== NER quick benchmark ===")
    ner.run_benchmarks(
        mode="both",
        text_size="medium",
        runs=2,
        preset="balanced",
        threshold=0.35,
    )

    print("\n=== Pipeline quick benchmark (DB-Bio) ===")
    config = pipeline.PipelineConfig(
        dataset="dbbio",
        split="test",
        samples=max(1, samples),
        policy=policy,
        baseline_only=False,
        rupta_only=False,
        output=None,
        rate_limit=0.0,
        print_summary=True,
    )
    pipeline.run_pipeline(config)


def handle_quick_cli(args) -> None:
    run_quick(samples=args.samples, policy=args.policy)
