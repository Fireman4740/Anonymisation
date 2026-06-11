"""Fallback shims for the AutoResearchClaw harness (``arc_pipegraph``).

``arc_pipegraph/`` is gitignored and only present on the GPU/harness machine.
On machines without it, the official runner (``eval.run_pipeline_evaluation``)
must still work: this module provides minimal, behavior-compatible stand-ins.

When ``arc_pipegraph`` is importable, the real implementations are re-exported
unchanged, so harness machines see zero difference.

Fallback semantics (documented divergences from the real harness):
- ``load_candidate`` does NOT clamp values to the candidate search space and
  does NOT force LLM/RUPTA flags on. Candidate JSON is taken as-is.
- ``compute_primary_metric`` is an unweighted mean of per-dataset ``score``
  fields (the harness may apply its own weighting).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

try:  # pragma: no cover - exercised only on harness machines
    from arc_pipegraph.objective import compute_primary_metric
    from arc_pipegraph.pipeline_adapter import (
        CandidateSpec,
        build_pipegraph_runtime_config,
        load_candidate,
    )

    HAS_ARC_PIPEGRAPH = True
except ModuleNotFoundError:
    HAS_ARC_PIPEGRAPH = False

    @dataclass(frozen=True)
    class CandidateSpec:  # type: ignore[no-redef]
        candidate_id: str
        config: Dict[str, Any] = field(default_factory=dict)
        ignored_keys: Tuple[str, ...] = ()
        warnings: Tuple[str, ...] = ()

    def load_candidate(path: str) -> "CandidateSpec":  # type: ignore[no-redef]
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Candidate file must contain a JSON object: {path}")
        return CandidateSpec(
            candidate_id=str(payload.get("candidate_id") or "candidate"),
            config=dict(payload.get("config") or {}),
        )

    def build_pipegraph_runtime_config(  # type: ignore[no-redef]
        candidate: "CandidateSpec",
        *,
        dataset_key: str,
        profile: str = "auto",
        eval_mode: str = "both",
        masking_mode: str = "benchmark",
    ) -> Dict[str, Any]:
        from eval.core.config import build_runtime_config

        return build_runtime_config(
            enable_detection=True,
            enable_deterministic=True,
            enable_ai=True,
            enable_anonymization=True,
            detection_mode="parallel",
            dataset_key=dataset_key,
            profile=profile,
            eval_mode=eval_mode,
            masking_mode=masking_mode,
            extra=candidate.config,
        )

    def compute_primary_metric(  # type: ignore[no-redef]
        dataset_results: Mapping[str, Mapping[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        weights = {key: 1.0 for key in dataset_results}
        scores = [
            float(result.get("score") or 0.0) for result in dataset_results.values()
        ]
        primary = sum(scores) / len(scores) if scores else 0.0
        return primary, {"weights": weights, "primary_metric_source": "fallback_mean"}


__all__ = [
    "HAS_ARC_PIPEGRAPH",
    "CandidateSpec",
    "load_candidate",
    "build_pipegraph_runtime_config",
    "compute_primary_metric",
]
