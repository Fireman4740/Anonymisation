from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from eval.core.metrics import (
    gold_text_leakage,
    runtime_metrics,
    score_dataset_axes,
    span_detection_axes,
    utility_preservation_stub,
)

Span = Tuple[int, int, str]
DatasetDoc = Tuple[str, str, List[Span]]
DocumentReport = List[Dict[str, Any]]
LoadBenchmarkDocsFn = Callable[..., Tuple[List[Any], str]]
LoadRatbenchProfilesFn = Callable[..., List[Dict[str, Any]]]


@dataclass(frozen=True)
class DatasetRunRequest:
    dataset: str
    dataset_key: str
    split: str = "test"
    language: str = "english"
    level: Optional[int] = None
    limit: Optional[int] = None


@dataclass
class DatasetEvaluationContext:
    args: argparse.Namespace
    root: str
    output_dir: Path
    pipeline: Any
    create_initial_state: Any
    load_benchmark_docs: LoadBenchmarkDocsFn
    load_ratbench_profiles: LoadRatbenchProfilesFn
    openrouter_key_available: Callable[[str], bool]


@dataclass(frozen=True)
class DatasetAdapter:
    """Dataset-specific contract used by the official evaluation runner."""

    name: str
    description: str
    supports: Dict[str, bool]
    default_split: str = "test"
    aliases: Tuple[str, ...] = ()
    extra_load_kwargs: Dict[str, Any] = field(default_factory=dict)
    loader: Optional[Callable[..., Tuple[List[Any], str]]] = None

    def expand_run_requests(self, args: argparse.Namespace) -> List[DatasetRunRequest]:
        split = str(getattr(args, "split", self.default_split) or self.default_split)
        return [
            DatasetRunRequest(
                dataset=self.name,
                dataset_key=self.name,
                split=split,
                language=str(getattr(args, "language", "english") or "english"),
                limit=getattr(args, "limit", None),
            )
        ]

    def load(
        self,
        request: Optional[DatasetRunRequest] = None,
        context: Optional[DatasetEvaluationContext] = None,
        *,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[List[Any], str]:
        if request is None:
            request = DatasetRunRequest(
                dataset=self.name,
                dataset_key=self.name,
                split=split or self.default_split,
                limit=limit,
            )
        load_kwargs: Dict[str, Any] = {**self.extra_load_kwargs, **kwargs}
        if self.loader is not None:
            return self.loader(split=request.split, limit=request.limit, **load_kwargs)
        if context is None:
            from eval.core.bootstrap import project_root
            from eval.core.datasets import load_benchmark_docs

            return load_benchmark_docs(
                dataset=self.name,
                project_root=project_root(),
                limit=request.limit,
                split=request.split,
                language=request.language,
                level=request.level,
                **load_kwargs,
            )
        return context.load_benchmark_docs(
            dataset=self.name,
            project_root=context.root,
            limit=request.limit,
            split=request.split,
            language=request.language,
            level=request.level,
            **load_kwargs,
        )

    def validate(self) -> None:
        docs, _ = self.load(limit=1)
        if not docs:
            raise ValueError(f"Dataset {self.name!r} loaded zero documents")

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "supports": dict(self.supports),
            "default_split": self.default_split,
            "aliases": sorted(self.aliases),
        }

    def protocol_metadata(
        self,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
    ) -> Dict[str, Any]:
        return {
            "name": self.name,
            "protocol": "unknown",
            "annotation_status": "unknown",
            "warning": None,
        }

    def enrich_report(
        self,
        report: DocumentReport,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
        runtime_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {}

    def build_axes(
        self,
        report: DocumentReport,
        aggregate: Dict[str, Any],
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
        runtime_config: Mapping[str, Any],
        elapsed_s: float,
        enrichment: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        protocol = self.protocol_metadata(request, context)
        span_axis = span_detection_axes(report, aggregate)
        leakage_axis = gold_text_leakage(report)
        utility_axis = utility_preservation_stub(self.name, records_meta=protocol)
        runtime_axis = runtime_metrics(report, elapsed_s)
        score_details = score_dataset_axes(
            dataset=self.name,
            span_detection=span_axis,
            anonymization_leakage=leakage_axis,
            runtime=runtime_axis,
            utility_preservation=utility_axis,
        )
        return (
            {
                "span_detection": span_axis,
                "anonymization_leakage": leakage_axis,
                "utility_preservation": utility_axis,
                "runtime": runtime_axis,
            },
            score_details,
        )
