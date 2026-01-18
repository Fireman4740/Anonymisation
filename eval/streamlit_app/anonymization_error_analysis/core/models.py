from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SourceSelection:
    source: str


@dataclass(frozen=True)
class LocalEvalConfig:
    dataset_kind: str
    dataset_label: str
    dataset_path: str
    split: Optional[str]
    run_full_dataset: bool
    limit: int
    enable_detection: bool
    enable_deterministic: bool
    enable_ai: bool
    enable_anonymization: bool
    detection_mode: str
    out_path: str
    save_run: bool
    run_name: str


@dataclass(frozen=True)
class RunSummary:
    path: str
    created_at: Optional[str]
    created_dt: Optional[Any]
    pipeline: Optional[str]
    dataset: Optional[str]
    run_name: Optional[str]
    limit: Optional[int]
    config: Optional[Dict[str, Any]]
    avg_prec: float
    avg_rec: float
    avg_f2: float
    leaky_docs: int
    total_docs: int


@dataclass(frozen=True)
class RunsFilter:
    run_paths: List[str]
    selected_path: Optional[str]
    start_date: Optional[Any]
    end_date: Optional[Any]
    dataset_filter: str
    config_contains: str
    config_key: str
    config_value: str


@dataclass(frozen=True)
class ReportSelection:
    report_paths: List[str]
    selected_path: Optional[str]


@dataclass(frozen=True)
class DocFilter:
    recall_range: Tuple[float, float]
    show_leaks_only: bool


@dataclass(frozen=True)
class DocSelection:
    selected_doc_id: Optional[str]


@dataclass(frozen=True)
class ReportMeta:
    title: str
    subtitle: Optional[str] = None
    source: Optional[str] = None
    dataset_name: Optional[str] = None
    path: Optional[str] = None
