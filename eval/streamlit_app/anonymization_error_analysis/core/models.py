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
    profile: str
    eval_mode: str
    masking_mode: str
    llm_detection_enabled: bool
    llm_audit_enabled: bool
    llm_paraphrase_enabled: bool
    rupta_enabled: bool
    rupta_max_iterations: int
    rupta_p_threshold: int
    llm_model: str
    llm_provider: str
    detection_threshold: float
    paraphrase_intensity: int
    doc_workers: Optional[int]
    save_run: bool
    run_name: str


@dataclass(frozen=True)
class RATBenchEvalConfig:
    language: str
    level: Optional[int]  # None = tous les niveaux
    limit: int
    run_full_dataset: bool
    enable_detection: bool
    enable_deterministic: bool
    enable_ai: bool
    enable_anonymization: bool
    detection_mode: str
    profile: str
    eval_mode: str
    masking_mode: str
    # LLM / RUPTA
    llm_detection_enabled: bool
    llm_audit_enabled: bool
    llm_paraphrase_enabled: bool
    rupta_enabled: bool
    rupta_max_iterations: int
    rupta_p_threshold: int
    llm_model: str
    llm_provider: str
    detection_threshold: float
    paraphrase_intensity: int
    doc_workers: Optional[int]
    # Risk re-identification
    enable_risk_eval: bool
    save_run: bool
    run_name: str


@dataclass(frozen=True)
class BenchmarkEvalConfig:
    type: str
    ratbench_config: Optional[RATBenchEvalConfig]
    local_config: Optional[LocalEvalConfig]


@dataclass(frozen=True)
class AblationConfig:
    run_new: bool
    dataset_kind: str
    limit: int
    suite: str
    save_run: bool


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
