from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def _require(value: Any, name: str) -> None:
    if value is None or value == "":
        raise ValueError(f"{name} must not be empty")


def _non_negative(value: int | float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass
class StyleProfile:
    formality: str
    signature_pattern: str
    verbosity: str
    emoji_usage: str
    favorite_connectors: List[str] = field(default_factory=list)
    jargon_pattern: str = ""

    def __post_init__(self) -> None:
        _require(self.formality, "formality")
        _require(self.signature_pattern, "signature_pattern")


@dataclass
class PromptSpec:
    prompt_name: str
    system_prompt: str
    version: str
    path: str


@dataclass
class LLMRunMeta:
    step_name: str
    model: str
    prompt_version: str
    llm_used: bool
    fallback_used: bool
    retry_count: int
    attempt_count: int = 0
    queue_wait_ms: int = 0
    cache_hit: bool = False
    validation_errors: List[str] = field(default_factory=list)
    latency_ms: int = 0
    estimated_cost: float = 0.0
    raw_response_excerpt: str = ""


@dataclass
class World:
    world_id: str
    language: str
    organization_id: str
    organization_name: str
    departments: List[str]
    teams: List[str]
    projects: List[str]
    products: List[str]
    incidents: List[str]
    calendar_events: List[str]

    def __post_init__(self) -> None:
        _require(self.world_id, "world_id")
        _require(self.organization_id, "organization_id")


@dataclass
class WorldDraft:
    organization_name: str
    departments: List[str]
    teams: List[str]
    projects: List[str]
    products: List[str]
    incidents: List[str]
    calendar_events: List[str]


@dataclass
class CharacterProfile:
    person_id: str
    full_name: str
    email: str
    phone: str
    username: str
    account_id: str
    language: str
    country: str
    location: str
    age_range: str
    gender: str
    nationality: str
    organization_id: str
    department: str
    team: str
    role: str
    seniority: str
    tenure_years: int
    degrees: List[str]
    skills: List[str]
    certifications: List[str]
    rare_traits: List[str]
    events: List[str]
    sensitive_attributes: List[str]
    style_profile: StyleProfile

    def __post_init__(self) -> None:
        _require(self.person_id, "person_id")
        _require(self.full_name, "full_name")
        _non_negative(self.tenure_years, "tenure_years")


@dataclass
class CharacterDraft:
    full_name: str
    country: str
    location: str
    age_range: str
    nationality: str
    department: str
    team: str
    role: str
    seniority: str
    tenure_years: int
    degrees: List[str]
    skills: List[str]
    certifications: List[str]
    rare_traits: List[str]
    events: List[str]
    sensitive_attributes: List[str]
    style_profile: StyleProfile


@dataclass
class CandidatePools:
    public: List[str] = field(default_factory=list)
    org_internal: List[str] = field(default_factory=list)
    insider: List[str] = field(default_factory=list)


@dataclass
class ScenarioSpec:
    scenario_id: str
    domain: str
    unit_type: str
    language: str
    author_id: str
    recipient_role: str
    document_goal: str
    difficulty: str
    required_signals: List[str] = field(default_factory=list)
    implicit_signals: List[str] = field(default_factory=list)
    include_signature: bool = False
    include_direct_identifiers: bool = False
    include_sensitive: bool = False
    urgency: str = "medium"
    noise_level: str = "medium"
    split: str = "train"

    def __post_init__(self) -> None:
        _require(self.scenario_id, "scenario_id")
        _require(self.author_id, "author_id")


@dataclass
class ScenarioDraft:
    unit_type: str
    recipient_role: str
    document_goal: str
    difficulty: str
    required_signals: List[str] = field(default_factory=list)
    implicit_signals: List[str] = field(default_factory=list)
    include_signature: bool = False
    include_direct_identifiers: bool = False
    include_sensitive: bool = False
    urgency: str = "medium"
    noise_level: str = "medium"
    split: str = "train"


@dataclass
class GroundedMention:
    label: str
    canonical_value: str
    snippet: str
    occurrence_hint: int = 1

    def __post_init__(self) -> None:
        _require(self.label, "label")
        _require(self.canonical_value, "canonical_value")
        _require(self.snippet, "snippet")
        if self.occurrence_hint < 1:
            raise ValueError("occurrence_hint must be >= 1")


@dataclass
class GeneratedTextDraft:
    text: str
    notes: List[str] = field(default_factory=list)
    grounding: List[GroundedMention] = field(default_factory=list)


@dataclass
class AnnotationSpan:
    start: int
    end: int
    label: str
    text: str
    confidence: float = 1.0
    source: str = "gold"

    def __post_init__(self) -> None:
        _non_negative(self.start, "start")
        _non_negative(self.end, "end")
        if self.end < self.start:
            raise ValueError("end must be >= start")


@dataclass
class AnnotationRelation:
    relation_type: str
    source_label: str
    target_label: str
    note: str = ""


@dataclass
class AnnotationBundle:
    spans: List[AnnotationSpan] = field(default_factory=list)
    relations: List[AnnotationRelation] = field(default_factory=list)
    doc_labels: Dict[str, Any] = field(default_factory=dict)
    human_review_required: bool = False


@dataclass
class GoldSeedAnnotation:
    spans: List[AnnotationSpan] = field(default_factory=list)
    relations: List[AnnotationRelation] = field(default_factory=list)
    doc_labels: Dict[str, Any] = field(default_factory=dict)
    source: str = "gold_seed"


@dataclass
class PredictedAnnotation:
    spans: List[AnnotationSpan] = field(default_factory=list)
    relations: List[AnnotationRelation] = field(default_factory=list)
    doc_labels: Dict[str, Any] = field(default_factory=dict)
    source: str = "predicted"
    llm_run: LLMRunMeta | None = None


@dataclass
class DocumentRecord:
    doc_id: str
    domain: str
    unit_type: str
    language: str
    author_id: str
    target_person_ids: List[str]
    world_id: str
    split: str
    text: str
    scenario: ScenarioSpec
    candidate_pools: CandidatePools
    annotations: AnnotationBundle
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnonymizationResult:
    doc_id: str
    strategy: str
    anonymized_text: str
    actions_performed: List[str]
    rationale: str
    estimated_privacy_gain: float
    estimated_utility_loss: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackCandidateScore:
    person_id: str
    score: float
    matched_signals: List[str] = field(default_factory=list)


@dataclass
class AttackResult:
    doc_id: str
    attacker_type: str
    top_k: List[AttackCandidateScore]
    best_person_id: str
    confidence: float
    candidate_pool_size: int
    matched_signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    meta: Dict[str, Any]
    summary: Dict[str, Any]
    details: List[Dict[str, Any]]


@dataclass
class LabelStudioPrediction:
    model_version: str
    score: float
    result: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LabelStudioTask:
    id: str
    data: Dict[str, Any]
    predictions: List[LabelStudioPrediction] = field(default_factory=list)


@dataclass
class DatasetBatchManifest:
    batch_name: str
    worlds_total: int
    characters_total: int
    documents_total: int
    llm_mode: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageCheckpointRecord:
    item_id: str
    step_name: str
    result: Any
    llm_run: LLMRunMeta
    completed_at: str
