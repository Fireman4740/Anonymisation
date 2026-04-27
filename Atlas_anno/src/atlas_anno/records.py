from __future__ import annotations

from typing import Any, Dict, List

from atlas_anno.schemas import (
    AnnotationBundle,
    AnnotationRelation,
    AnnotationSpan,
    AnonymizationResult,
    AttackCandidateScore,
    AttackResult,
    CandidatePools,
    CharacterProfile,
    DatasetBatchManifest,
    DocumentRecord,
    EvaluationReport,
    GroundedMention,
    GoldSeedAnnotation,
    GeneratedTextDraft,
    LabelStudioPrediction,
    LabelStudioTask,
    LLMRunMeta,
    PredictedAnnotation,
    PromptSpec,
    ScenarioDraft,
    ScenarioSpec,
    StageCheckpointRecord,
    StyleProfile,
    World,
    WorldDraft,
    CharacterDraft,
)


def style_profile_from_dict(payload: Dict[str, Any]) -> StyleProfile:
    return StyleProfile(**payload)


def prompt_spec_from_dict(payload: Dict[str, Any]) -> PromptSpec:
    return PromptSpec(**payload)


def llm_run_meta_from_dict(payload: Dict[str, Any]) -> LLMRunMeta:
    return LLMRunMeta(**payload)


def world_from_dict(payload: Dict[str, Any]) -> World:
    return World(**payload)


def world_draft_from_dict(payload: Dict[str, Any]) -> WorldDraft:
    return WorldDraft(**payload)


def character_from_dict(payload: Dict[str, Any]) -> CharacterProfile:
    payload = dict(payload)
    payload["style_profile"] = style_profile_from_dict(payload["style_profile"])
    return CharacterProfile(**payload)


def character_draft_from_dict(payload: Dict[str, Any]) -> CharacterDraft:
    payload = dict(payload)
    payload["style_profile"] = style_profile_from_dict(payload["style_profile"])
    return CharacterDraft(**payload)


def candidate_pools_from_dict(payload: Dict[str, Any]) -> CandidatePools:
    return CandidatePools(**payload)


def scenario_from_dict(payload: Dict[str, Any]) -> ScenarioSpec:
    return ScenarioSpec(**payload)


def scenario_draft_from_dict(payload: Dict[str, Any]) -> ScenarioDraft:
    return ScenarioDraft(**payload)


def grounded_mention_from_dict(payload: Dict[str, Any]) -> GroundedMention:
    return GroundedMention(**payload)


def generated_text_draft_from_dict(payload: Dict[str, Any]) -> GeneratedTextDraft:
    payload = dict(payload)
    payload["grounding"] = [grounded_mention_from_dict(item) for item in payload.get("grounding", [])]
    return GeneratedTextDraft(**payload)


def annotation_span_from_dict(payload: Dict[str, Any]) -> AnnotationSpan:
    return AnnotationSpan(**payload)


def annotation_relation_from_dict(payload: Dict[str, Any]) -> AnnotationRelation:
    return AnnotationRelation(**payload)


def annotation_bundle_from_dict(payload: Dict[str, Any]) -> AnnotationBundle:
    payload = dict(payload)
    payload["spans"] = [annotation_span_from_dict(item) for item in payload.get("spans", [])]
    payload["relations"] = [annotation_relation_from_dict(item) for item in payload.get("relations", [])]
    return AnnotationBundle(**payload)


def document_from_dict(payload: Dict[str, Any]) -> DocumentRecord:
    payload = dict(payload)
    payload["scenario"] = scenario_from_dict(payload["scenario"])
    payload["candidate_pools"] = candidate_pools_from_dict(payload["candidate_pools"])
    payload["annotations"] = annotation_bundle_from_dict(payload["annotations"])
    return DocumentRecord(**payload)


def anonymization_result_from_dict(payload: Dict[str, Any]) -> AnonymizationResult:
    return AnonymizationResult(**payload)


def attack_candidate_score_from_dict(payload: Dict[str, Any]) -> AttackCandidateScore:
    return AttackCandidateScore(**payload)


def attack_result_from_dict(payload: Dict[str, Any]) -> AttackResult:
    payload = dict(payload)
    payload["top_k"] = [attack_candidate_score_from_dict(item) for item in payload.get("top_k", [])]
    return AttackResult(**payload)


def evaluation_report_from_dict(payload: Dict[str, Any]) -> EvaluationReport:
    return EvaluationReport(**payload)


def gold_seed_annotation_from_dict(payload: Dict[str, Any]) -> GoldSeedAnnotation:
    payload = dict(payload)
    payload["spans"] = [annotation_span_from_dict(item) for item in payload.get("spans", [])]
    payload["relations"] = [annotation_relation_from_dict(item) for item in payload.get("relations", [])]
    return GoldSeedAnnotation(**payload)


def predicted_annotation_from_dict(payload: Dict[str, Any]) -> PredictedAnnotation:
    payload = dict(payload)
    payload["spans"] = [annotation_span_from_dict(item) for item in payload.get("spans", [])]
    payload["relations"] = [annotation_relation_from_dict(item) for item in payload.get("relations", [])]
    if payload.get("llm_run") is not None:
        payload["llm_run"] = llm_run_meta_from_dict(payload["llm_run"])
    return PredictedAnnotation(**payload)


def label_studio_prediction_from_dict(payload: Dict[str, Any]) -> LabelStudioPrediction:
    return LabelStudioPrediction(**payload)


def label_studio_task_from_dict(payload: Dict[str, Any]) -> LabelStudioTask:
    payload = dict(payload)
    payload["predictions"] = [label_studio_prediction_from_dict(item) for item in payload.get("predictions", [])]
    return LabelStudioTask(**payload)


def dataset_batch_manifest_from_dict(payload: Dict[str, Any]) -> DatasetBatchManifest:
    return DatasetBatchManifest(**payload)


def stage_checkpoint_record_from_dict(payload: Dict[str, Any]) -> StageCheckpointRecord:
    payload = dict(payload)
    payload["llm_run"] = llm_run_meta_from_dict(payload["llm_run"])
    return StageCheckpointRecord(**payload)


def worlds_from_rows(rows: List[Dict[str, Any]]) -> List[World]:
    return [world_from_dict(row) for row in rows]


def characters_from_rows(rows: List[Dict[str, Any]]) -> List[CharacterProfile]:
    return [character_from_dict(row) for row in rows]


def scenarios_from_rows(rows: List[Dict[str, Any]]) -> List[ScenarioSpec]:
    return [scenario_from_dict(row) for row in rows]


def documents_from_rows(rows: List[Dict[str, Any]]) -> List[DocumentRecord]:
    return [document_from_dict(row) for row in rows]


def anonymization_results_from_rows(rows: List[Dict[str, Any]]) -> List[AnonymizationResult]:
    return [anonymization_result_from_dict(row) for row in rows]


def attack_results_from_rows(rows: List[Dict[str, Any]]) -> List[AttackResult]:
    return [attack_result_from_dict(row) for row in rows]
