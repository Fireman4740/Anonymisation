from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Type, TypeVar

from atlas_anno.schemas import (
    AnnotationBundle,
    AnnotationRelation,
    AnnotationSpan,
    AnonymizationResult,
    AttackPair,
    AttackCandidateScore,
    AttackResult,
    AuxiliaryKnowledge,
    CandidatePools,
    CharacterProfile,
    ContextualCue,
    DatasetBatchManifest,
    DocumentRecord,
    EvaluationReport,
    GroundedMention,
    GoldSeedAnnotation,
    GeneratedTextDraft,
    LabelStudioPrediction,
    LabelStudioTask,
    LLMRunMeta,
    MentionPlanEntry,
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


_T = TypeVar("_T")


def _known_fields(cls: Type[_T], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Filtre le payload sur les champs déclarés de la dataclass (compat avant).

    Les clés inconnues (écrites par une version de schéma plus récente) sont
    conservées dans payload["metadata"]["_unknown_fields"] si la dataclass
    expose un champ metadata, sinon ignorées.
    """
    names = {item.name for item in dataclasses.fields(cls)}
    known = {key: value for key, value in payload.items() if key in names}
    unknown = {key: value for key, value in payload.items() if key not in names}
    if unknown and "metadata" in names:
        metadata = dict(known.get("metadata") or {})
        metadata.setdefault("_unknown_fields", {}).update(unknown)
        known["metadata"] = metadata
    return known


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    normalized: List[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            preferred = item.get("name") or item.get("label") or item.get("title") or next(
                (candidate for candidate in item.values() if isinstance(candidate, str)),
                "",
            )
            text = str(preferred).strip()
        else:
            text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def style_profile_from_dict(payload: Dict[str, Any]) -> StyleProfile:
    return StyleProfile(**_known_fields(StyleProfile, payload))


def prompt_spec_from_dict(payload: Dict[str, Any]) -> PromptSpec:
    return PromptSpec(**payload)


def llm_run_meta_from_dict(payload: Dict[str, Any]) -> LLMRunMeta:
    return LLMRunMeta(**_known_fields(LLMRunMeta, payload))


def world_from_dict(payload: Dict[str, Any]) -> World:
    payload = dict(payload)
    for key in ("departments", "teams", "projects", "products", "incidents", "calendar_events"):
        payload[key] = _string_list(payload.get(key, []))
    return World(**_known_fields(World, payload))


def world_draft_from_dict(payload: Dict[str, Any]) -> WorldDraft:
    payload = dict(payload)
    for key in ("departments", "teams", "projects", "products", "incidents", "calendar_events"):
        payload[key] = _string_list(payload.get(key, []))
    return WorldDraft(**payload)


def contextual_cue_from_dict(payload: Dict[str, Any]) -> ContextualCue:
    return ContextualCue(**_known_fields(ContextualCue, payload))


def character_from_dict(payload: Dict[str, Any]) -> CharacterProfile:
    payload = dict(payload)
    payload["style_profile"] = style_profile_from_dict(payload["style_profile"])
    if payload.get("contextual_cues"):
        payload["contextual_cues"] = [contextual_cue_from_dict(item) for item in payload["contextual_cues"]]
    return CharacterProfile(**_known_fields(CharacterProfile, payload))


def character_draft_from_dict(payload: Dict[str, Any]) -> CharacterDraft:
    payload = dict(payload)
    payload["style_profile"] = style_profile_from_dict(payload["style_profile"])
    return CharacterDraft(**payload)


def candidate_pools_from_dict(payload: Dict[str, Any]) -> CandidatePools:
    return CandidatePools(**payload)


def mention_plan_entry_from_dict(payload: Dict[str, Any]) -> MentionPlanEntry:
    return MentionPlanEntry(**_known_fields(MentionPlanEntry, payload))


def scenario_from_dict(payload: Dict[str, Any]) -> ScenarioSpec:
    payload = dict(payload)
    if payload.get("mention_plan"):
        payload["mention_plan"] = [mention_plan_entry_from_dict(item) for item in payload["mention_plan"]]
    return ScenarioSpec(**_known_fields(ScenarioSpec, payload))


def scenario_draft_from_dict(payload: Dict[str, Any]) -> ScenarioDraft:
    return ScenarioDraft(**payload)


def grounded_mention_from_dict(payload: Dict[str, Any]) -> GroundedMention:
    return GroundedMention(**_known_fields(GroundedMention, payload))


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
    return DocumentRecord(**_known_fields(DocumentRecord, payload))


def anonymization_result_from_dict(payload: Dict[str, Any]) -> AnonymizationResult:
    return AnonymizationResult(**payload)


def attack_candidate_score_from_dict(payload: Dict[str, Any]) -> AttackCandidateScore:
    return AttackCandidateScore(**payload)


def attack_result_from_dict(payload: Dict[str, Any]) -> AttackResult:
    payload = dict(payload)
    payload["top_k"] = [attack_candidate_score_from_dict(item) for item in payload.get("top_k", [])]
    return AttackResult(**payload)


def auxiliary_knowledge_from_dict(payload: Dict[str, Any]) -> AuxiliaryKnowledge:
    return AuxiliaryKnowledge(**_known_fields(AuxiliaryKnowledge, payload))


def attack_pair_from_dict(payload: Dict[str, Any]) -> AttackPair:
    payload = dict(payload)
    payload["aux_knowledge"] = auxiliary_knowledge_from_dict(payload.get("aux_knowledge", {"level": "none"}))
    return AttackPair(**_known_fields(AttackPair, payload))


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
    return DatasetBatchManifest(**_known_fields(DatasetBatchManifest, payload))


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


def attack_pairs_from_rows(rows: List[Dict[str, Any]]) -> List[AttackPair]:
    return [attack_pair_from_dict(row) for row in rows]
