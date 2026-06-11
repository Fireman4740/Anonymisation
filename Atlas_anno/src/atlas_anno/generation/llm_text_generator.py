from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Sequence

from atlas_anno.config import load_config
from atlas_anno.constants import PROMPT_LLM_TEXT_NATURALIZER
from atlas_anno.generation.llm_generation import _default_meta, _runtime_value
from atlas_anno.generation.naturalization import (
    build_persona_block,
    build_style_directives,
    variant_order,
)
from atlas_anno.generation.surface_forms import build_surface_overrides
from atlas_anno.generation.text_generator import _complete_grounding
from atlas_anno.io import serialize
from atlas_anno.llm import OpenRouterClient
from atlas_anno.prompts import load_prompt_spec
from atlas_anno.records import document_from_dict, grounded_mention_from_dict
from atlas_anno.runtime import run_parallel_stage
from atlas_anno.schemas import (
    CharacterProfile,
    DocumentRecord,
    GeneratedTextDraft,
    GroundedMention,
    LLMRunMeta,
    World,
)
from atlas_anno.surface_grounding import find_occurrences, normalize_surface_grounding, unresolved_mentions


AUTHOR_ATTRIBUTE_LABELS = {
    "PERSON_NAME": "full_name",
    "EMAIL": "email",
    "PHONE": "phone",
    "USERNAME": "username",
    "ACCOUNT_ID": "account_id",
    "NIR": "nir_like",
    "ADDRESS": "address",
    "LOCATION": "location",
    "AGE_RANGE": "age_range",
    "NATIONALITY": "nationality",
    "DEPARTMENT": "department",
    "TEAM": "team",
    "ROLE": "role",
}


def _planned_grounding(document: DocumentRecord) -> List[GroundedMention]:
    return normalize_surface_grounding(document.metadata.get("surface_grounding", []))


def _canonical_for_label(label: str, signal_values: Dict[str, List[str]], author: CharacterProfile) -> str:
    values = signal_values.get(label, [])
    if values:
        return str(values[0])
    field_name = AUTHOR_ATTRIBUTE_LABELS.get(label)
    if field_name:
        value = getattr(author, field_name, "")
        if value:
            return str(value)
    for cue in author.contextual_cues:
        if cue.reveals_label == label:
            return cue.reveals_value
    return label


def _llm_draft_from_payload(
    payload: Any,
    document: DocumentRecord,
    author: CharacterProfile,
) -> GeneratedTextDraft:
    if not isinstance(payload, dict):
        raise ValueError("llm text response must be a JSON object")
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("llm text response missing text")
    signal_values = document.metadata.get("signal_values", {})
    mode_by_label = {entry.label: entry.difficulty_mode for entry in document.scenario.mention_plan}
    occurrence_by_snippet: Dict[str, int] = {}
    grounding: List[GroundedMention] = []
    for item in payload.get("grounding", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        if not label or not snippet:
            continue
        occurrence = int(item.get("occurrence_hint") or occurrence_by_snippet.get(snippet, 0) + 1)
        occurrence_by_snippet[snippet] = occurrence
        mention_payload = {
            "label": label,
            "canonical_value": str(item.get("canonical_value") or _canonical_for_label(label, signal_values, author)),
            "snippet": snippet,
            "occurrence_hint": occurrence,
            "difficulty_mode": str(item.get("difficulty_mode") or mode_by_label.get(label, "explicit_easy")),
            "hardness": int(item.get("hardness") or 1),
            "certainty": float(item.get("certainty") or 1.0),
            "cue_type": str(item.get("cue_type") or ""),
        }
        grounding.append(grounded_mention_from_dict(mention_payload))
    return GeneratedTextDraft(text=text, notes=list(payload.get("notes", [])), grounding=grounding)


def _drafts_from_payload(
    payload: Any,
    document: DocumentRecord,
    author: CharacterProfile,
) -> List[GeneratedTextDraft]:
    """Accepte la réponse simple ({text, ...}) ou l'enveloppe Verbalized
    Sampling ({variants: [{probability, text, ...}]})."""
    if isinstance(payload, dict) and isinstance(payload.get("variants"), list):
        drafts: List[GeneratedTextDraft] = []
        errors: List[str] = []
        for item in payload["variants"]:
            if not isinstance(item, dict):
                continue
            try:
                drafts.append(_llm_draft_from_payload(item, document, author))
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(str(exc))
        if not drafts:
            raise ValueError(f"verbalized response has no parsable variant: {errors[:3]}")
        return drafts
    return [_llm_draft_from_payload(payload, document, author)]


def _author_attribute_values(author: CharacterProfile) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {}
    for label, field_name in AUTHOR_ATTRIBUTE_LABELS.items():
        value = getattr(author, field_name, "")
        if value:
            values[label] = [str(value)]
    if author.sensitive_attributes:
        values["SENSITIVE_ATTRIBUTES"] = [str(item) for item in author.sensitive_attributes if item]
    return values


def _validate_draft(document: DocumentRecord, author: CharacterProfile, draft: GeneratedTextDraft) -> List[str]:
    issues: List[str] = []
    planned = _planned_grounding(document)
    completed = _complete_grounding(
        draft,
        document.metadata.get("signal_values", {}),
        author,
        document.scenario,
        World(
            world_id=document.world_id,
            language=document.language,
            organization_id=author.organization_id,
            organization_name=str(document.metadata.get("world_name", "")),
            departments=[],
            teams=[],
            projects=[],
            products=[],
            incidents=[],
            calendar_events=[],
        ),
    )
    for mention in planned:
        if not find_occurrences(completed.text, mention.snippet):
            issues.append(f"missing_planned_snippet:{mention.label}:{mention.snippet}")
    for mention in unresolved_mentions(completed.text, completed.grounding):
        issues.append(f"unresolved_grounding:{mention.label}:{mention.snippet}")
    for mention in completed.grounding:
        if mention.difficulty_mode in {"explicit_hard", "implicit"} and mention.canonical_value in completed.text:
            issues.append(f"difficulty_canonical_leak:{mention.label}")
        if mention.difficulty_mode == "implicit" and not mention.cue_type:
            issues.append(f"implicit_missing_cue_type:{mention.label}")

    planned_labels = {mention.label for mention in planned}
    signal_labels = set(document.metadata.get("signal_values", {}))
    allowed_labels = planned_labels | signal_labels
    for label, values in _author_attribute_values(author).items():
        if label in allowed_labels:
            continue
        for value in values:
            if value and value in completed.text:
                issues.append(f"unintended_leak:{label}")
                break
    return issues


def _prompt_payload(document: DocumentRecord, author: CharacterProfile, world: World, diagnostics: Sequence[str] | None = None) -> str:
    overrides = build_surface_overrides(document.scenario, author, world)
    planned = []
    for mention in _planned_grounding(document):
        planned.append(
            {
                "label": mention.label,
                "canonical_value": mention.canonical_value,
                "snippet": mention.snippet,
                "difficulty_mode": mention.difficulty_mode,
                "cue_type": mention.cue_type,
                "must_appear_verbatim": True,
            }
        )
    payload = {
        "document": {
            "doc_id": document.doc_id,
            "domain": document.domain,
            "unit_type": document.unit_type,
            "goal": document.scenario.document_goal,
            "recipient_role": document.scenario.recipient_role,
            "difficulty": document.scenario.difficulty,
            "urgency": document.scenario.urgency,
            "noise_level": document.scenario.noise_level,
        },
        "persona": {
            "person_id": author.person_id,
            "full_name": author.full_name,
            "role": author.role,
            "team": author.team,
            "department": author.department,
            "location": author.location,
            "age_range": author.age_range,
            "nationality": author.nationality,
            "seniority": author.seniority,
            "tenure_years": author.tenure_years,
            "degrees": author.degrees,
            "certifications": author.certifications,
            "rare_traits": author.rare_traits,
            "events": author.events,
            "sensitive_attributes": author.sensitive_attributes,
            "style_profile": serialize(author.style_profile),
            "contextual_cues": serialize(author.contextual_cues),
        },
        "world": serialize(world),
        "signal_values": document.metadata.get("signal_values", {}),
        "mention_plan": serialize(document.scenario.mention_plan),
        "surface_overrides": {label: serialize(override) for label, override in overrides.items()},
        "planned_grounding": planned,
        "forbidden_unplanned_attributes": _author_attribute_values(author),
        "repair_diagnostics": list(diagnostics or []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _error_document(document: DocumentRecord, errors: List[str]) -> DocumentRecord:
    """Marque le document en erreur sans fallback déterministe."""
    document.metadata["text_generation_mode"] = "error"
    document.metadata["generation_error"] = errors
    document.metadata["mention_difficulty"] = dict(Counter(mention.difficulty_mode for mention in _planned_grounding(document)))
    return document


def generate_llm_texts(
    documents: List[DocumentRecord],
    characters_by_id: Dict[str, CharacterProfile],
    worlds_by_id: Dict[str, World],
    client: OpenRouterClient,
    llm_mode: str,
    runtime_options: Dict[str, Any] | None = None,
) -> tuple[List[DocumentRecord], Dict[str, LLMRunMeta], Dict[str, Any]]:
    runtime_options = runtime_options or {}
    prompt_spec = load_prompt_spec(PROMPT_LLM_TEXT)
    if llm_mode != "primary-fallback":
        return (
            [_error_document(document, ["llm mode disabled"]) for document in documents],
            {document.doc_id: _default_meta("llm_text_generation", client.settings.atlas_model_creative, prompt_spec.version, "llm mode disabled") for document in documents},
            {"total_items": len(documents), "processed_items": 0, "resumed_items": 0, "cache_hits": 0, "fallback_items": 0, "error_items": len(documents), "llm_used_items": 0, "retry_total": 0, "attempt_total": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0, "elapsed_seconds": 0.0, "peak_concurrency": 0, "repair_retries": 0},
        )

    max_repairs = int(_runtime_value(runtime_options, "text_repair_retries", 2))

    def _worker(document: DocumentRecord) -> tuple[DocumentRecord, LLMRunMeta]:
        author = characters_by_id[document.author_id]
        world = worlds_by_id[document.world_id]
        fallback = GeneratedTextDraft(
            text=document.text,
            notes=list(document.metadata.get("text_notes", [])),
            grounding=_planned_grounding(document),
        )
        diagnostics: List[str] = []
        combined_meta: LLMRunMeta | None = None
        accepted: GeneratedTextDraft | None = None
        repairs_used = 0
        for repair_index in range(max_repairs + 1):
            value, meta = client.complete_json(
                step_name="llm_text_generation",
                prompt_spec=prompt_spec,
                user_prompt=_prompt_payload(document, author, world, diagnostics),
                model=client.settings.atlas_model_creative,
                validator=lambda payload: _llm_draft_from_payload(payload, document, author),
                fallback_value=serialize(fallback),
                temperature=0.7,
                allow_fallback=False,
            )
            if combined_meta is None:
                combined_meta = meta
            else:
                combined_meta.llm_used = combined_meta.llm_used or meta.llm_used
                combined_meta.fallback_used = combined_meta.fallback_used and meta.fallback_used
                combined_meta.error = combined_meta.error and meta.error
                combined_meta.retry_count += meta.retry_count
                combined_meta.attempt_count += meta.attempt_count
                combined_meta.queue_wait_ms += meta.queue_wait_ms
                combined_meta.validation_errors.extend(meta.validation_errors)
                combined_meta.latency_ms += meta.latency_ms
                combined_meta.estimated_cost += meta.estimated_cost
            # Sur erreur LLM (réponse nulle / HTTP fail) : pas la peine de retenter via la boucle interne.
            if meta.error and value is None:
                break
            if value is None:
                break
            try:
                draft = value if isinstance(value, GeneratedTextDraft) else _llm_draft_from_payload(value, document, author)
            except (ValueError, TypeError, KeyError) as exc:
                combined_meta.validation_errors.append(f"draft parse error: {exc}")
                break
            issues = _validate_draft(document, author, draft)
            if not issues:
                accepted = draft
                break
            diagnostics = issues
            repairs_used = repair_index + 1

        if combined_meta is None:
            combined_meta = _default_meta("llm_text_generation", client.settings.atlas_model_creative, prompt_spec.version, "no generation attempt")
            combined_meta.error = True

        if accepted is None:
            # Aucun fallback déterministe : document marqué en erreur.
            combined_meta.error = True
            combined_meta.fallback_used = False
            _error_document(document, combined_meta.validation_errors)
        else:
            document.metadata["text_generation_mode"] = "llm" if combined_meta.llm_used and not combined_meta.error else "error"
            completed = _complete_grounding(accepted, document.metadata.get("signal_values", {}), author, document.scenario, world)
            document.text = completed.text
            document.metadata["surface_grounding"] = [serialize(mention) for mention in completed.grounding]
            document.metadata["text_notes"] = list(completed.notes)
            document.metadata["mention_difficulty"] = dict(Counter(mention.difficulty_mode for mention in completed.grounding))

        document.metadata["text_repair_retries"] = min(repairs_used, max_repairs)
        return document, combined_meta

    documents_out, run_map, stage_stats = run_parallel_stage(
        items=documents,
        stage_name="llm_text_generation",
        label="llm-texts",
        batch_name=str(_runtime_value(runtime_options, "batch_name", "pilot_100")),
        prompt_version=prompt_spec.version,
        model=client.settings.atlas_model_creative,
        max_workers=int(_runtime_value(runtime_options, "creative_workers", 8)),
        resume_enabled=bool(_runtime_value(runtime_options, "resume_enabled", True)),
        checkpoint_every=int(_runtime_value(runtime_options, "checkpoint_every", 1)),
        item_id_fn=lambda document: document.doc_id,
        worker_fn=_worker,
        result_from_dict=document_from_dict,
    )
    stage_stats["repair_retries"] = sum(int(document.metadata.get("text_repair_retries", 0)) for document in documents_out)
    return documents_out, run_map, stage_stats
