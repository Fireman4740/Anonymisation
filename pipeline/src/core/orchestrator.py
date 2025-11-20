
"""High level orchestration of the anonymisation pipeline."""

from __future__ import annotations

import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .policy import AnonymizationPolicy, preset
from ..config.config_loader import get_model_overrides, load_config
from ..services.detection.detection import (
    DetectedEntity,
    DetectionService,
    create_detection_service,
)
from ..services.generalization.generalizer import Generalization, GeneralizationService
from ..services.llm.llm_pipeline import LLMPipelineService, create_llm_pipeline
from ..utils.utils_pseudo import PseudoMapper
from ..utils.validation import count_placeholder_types, validate_anonymization


def anonymize_text(
    value: str,
    scope_id: str,
    secret_salt: str,
    level: str = "L1",
    ner_results: Optional[List[Dict[str, Any]]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    detection_service: Optional[DetectionService] = None,
    generalization_service: Optional[GeneralizationService] = None,
    llm_service: Optional[LLMPipelineService] = None,
) -> Dict[str, Any]:
    """Anonymise *value* according to the selected policy."""

    overrides = overrides or {}

    try:
        policy = preset(level)
    except Exception:
        policy = preset("L0")

    applied_policy_overrides = _apply_policy_overrides(policy, overrides)
    pseudo_mapper = PseudoMapper(secret=secret_salt, scope_id=scope_id)

    detection_service = detection_service or create_detection_service(policy, overrides)
    generalization_service = generalization_service or GeneralizationService(policy)

    llm_models = _resolve_llm_models(overrides)
    if llm_service is None and (
        policy.llm_detection
        or policy.llm_paraphrase
        or policy.llm_audit
        or policy.rupta_enabled
    ):
        llm_service = create_llm_pipeline(policy, llm_models)

    skip_regex_tags = _normalise_tags(overrides.get("skip_regex_tags"))

    warnings: List[str] = []
    errors: List[str] = []

    try:
        entities = detection_service.detect_all(
            text=value,
            skip_regex_tags=skip_regex_tags,
            external_ner=ner_results,
        )
    except Exception as exc:
        return _error_payload(value, level, exc)

    anonymized_text, replacements = _apply_pseudonymization(value, entities, pseudo_mapper)
    generalized_text, generalizations = generalization_service.apply_all(anonymized_text)

    text = generalized_text
    baseline_text = generalized_text
    placeholder_counts = count_placeholder_types(generalized_text)
    expected_counts = _prepare_expected_counts(
        placeholder_counts,
        overrides.get("expected_placeholder_counts"),
    )
    enforce_counts = bool(
        overrides.get("rupta_preserve_entity_counts", policy.rupta_preserve_entity_counts)
        or overrides.get("paraphrase_preserve_multiplicity", policy.paraphrase_preserve_multiplicity)
    )
    paraphrase_applied = False
    audit_report: Optional[Dict[str, Any]] = None
    rupta_metrics: Optional[Dict[str, Any]] = None
    llm_errors: List[str] = []

    if llm_service:
        text, paraphrase_applied, llm_errors = _apply_paraphrase_if_needed(
            text,
            policy,
            llm_service,
            overrides,
            expected_counts,
        )

        audit_report, audit_err = _run_audit_if_needed(text, policy, llm_service)
        if audit_err:
            llm_errors.append(audit_err)

        rupta_metrics, text, rupta_err = _run_rupta_if_needed(
            original=value,
            current=text,
            policy=policy,
            llm_service=llm_service,
            overrides=overrides,
        )
        if rupta_err:
            llm_errors.append(rupta_err)

    validation_issues = validate_anonymization(
        original=value,
        anonymized=text,
        expected_counts=expected_counts,
        forbidden_patterns=overrides.get("forbidden_patterns"),
    )
    if validation_issues:
        warnings.extend(validation_issues)
        if enforce_counts:
            if text != baseline_text:
                llm_errors.append("Output reverted after validation failure")
            text = baseline_text
            paraphrase_applied = False
            rupta_metrics = None

    mappings = _build_mappings(replacements)

    audit_section: Dict[str, Any] = {
        "entities": [_entity_to_dict(e) for e in entities],
        "replacements": replacements,
        "generalizations": [_generalization_to_dict(g) for g in generalizations],
        "paraphrase_applied": paraphrase_applied,
        "rupta_applied": bool(rupta_metrics),
        "mappings": mappings,
        "llm_errors": llm_errors,
        "metadata": {
            "policy_overrides": applied_policy_overrides,
            "skip_regex_tags": sorted(skip_regex_tags),
            "llm_models": llm_service.models if llm_service else llm_models or {},
        },
    }

    if audit_report:
        audit_section["llm_audit"] = audit_report
    if rupta_metrics:
        audit_section["rupta_metrics"] = rupta_metrics

    if policy.mapping_retention == "discard":
        audit_section.pop("mappings", None)

    warnings.extend(llm_errors)

    metrics = {
        "entities_detected": len(entities),
        "entities_replaced": len(replacements),
        "generalizations": len(generalizations),
        "length_before": len(value),
        "length_after": len(text),
    }

    evaluation = {
        "is_valid": not errors,
        "metrics": metrics,
        "validation_errors": errors,
        "warnings": warnings,
    }

    return {
        "anonymized_text": text,
        "audit": audit_section,
        "evaluation": evaluation,
        "policy": policy.to_dict(),
    }


def _apply_paraphrase_if_needed(
    text: str,
    policy: AnonymizationPolicy,
    llm_service: LLMPipelineService,
    overrides: Dict[str, Any],
    expected_counts: Dict[str, int],
) -> Tuple[str, bool, List[str]]:
    if not policy.llm_paraphrase:
        return text, False, []

    intensity = int(overrides.get("paraphrase_intensity", policy.paraphrase_intensity) or 0)
    temperature = overrides.get("paraphrase_temperature")
    if temperature is None:
        temperature = _intensity_to_temperature(intensity)
    else:
        try:
            temperature = float(temperature)
        except Exception:
            temperature = _intensity_to_temperature(intensity)

    preserve_multiplicity = bool(
        overrides.get(
            "paraphrase_preserve_multiplicity",
            policy.paraphrase_preserve_multiplicity,
        )
    )

    paraphrased, err = llm_service.paraphrase(
        text,
        temperature=temperature,
        ensure_placeholders_preserved=True,
        preserve_multiplicity=preserve_multiplicity,
        expected_counts=expected_counts if preserve_multiplicity else None,
        intensity=intensity,
    )

    if err:
        return text, False, [err]

    return paraphrased, paraphrased != text, []


def _run_audit_if_needed(
    text: str,
    policy: AnonymizationPolicy,
    llm_service: LLMPipelineService,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not policy.llm_audit:
        return None, None

    report, err = llm_service.audit(text)
    if err:
        return None, err
    return report, None


def _run_rupta_if_needed(
    original: str,
    current: str,
    policy: AnonymizationPolicy,
    llm_service: Optional[LLMPipelineService],
    overrides: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    if not (policy.rupta_enabled and llm_service):
        return None, current, None

    ground_truth_people = overrides.get("rupta_ground_truth_people")
    ground_truth_label = overrides.get("rupta_ground_truth_label")

    if not ground_truth_people or not ground_truth_label:
        return None, current, None

    result, err = llm_service.optimize_with_rupta(
        original_text=original,
        initial_anonymized_text=current,
        ground_truth_people=ground_truth_people,
        ground_truth_label=ground_truth_label,
    )

    if err:
        return None, current, err

    metrics = {
        "privacy": result.privacy_score,
        "utility": result.utility_score,
        "iterations": result.iterations,
        "converged": result.converged,
    }
    return metrics, result.final_text, None


def _build_mappings(replacements: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mappings: Dict[str, Dict[str, Any]] = {}
    for repl in replacements:
        placeholder = repl.get("replacement")
        if not isinstance(placeholder, str):
            continue
        mappings.setdefault(
            placeholder,
            {
                "surface": repl.get("surface", ""),
                "etype": repl.get("etype", ""),
                "source": repl.get("source", ""),
            },
        )
    return mappings


def _apply_pseudonymization(
    text: str,
    entities: Iterable[DetectedEntity],
    mapper: PseudoMapper,
) -> Tuple[str, List[Dict[str, Any]]]:
    replacements: List[Dict[str, Any]] = []
    pieces: List[str] = []
    cursor = 0

    for entity in sorted(entities, key=lambda e: (e.start, e.end)):
        placeholder = mapper.placeholder(entity.etype, entity.surface)
        pieces.append(text[cursor:entity.start])
        pieces.append(placeholder)
        replacements.append(
            {
                "start": entity.start,
                "end": entity.end,
                "surface": entity.surface,
                "replacement": placeholder,
                "etype": entity.etype,
                "source": entity.source,
                "score": entity.score,
                "metadata": dict(entity.metadata),
            }
        )
        cursor = entity.end

    pieces.append(text[cursor:])
    return "".join(pieces), replacements


def _entity_to_dict(entity: DetectedEntity) -> Dict[str, Any]:
    return {
        "start": entity.start,
        "end": entity.end,
        "surface": entity.surface,
        "etype": entity.etype,
        "source": entity.source,
        "score": entity.score,
        "metadata": dict(entity.metadata),
    }


def _generalization_to_dict(generalization: Generalization) -> Dict[str, Any]:
    return {
        "start": generalization.start,
        "end": generalization.end,
        "surface": generalization.surface,
        "replacement": generalization.replacement,
        "etype": generalization.etype,
        "policy_rule": generalization.policy_rule,
    }


def _apply_policy_overrides(
    policy: AnonymizationPolicy,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    applied: Dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(policy, key):
            continue
        try:
            setattr(policy, key, value)
            applied[key] = value
        except Exception:
            continue
    return applied


def _resolve_llm_models(overrides: Dict[str, Any]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    try:
        cfg = load_config()
        models.update(get_model_overrides(cfg))
    except Exception:
        pass

    override_models = overrides.get("llm_models")
    if isinstance(override_models, dict):
        models.update({k: v for k, v in override_models.items() if isinstance(v, str) and v})
    return models


def _normalise_tags(raw: Any) -> set[str]:
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {str(tag).upper() for tag in raw if tag is not None}


def _prepare_expected_counts(
    placeholder_counts: Dict[str, int],
    extra_counts: Any,
) -> Dict[str, int]:
    expected: Dict[str, int] = {}
    for key, value in placeholder_counts.items():
        expected[key] = value
        expected[f"[{key}_"] = value
    if isinstance(extra_counts, dict):
        for k, v in extra_counts.items():
            expected[str(k)] = v
    return expected


def _intensity_to_temperature(intensity: int) -> float:
    mapping = {0: 0.2, 1: 0.35, 2: 0.55, 3: 0.75}
    return mapping.get(max(0, min(intensity, 3)), 0.35)


def _error_payload(value: str, level: str, exc: Exception) -> Dict[str, Any]:
    return {
        "anonymized_text": value,
        "audit": {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "entities": [],
        },
        "evaluation": {
            "is_valid": False,
            "metrics": {},
            "validation_errors": [str(exc)],
            "warnings": [],
        },
        "policy": preset(level).to_dict() if level else {},
    }


anonymize_text_refactored = anonymize_text


__all__ = ["anonymize_text", "anonymize_text_refactored"]
