from __future__ import annotations

import json
import re
from typing import Dict, List

from atlas_anno.config import load_config
from atlas_anno.console import log
from atlas_anno.constants import PROMPT_PREANNOTATION
from atlas_anno.generation.world_builder import COMPANIES, EVENTS, INCIDENTS, PRODUCTS, PROJECTS, TEAMS
from atlas_anno.io import serialize
from atlas_anno.llm import OpenRouterClient
from atlas_anno.prompts import load_prompt_spec
from atlas_anno.records import document_from_dict
from atlas_anno.runtime import build_runtime_options, run_parallel_stage
from atlas_anno.schemas import (
    AnnotationBundle,
    AnnotationRelation,
    AnnotationSpan,
    DocumentRecord,
    GoldSeedAnnotation,
    GroundedMention,
    LLMRunMeta,
    PredictedAnnotation,
)
from atlas_anno.settings import load_settings
from atlas_anno.storage import annotations_path, load_batch_manifest, load_documents, save_batch_manifest, save_documents
from atlas_anno.surface_grounding import document_surface_grounding, find_occurrences, resolve_grounded_mention


DIRECT_LABELS = {"PERSON_NAME", "EMAIL", "PHONE", "USERNAME", "ACCOUNT_ID", "ORG_NAME_STRONG", "PROJECT_NAME_STRONG"}
SENSITIVE_LABELS = {"HEALTH", "ETHNICITY", "RELIGION", "DISABILITY", "FAMILY_STATUS", "SEXUAL_ORIENTATION", "LEGAL", "FINANCIAL"}
REPEATABLE_SURFACE_LABELS = {"ORG_NAME_STRONG", "TEAM", "PRODUCT_CONTEXT", "PROJECT_NAME_STRONG", "EVENT_DATE"}

PROJECT_LABELS = {
    "portail_client": "le portail client",
    "connecteur_x": "le connecteur X",
    "moteur_routing": "le moteur de routage",
    "fusion_tenant": "la fusion de tenants",
    "migration_sso": "la migration SSO",
    "consolidation_facturation": "la consolidation facturation",
    "refonte_workflow": "la refonte workflow",
    "socle_reporting": "le socle reporting",
    "passerelle_partenaire": "la passerelle partenaire",
    "pilotage_habilitations": "le pilotage des habilitations",
}
EVENT_LABELS = {
    "migration_auth_q1": "la migration auth du T1",
    "post_merger_hire_2024": "les recrutements post-fusion de 2024",
    "release_support_v3": "la mise en production support V3",
    "audit_iso_q2": "l'audit ISO du T2",
    "freeze_fin_mois": "le freeze de fin de mois",
    "bascule_tenant_avril": "la bascule tenant d'avril",
    "revue_habilitations_mai": "la revue des habilitations de mai",
    "go_live_reporting_ete": "le go-live reporting de l'ete",
    "incident_2042_sso": "l'incident SSO 2042",
    "incident_3011_facturation": "l'incident facturation 3011",
    "incident_1178_mars": "l'incident de mars 1178",
    "incident_4102_escalade": "l'incident d'escalade 4102",
    "incident_2217_sync": "l'incident de synchro 2217",
    "incident_8872_reporting": "l'incident reporting 8872",
    "incident_5570_partner": "l'incident partenaire 5570",
    "incident_6621_capacity": "l'incident capacite 6621",
}
AGE_LABELS = {
    "25-29": "j'ai entre 25 et 29 ans",
    "30-34": "j'ai entre 30 et 34 ans",
    "35-39": "j'ai entre 35 et 39 ans",
    "40-49": "j'ai entre 40 et 49 ans",
}


def _dedupe_spans(spans: List[AnnotationSpan]) -> List[AnnotationSpan]:
    deduped = {}
    for span in spans:
        deduped[(span.start, span.end, span.label)] = span
    return sorted(deduped.values(), key=lambda item: (item.start, item.end, item.label))


def _regex_spans(text: str) -> List[AnnotationSpan]:
    spans: List[AnnotationSpan] = []
    patterns = [
        (r"[\w\.-]+@[\w\.-]+\.\w+", "EMAIL"),
        (r"\+33 6 \d{2} \d{2} \d{2}", "PHONE"),
        (r"ACC-\d{2}-\d{4}", "ACCOUNT_ID"),
    ]
    for pattern, label in patterns:
        for match in re.finditer(pattern, text):
            spans.append(AnnotationSpan(start=match.start(), end=match.end(), label=label, text=match.group(0), confidence=0.95, source="regex"))
    return spans


def _humanize_project(value: str) -> str:
    return PROJECT_LABELS.get(value, value.replace("_", " ").replace("-", " "))


def _humanize_event(value: str) -> str:
    return EVENT_LABELS.get(value, value.replace("_", " ").replace("-", " "))


def _surface_variants(label: str, value: str) -> List[str]:
    variants = [value]
    if label == "USERNAME":
        variants.append(f"login {value}")
    elif label == "PROJECT_NAME_STRONG":
        variants.append(_humanize_project(value))
    elif label == "EVENT_DATE":
        variants.append(_humanize_event(value))
    elif label == "AGE_RANGE":
        age_key = value.replace("tranche d'age ", "").strip()
        variants.append(AGE_LABELS.get(age_key, age_key))
    elif label == "TENURE":
        digits = re.findall(r"\d+", value)
        if digits:
            years = int(digits[0])
            variants.append("je suis en poste depuis 1 an" if years == 1 else f"je suis en poste depuis {years} ans")
    return [variant for variant in dict.fromkeys(variants) if variant]


def _expanded_surface_grounding(document: DocumentRecord) -> List[GroundedMention]:
    grounding = list(document_surface_grounding(document))
    seen = {(mention.label, mention.snippet, mention.occurrence_hint) for mention in grounding}
    for mention in list(grounding):
        if mention.label not in REPEATABLE_SURFACE_LABELS:
            continue
        for occurrence_hint, _ in enumerate(find_occurrences(document.text, mention.snippet), start=1):
            key = (mention.label, mention.snippet, occurrence_hint)
            if key in seen:
                continue
            grounding.append(
                GroundedMention(
                    label=mention.label,
                    canonical_value=mention.canonical_value,
                    snippet=mention.snippet,
                    occurrence_hint=occurrence_hint,
                )
            )
            seen.add(key)
    return grounding


def _signal_value_spans(document: DocumentRecord, *, confidence: float, source: str) -> List[AnnotationSpan]:
    spans: List[AnnotationSpan] = []
    signal_values: Dict[str, List[str]] = document.metadata.get("signal_values", {})
    for label, values in signal_values.items():
        for value in values:
            for snippet in _surface_variants(label, value):
                for start, end in find_occurrences(document.text, snippet):
                    spans.append(
                        AnnotationSpan(
                            start=start,
                            end=end,
                            label=label,
                            text=snippet,
                            confidence=confidence,
                            source=source,
                        )
                    )
    return spans


def _lexicon_spans(document: DocumentRecord, *, confidence: float, source: str) -> List[AnnotationSpan]:
    spans: List[AnnotationSpan] = []
    lexicon = {
        "ORG_NAME_STRONG": list(COMPANIES),
        "TEAM": list(TEAMS),
        "PRODUCT_CONTEXT": list(PRODUCTS),
        "PROJECT_NAME_STRONG": [_humanize_project(value) for value in PROJECTS] + list(PROJECTS),
        "EVENT_DATE": [_humanize_event(value) for value in INCIDENTS + EVENTS] + list(INCIDENTS + EVENTS),
    }
    for label, values in lexicon.items():
        for snippet in dict.fromkeys(values):
            for start, end in find_occurrences(document.text, snippet):
                spans.append(
                    AnnotationSpan(
                        start=start,
                        end=end,
                        label=label,
                        text=snippet,
                        confidence=confidence,
                        source=source,
                    )
                )
    return spans


def _llm_annotation_validator(payload: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("preannotation payload must be a dict")
    spans = payload.get("spans", [])
    if not isinstance(spans, list):
        raise ValueError("spans must be a list")
    normalized_spans = []
    for item in spans:
        if not isinstance(item, dict):
            raise ValueError("span entry must be a dict")
        snippet = str(item.get("snippet", item.get("text", ""))).strip()
        label = str(item.get("label", "")).strip()
        if not snippet or not label:
            raise ValueError("snippet and label are required")
        normalized_spans.append(
            {
                "snippet": snippet,
                "label": label,
                "confidence": float(item.get("confidence", 0.75)),
            }
        )
    relations = payload.get("relations", [])
    if not isinstance(relations, list):
        raise ValueError("relations must be a list")
    doc_labels = payload.get("doc_labels", {})
    if not isinstance(doc_labels, dict):
        raise ValueError("doc_labels must be a dict")
    return {
        "spans": normalized_spans,
        "relations": relations,
        "doc_labels": doc_labels,
        "human_review_required": bool(payload.get("human_review_required", False)),
    }


def _derive_relations(spans: List[AnnotationSpan]) -> List[AnnotationRelation]:
    labels_present = {span.label for span in spans}
    relations: List[AnnotationRelation] = []
    if {"ROLE", "TEAM"}.issubset(labels_present):
        relations.append(AnnotationRelation("composition", "ROLE", "TEAM", "role-team combination"))
    if {"AGE_RANGE", "CERTIFICATION"}.issubset(labels_present):
        relations.append(AnnotationRelation("composition", "AGE_RANGE", "CERTIFICATION", "rare certified age profile"))
    return relations


def _build_doc_labels(document: DocumentRecord, spans: List[AnnotationSpan], relations: List[AnnotationRelation]) -> Dict[str, object]:
    labels_present = {span.label for span in spans}
    sensitive_count = sum(1 for span in spans if span.label in SENSITIVE_LABELS)
    direct_count = sum(1 for span in spans if span.label in DIRECT_LABELS)
    qid_count = len(spans) - sensitive_count - direct_count
    return {
        "direct_risk_level": "high" if direct_count >= 2 else "medium" if direct_count else "low",
        "qid_risk_level": "high" if qid_count >= 4 else "medium" if qid_count >= 2 else "low",
        "style_risk_level": "medium" if "SIGNATURE_PATTERN" in labels_present or "JARGON_PATTERN" in labels_present else "low",
        "compositional_risk_level": "high" if relations else "medium" if qid_count else "low",
        "overall_reid_risk": document.metadata.get("difficulty", "medium"),
        "utility_sensitivity": "high" if document.domain == "support_ticket" else "medium",
        "human_review_required": bool(sensitive_count or document.metadata.get("difficulty") == "hard"),
    }


def build_gold_annotations(document: DocumentRecord) -> GoldSeedAnnotation:
    spans: List[AnnotationSpan] = []
    surface_grounding = _expanded_surface_grounding(document)
    if surface_grounding:
        for mention in surface_grounding:
            resolved = resolve_grounded_mention(document.text, mention)
            if resolved is None:
                continue
            start, end = resolved
            spans.append(
                AnnotationSpan(
                    start=start,
                    end=end,
                    label=mention.label,
                    text=mention.snippet,
                    confidence=1.0,
                    source="gold",
                )
            )
    else:
        spans.extend(_signal_value_spans(document, confidence=1.0, source="gold"))
        spans.extend(_lexicon_spans(document, confidence=0.97, source="gold_lexical"))
    deduped = _dedupe_spans(spans)
    relations = _derive_relations(deduped)
    return GoldSeedAnnotation(
        spans=deduped,
        relations=relations,
        doc_labels=_build_doc_labels(document, deduped, relations),
    )


def _llm_spans(
    document: DocumentRecord,
    gold: GoldSeedAnnotation,
    mode: str,
    client: OpenRouterClient | None = None,
) -> tuple[List[AnnotationSpan], List[AnnotationRelation], Dict[str, object], LLMRunMeta]:
    prompt_spec = load_prompt_spec(PROMPT_PREANNOTATION)
    local_client = client or OpenRouterClient(load_settings())
    if mode != "hybrid-llm":
        meta = LLMRunMeta(
            step_name="preannotation",
            model=local_client.settings.atlas_model_reasoning,
            prompt_version=prompt_spec.version,
            llm_used=False,
            fallback_used=True,
            retry_count=0,
            validation_errors=["preannotation mode disabled"],
            latency_ms=0,
            estimated_cost=0.0,
        )
        return [], [], {}, meta

    ontology_payload = json.dumps(load_config().ontology, ensure_ascii=False, indent=2)
    user_prompt = (
        "Annote les spans sensibles d'un document synthétique en français.\n"
        "Retourne strictement un JSON avec les champs: spans, relations, doc_labels, human_review_required.\n"
        "Chaque span doit contenir snippet, label, confidence.\n\n"
        "Couvre aussi les noms d'organisation, d'equipe, de produit et de projet.\n"
        "Si un meme snippet apparait plusieurs fois dans le texte, annote chaque occurrence pertinente.\n\n"
        f"Texte:\n{document.text}\n\n"
        f"Signaux seed:\n{json.dumps(document.metadata.get('signal_values', {}), ensure_ascii=False, indent=2)}\n\n"
        f"Grounding realise:\n{json.dumps(document.metadata.get('surface_grounding', []), ensure_ascii=False, indent=2)}\n\n"
        f"Labels autorisés:\n{ontology_payload}\n\n"
        f"Gold seed:\n{json.dumps(serialize(gold), ensure_ascii=False, indent=2)}"
    )
    payload, meta = local_client.complete_json(
        step_name="preannotation",
        prompt_spec=prompt_spec,
        user_prompt=user_prompt,
        model=local_client.settings.atlas_model_reasoning,
        validator=_llm_annotation_validator,
        fallback_value={"spans": [], "relations": [], "doc_labels": {}, "human_review_required": False},
        temperature=0.0,
    )
    spans: List[AnnotationSpan] = []
    for item in payload.get("spans", []):
        snippet = item["snippet"]
        for start, end in find_occurrences(document.text, snippet):
            spans.append(
                AnnotationSpan(
                    start=start,
                    end=end,
                    label=item["label"],
                    text=snippet,
                    confidence=float(item.get("confidence", 0.75)),
                    source="llm",
                )
            )
            break
    relations = []
    for item in payload.get("relations", []):
        if isinstance(item, dict):
            relations.append(
                AnnotationRelation(
                    relation_type=str(item.get("relation_type", "composition")),
                    source_label=str(item.get("source_label", "")),
                    target_label=str(item.get("target_label", "")),
                    note=str(item.get("note", "")),
                )
            )
    return spans, relations, dict(payload.get("doc_labels", {})), meta


def build_predicted_annotations(
    document: DocumentRecord,
    gold: GoldSeedAnnotation,
    mode: str = "hybrid-llm",
    client: OpenRouterClient | None = None,
) -> PredictedAnnotation:
    spans = _regex_spans(document.text)
    surface_grounding = _expanded_surface_grounding(document)
    if surface_grounding:
        for mention in surface_grounding:
            resolved = resolve_grounded_mention(document.text, mention)
            if resolved is None:
                continue
            start, end = resolved
            confidence = 0.98 if mention.label in DIRECT_LABELS else 0.90
            spans.append(
                AnnotationSpan(
                    start=start,
                    end=end,
                    label=mention.label,
                    text=mention.snippet,
                    confidence=confidence,
                    source="seed",
                )
            )
    else:
        spans.extend(_signal_value_spans(document, confidence=0.90, source="seed"))
    spans.extend(_lexicon_spans(document, confidence=0.86, source="lexical"))

    llm_spans, llm_relations, llm_doc_labels, llm_meta = _llm_spans(document, gold, mode, client=client)
    spans.extend(llm_spans)
    relations = list(gold.relations) + llm_relations
    doc_labels = dict(gold.doc_labels)
    doc_labels.update(llm_doc_labels)
    review_required = bool(
        document.metadata.get("difficulty") == "hard"
        or any(span.label in SENSITIVE_LABELS for span in spans)
        or llm_meta.fallback_used
    )
    predicted = PredictedAnnotation(
        spans=_dedupe_spans(spans),
        relations=relations,
        doc_labels=doc_labels,
        source="predicted",
        llm_run=llm_meta,
    )
    predicted.doc_labels["human_review_required"] = review_required
    return predicted


def build_annotation_bundle_from_reviewed_spans(document: DocumentRecord, spans: List[AnnotationSpan]) -> AnnotationBundle:
    deduped = _dedupe_spans(spans)
    relations = _derive_relations(deduped)
    doc_labels = _build_doc_labels(document, deduped, relations)
    return AnnotationBundle(
        spans=deduped,
        relations=relations,
        doc_labels=doc_labels,
        human_review_required=bool(doc_labels.get("human_review_required")),
    )


def _annotate_document(document: DocumentRecord, mode: str, client: OpenRouterClient) -> tuple[DocumentRecord, LLMRunMeta]:
    gold = build_gold_annotations(document)
    predicted = build_predicted_annotations(document, gold, mode, client=client)
    document.annotations = AnnotationBundle(
        spans=predicted.spans,
        relations=predicted.relations,
        doc_labels=predicted.doc_labels,
        human_review_required=bool(predicted.doc_labels.get("human_review_required")),
    )
    document.metadata["gold_seed_annotations"] = serialize(gold)
    document.metadata["predicted_annotations"] = serialize(predicted)
    document.metadata["review_target_annotations"] = serialize(predicted)
    document.metadata["preannotation_llm_run"] = serialize(predicted.llm_run) if predicted.llm_run else None
    document.metadata["human_review_required"] = bool(predicted.doc_labels.get("human_review_required"))
    document.metadata.setdefault("review_status", "machine-predicted")
    return document, predicted.llm_run or LLMRunMeta(
        step_name="preannotation",
        model=client.settings.atlas_model_reasoning,
        prompt_version=load_prompt_spec(PROMPT_PREANNOTATION).version,
        llm_used=False,
        fallback_used=True,
        retry_count=0,
        validation_errors=["missing llm run metadata"],
        latency_ms=0,
        estimated_cost=0.0,
    )


def run_preannotation_command(
    mode: str = "hybrid-llm",
    batch: str = "pilot_100",
    reasoning_workers: int | None = None,
    resume_enabled: bool | None = None,
    cache_enabled: bool | None = None,
) -> None:
    runtime_options = build_runtime_options(
        batch_name=batch,
        reasoning_workers=reasoning_workers,
        creative_workers=None,
        resume_enabled=resume_enabled,
        cache_enabled=cache_enabled,
    )
    client = OpenRouterClient(load_settings(), runtime_overrides=runtime_options)
    prompt_spec = load_prompt_spec(PROMPT_PREANNOTATION)
    log(
        f"Starting preannotation mode={mode} batch={batch} "
        f"reasoning_workers={runtime_options['reasoning_workers']} "
        f"resume={runtime_options['resume_enabled']} cache={runtime_options['cache_enabled']}"
    )
    documents = load_documents(annotated=False)
    annotated_documents, _, stage_stats = run_parallel_stage(
        items=documents,
        stage_name="preannotation",
        label="preannotate",
        batch_name=batch,
        prompt_version=prompt_spec.version,
        model=client.settings.atlas_model_reasoning,
        max_workers=int(runtime_options["reasoning_workers"]),
        resume_enabled=bool(runtime_options["resume_enabled"]),
        checkpoint_every=int(runtime_options["checkpoint_every"]),
        item_id_fn=lambda document: document.doc_id,
        worker_fn=lambda document: _annotate_document(document, mode, client),
        result_from_dict=document_from_dict,
    )
    save_documents(annotated_documents, annotated=True)

    manifest = load_batch_manifest(batch)
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts["annotations"] = str(annotations_path())
    manifest["artifacts"] = artifacts
    preannotation_section = {
        "mode": mode,
        "stats": stage_stats,
        "runtime": {
            "reasoning_workers": runtime_options["reasoning_workers"],
            "resume_enabled": runtime_options["resume_enabled"],
            "cache_enabled": runtime_options["cache_enabled"],
            "checkpoint_every": runtime_options["checkpoint_every"],
        },
    }
    manifest["preannotation"] = preannotation_section
    save_batch_manifest(batch, manifest)
    log(f"Preannotations saved: {len(annotated_documents)}")
