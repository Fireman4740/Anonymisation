from __future__ import annotations

import json
import re
from typing import Dict, List

from atlas_anno.console import ProgressBar, log
from atlas_anno.constants import PROMPT_ANONYMIZER
from atlas_anno.io import serialize
from atlas_anno.llm import OpenRouterClient
from atlas_anno.prompts import load_prompt_spec
from atlas_anno.schemas import AnonymizationResult, AnnotationSpan, DocumentRecord
from atlas_anno.settings import load_settings
from atlas_anno.storage import load_documents, save_anonymization_results


GENERALIZATION_MAP = {
    "PERSON_NAME": "une collaboratrice",
    "EMAIL": "une adresse email interne",
    "PHONE": "un numero professionnel",
    "USERNAME": "un identifiant interne",
    "ACCOUNT_ID": "un compte client",
    "ROLE": "une fonction technique",
    "DEGREE": "un diplome avance",
    "AGE_RANGE": "une tranche d'age jeune",
    "TENURE": "une anciennete moyenne",
    "TEAM": "une equipe technique",
    "DEPARTMENT": "un departement interne",
    "LOCATION": "un site regional",
    "NATIONALITY": "une nationalite",
    "CERTIFICATION": "une certification cloud",
    "SKILL_RARE": "une competence rare",
    "EVENT_DATE": "un incident recent",
    "PRODUCT_CONTEXT": "un produit interne",
    "RARE_RESPONSIBILITY": "une responsabilite specialisee",
    "HEALTH": "une contrainte medicale",
    "ETHNICITY": "un contexte personnel sensible",
    "RELIGION": "une contrainte personnelle sensible",
    "DISABILITY": "un besoin d'accessibilite",
    "FAMILY_STATUS": "une contrainte familiale",
    "SEXUAL_ORIENTATION": "un element personnel sensible",
    "LEGAL": "un sujet juridique",
    "FINANCIAL": "un sujet financier",
}


def _apply_replacements(text: str, spans: List[AnnotationSpan], strategy: str) -> str:
    result = text
    for span in sorted(spans, key=lambda item: (item.start, item.end), reverse=True):
        if strategy == "masking":
            replacement = f"<{span.label}>"
        else:
            replacement = GENERALIZATION_MAP.get(span.label, f"information_{span.label.lower()}")
        result = result[: span.start] + replacement + result[span.end :]
    if strategy == "rewrite_balanced":
        result = re.sub(r"\s+", " ", result).replace(" .", ".").replace(" ,", ",").strip()
        result = result.replace("Bonjour support_manager,", "Bonjour,")
    return result


def _rewrite_balanced_fallback(document: DocumentRecord, spans: List[AnnotationSpan]) -> str:
    return _apply_replacements(document.text, spans, "rewrite_balanced")


def _rewrite_balanced_llm(document: DocumentRecord, spans: List[AnnotationSpan]) -> Dict[str, object]:
    client = OpenRouterClient(load_settings())
    prompt_spec = load_prompt_spec(PROMPT_ANONYMIZER)
    fallback_text = _rewrite_balanced_fallback(document, spans)
    fallback_value = {
        "anonymized_text": fallback_text,
        "actions_performed": sorted({f"rewrite_balanced:{span.label}" for span in spans}),
        "rationale": "fallback heuristic rewrite",
        "estimated_privacy_gain": _estimate_metrics(document, fallback_text, spans)["privacy_gain"],
        "estimated_utility_loss": _estimate_metrics(document, fallback_text, spans)["utility_loss"],
    }

    def _validator(payload: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("anonymizer payload must be a dict")
        text = str(payload.get("anonymized_text", "")).strip()
        actions = payload.get("actions_performed", [])
        if not text:
            raise ValueError("anonymized_text is required")
        if not isinstance(actions, list):
            raise ValueError("actions_performed must be a list")
        return {
            "anonymized_text": text,
            "actions_performed": [str(item) for item in actions],
            "rationale": str(payload.get("rationale", "")),
            "estimated_privacy_gain": float(payload.get("estimated_privacy_gain", 0.0)),
            "estimated_utility_loss": float(payload.get("estimated_utility_loss", 0.0)),
        }

    user_prompt = (
        "Réécris ce texte pour réduire le risque de ré-identification tout en gardant le sens métier.\n"
        "Retourne strictement un JSON avec les champs anonymized_text, actions_performed, rationale, estimated_privacy_gain, estimated_utility_loss.\n\n"
        f"Texte brut:\n{document.text}\n\n"
        f"Annotations prédictes:\n{json.dumps(serialize(document.metadata.get('predicted_annotations', {})), ensure_ascii=False, indent=2)}\n\n"
        f"Texte fallback:\n{fallback_text}"
    )
    value, meta = client.complete_json(
        step_name="rewrite_balanced",
        prompt_spec=prompt_spec,
        user_prompt=user_prompt,
        model=client.settings.atlas_model_creative,
        validator=_validator,
        fallback_value=fallback_value,
        temperature=0.1,
    )
    payload = dict(value)
    payload["llm_run"] = serialize(meta)
    return payload


def _estimate_metrics(document: DocumentRecord, anonymized_text: str, spans: List[AnnotationSpan]) -> Dict[str, float]:
    total = len(spans) or 1
    removed = sum(1 for span in spans if span.text not in anonymized_text)
    privacy_gain = round(removed / total, 4)
    utility_loss = round(min(1.0, abs(len(document.text) - len(anonymized_text)) / max(1, len(document.text))), 4)
    return {"privacy_gain": privacy_gain, "utility_loss": utility_loss}


def anonymize_documents(documents: List[DocumentRecord], strategy: str, mode: str = "auto") -> List[AnonymizationResult]:
    results: List[AnonymizationResult] = []
    progress = ProgressBar(total=len(documents), label=f"anonymize:{strategy}")
    for document in documents:
        spans = document.annotations.spans
        if strategy == "rewrite_balanced" and mode == "llm":
            payload = _rewrite_balanced_llm(document, spans)
            anonymized_text = str(payload["anonymized_text"])
            actions = list(payload["actions_performed"])
            metrics = {
                "privacy_gain": float(payload["estimated_privacy_gain"]),
                "utility_loss": float(payload["estimated_utility_loss"]),
            }
            rationale = str(payload.get("rationale", ""))
            metadata = {
                "split": document.split,
                "difficulty": document.metadata.get("difficulty", "medium"),
                "llm_run": payload.get("llm_run"),
                "mode": mode,
            }
        else:
            anonymized_text = _apply_replacements(document.text, spans, strategy)
            metrics = _estimate_metrics(document, anonymized_text, spans)
            actions = sorted({f"{strategy}:{span.label}" for span in spans})
            rationale = f"Applied {strategy} over {len(spans)} spans"
            metadata = {"split": document.split, "difficulty": document.metadata.get("difficulty", "medium"), "mode": mode}
        results.append(
            AnonymizationResult(
                doc_id=document.doc_id,
                strategy=strategy,
                anonymized_text=anonymized_text,
                actions_performed=actions,
                rationale=rationale,
                estimated_privacy_gain=metrics["privacy_gain"],
                estimated_utility_loss=metrics["utility_loss"],
                metadata=metadata,
            )
        )
        progress.advance(extra=document.doc_id)
    progress.close()
    return results


def run_anonymization_command(strategy: str, mode: str = "auto") -> None:
    log(f"Starting anonymization strategy={strategy} mode={mode}")
    documents = load_documents(annotated=True)
    results = anonymize_documents(documents, strategy, mode=mode)
    save_anonymization_results(strategy, results)
    log(f"Anonymization saved: {len(results)} documents")
