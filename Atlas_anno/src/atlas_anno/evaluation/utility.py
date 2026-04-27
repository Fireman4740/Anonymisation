from __future__ import annotations

import re
from typing import Dict, List

from atlas_anno.schemas import AnonymizationResult, DocumentRecord


def _predict_domain(text: str) -> str:
    lowered = text.lower()
    email_markers = (
        "bonjour",
        "rebonjour",
        "bien cordialement",
        "merci d'avance",
        "retour de mail",
        "point rapide",
    )
    ticket_markers = (
        "ticket",
        "incident",
        "support",
        "service desk",
        "acceder",
        "acces",
        "bloqu",
        "compte",
        "facturation",
        "synchro",
        "habilitation",
        "reporting",
        "donnees",
        "trace",
        "contournement",
    )
    email_score = sum(1 for marker in email_markers if marker in lowered)
    ticket_score = sum(1 for marker in ticket_markers if marker in lowered)
    if ticket_score > email_score:
        return "support_ticket"
    if email_score:
        return "email"
    return "email"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def evaluate_utility(documents: List[DocumentRecord], anonymized: Dict[str, AnonymizationResult]) -> Dict[str, object]:
    details = []
    domain_matches = 0
    lexical_total = 0.0
    for document in documents:
        result = anonymized[document.doc_id]
        predicted = _predict_domain(result.anonymized_text)
        domain_matches += int(predicted == document.domain)
        original_tokens = set(_tokenize(document.text))
        anonymized_tokens = set(_tokenize(result.anonymized_text))
        lexical_retention = len(original_tokens & anonymized_tokens) / max(1, len(original_tokens))
        lexical_total += lexical_retention
        details.append(
            {
                "doc_id": document.doc_id,
                "predicted_domain": predicted,
                "true_domain": document.domain,
                "lexical_retention": round(lexical_retention, 4),
            }
        )

    n_documents = len(documents) or 1
    domain_accuracy = domain_matches / n_documents
    lexical_retention = lexical_total / n_documents
    utility_score = round((0.60 * domain_accuracy) + (0.40 * lexical_retention), 4)
    return {
        "summary": {
            "documents": len(documents),
            "domain_accuracy": round(domain_accuracy, 4),
            "lexical_retention": round(lexical_retention, 4),
            "utility_score": utility_score,
        },
        "details": details,
    }
