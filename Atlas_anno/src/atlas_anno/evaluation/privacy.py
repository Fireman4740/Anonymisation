from __future__ import annotations

from typing import Dict, List

from atlas_anno.schemas import AnonymizationResult, DocumentRecord
from atlas_anno.surface_grounding import document_surface_grounding, resolve_grounded_mention


DIRECT_LABELS = {"PERSON_NAME", "EMAIL", "PHONE", "USERNAME", "ACCOUNT_ID"}
SENSITIVE_LABELS = {"HEALTH", "ETHNICITY", "RELIGION", "DISABILITY", "FAMILY_STATUS", "SEXUAL_ORIENTATION", "LEGAL", "FINANCIAL"}


def evaluate_privacy(documents: List[DocumentRecord], anonymized: Dict[str, AnonymizationResult], attack_success: Dict[str, bool] | None = None) -> Dict[str, object]:
    details = []
    total_direct = total_qid = total_sensitive = 0
    remaining_direct = remaining_qid = remaining_sensitive = 0
    attack_success = attack_success or {}
    resistant_count = 0

    for document in documents:
        result = anonymized[document.doc_id]
        direct = qid = sensitive = 0
        direct_left = qid_left = sensitive_left = 0
        surface_grounding = document_surface_grounding(document)
        if surface_grounding:
            for mention in surface_grounding:
                still_present = resolve_grounded_mention(result.anonymized_text, mention) is not None
                if mention.label in DIRECT_LABELS:
                    direct += 1
                    direct_left += int(still_present)
                elif mention.label in SENSITIVE_LABELS:
                    sensitive += 1
                    sensitive_left += int(still_present)
                else:
                    qid += 1
                    qid_left += int(still_present)
        else:
            signal_values = document.metadata.get("signal_values", {})
            for label, values in signal_values.items():
                for value in values:
                    if label in DIRECT_LABELS:
                        direct += 1
                        if value in result.anonymized_text:
                            direct_left += 1
                    elif label in SENSITIVE_LABELS:
                        sensitive += 1
                        if value in result.anonymized_text:
                            sensitive_left += 1
                    else:
                        qid += 1
                        if value in result.anonymized_text:
                            qid_left += 1
        resistant = not attack_success.get(document.doc_id, False)
        resistant_count += int(resistant)
        total_direct += direct
        total_qid += qid
        total_sensitive += sensitive
        remaining_direct += direct_left
        remaining_qid += qid_left
        remaining_sensitive += sensitive_left
        details.append(
            {
                "doc_id": document.doc_id,
                "direct_remaining": direct_left,
                "qid_remaining": qid_left,
                "sensitive_remaining": sensitive_left,
                "attack_resistant": resistant,
            }
        )

    direct_removal = 1.0 - (remaining_direct / total_direct if total_direct else 0.0)
    qid_reduction = 1.0 - (remaining_qid / total_qid if total_qid else 0.0)
    sensitive_reduction = 1.0 - (remaining_sensitive / total_sensitive if total_sensitive else 0.0)
    attack_resistance = resistant_count / len(documents) if documents else 0.0
    privacy_score = round((0.35 * direct_removal) + (0.25 * qid_reduction) + (0.20 * sensitive_reduction) + (0.20 * attack_resistance), 4)
    return {
        "summary": {
            "documents": len(documents),
            "direct_removal": round(direct_removal, 4),
            "qid_reduction": round(qid_reduction, 4),
            "sensitive_reduction": round(sensitive_reduction, 4),
            "attack_resistance": round(attack_resistance, 4),
            "privacy_score": privacy_score,
        },
        "details": details,
    }
