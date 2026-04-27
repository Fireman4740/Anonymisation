from __future__ import annotations

from typing import List

from atlas_anno.schemas import CharacterProfile, DocumentRecord
from atlas_anno.surface_grounding import document_surface_grounding, unresolved_mentions

SENSITIVE_LABELS = {"HEALTH", "ETHNICITY", "RELIGION", "DISABILITY", "FAMILY_STATUS", "SEXUAL_ORIENTATION", "LEGAL", "FINANCIAL"}
DIRECT_LABELS = {"PERSON_NAME", "EMAIL", "PHONE", "USERNAME", "ACCOUNT_ID"}
DEFAULT_SURFACE_LABELS = {"ORG_NAME_STRONG", "PRODUCT_CONTEXT"}


def _expected_surface_labels(document: DocumentRecord) -> set[str]:
    labels = set(document.scenario.required_signals)
    labels.update(document.scenario.implicit_signals)
    labels.update(DEFAULT_SURFACE_LABELS)
    if document.scenario.include_signature:
        labels.add("SIGNATURE_PATTERN")
    if document.scenario.include_direct_identifiers:
        labels.update(DIRECT_LABELS)
    if document.scenario.include_sensitive:
        signal_values = document.metadata.get("signal_values", {})
        labels.update(label for label in signal_values if label in SENSITIVE_LABELS)
    return labels


def audit_document(document: DocumentRecord, author: CharacterProfile) -> List[str]:
    issues: List[str] = []
    surface_grounding = document_surface_grounding(document)
    if surface_grounding:
        labels_in_text = {mention.label for mention in surface_grounding}
        for label in _expected_surface_labels(document):
            if label not in labels_in_text:
                issues.append(f"missing_surface_label:{label}")
        for mention in unresolved_mentions(document.text, surface_grounding):
            issues.append(f"missing_surface:{mention.label}")
    else:
        signal_values = document.metadata.get("signal_values", {})
        for label, values in signal_values.items():
            if label == "SIGNATURE_PATTERN":
                continue
            for value in values:
                if value and value not in document.text:
                    issues.append(f"missing_signal:{label}")
    if document.author_id != author.person_id:
        issues.append("author_mismatch")
    if document.target_person_ids != [author.person_id]:
        issues.append("target_mismatch")
    if document.language != author.language:
        issues.append("language_mismatch")
    return issues
