from __future__ import annotations

from typing import Dict, List, Tuple

from atlas_anno.schemas import CharacterProfile, DocumentRecord
from atlas_anno.surface_grounding import document_surface_grounding, resolve_grounded_mention, unresolved_mentions

SENSITIVE_LABELS = {"HEALTH", "ETHNICITY", "RELIGION", "DISABILITY", "FAMILY_STATUS", "SEXUAL_ORIENTATION", "LEGAL", "FINANCIAL"}
DIRECT_LABELS = {"PERSON_NAME", "EMAIL", "PHONE", "USERNAME", "ACCOUNT_ID"}
DEFAULT_SURFACE_LABELS = {"ORG_NAME_STRONG", "PRODUCT_CONTEXT"}

LEAK_ATTRIBUTE_LABELS = {
    "PERSON_NAME": "full_name",
    "EMAIL": "email",
    "PHONE": "phone",
    "USERNAME": "username",
    "ACCOUNT_ID": "account_id",
    "NIR": "nir_like",
    "ADDRESS": "address",
}


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


def _is_word_char(value: str) -> bool:
    return bool(value) and (value.isalnum() or value == "_")


def _has_word_boundary(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (_is_word_char(before) and _is_word_char(text[start:start + 1])) and not (
        _is_word_char(after) and _is_word_char(text[end - 1:end])
    )


def _span_issues(document: DocumentRecord) -> List[str]:
    issues: List[str] = []
    spans: List[Tuple[int, int, str]] = []
    for mention in document_surface_grounding(document):
        resolved = resolve_grounded_mention(document.text, mention)
        if resolved is None:
            continue
        start, end = resolved
        spans.append((start, end, mention.label))
        if not _has_word_boundary(document.text, start, end):
            issues.append(f"span_boundary:{mention.label}")
    for index, (start, end, label) in enumerate(spans):
        for other_start, other_end, other_label in spans[index + 1:]:
            overlaps = start < other_end and other_start < end
            contains = (start <= other_start and other_end <= end) or (other_start <= start and end <= other_end)
            if overlaps and not contains:
                issues.append(f"span_overlap:{label}:{other_label}")
    return issues


def _difficulty_issues(document: DocumentRecord, author: CharacterProfile) -> List[str]:
    issues: List[str] = []
    for mention in document_surface_grounding(document):
        if mention.difficulty_mode == "explicit_hard" and mention.canonical_value and mention.canonical_value in document.text:
            issues.append(f"difficulty_violation:{mention.label}")
        if mention.difficulty_mode == "implicit":
            if not mention.cue_type:
                issues.append(f"difficulty_violation:{mention.label}")
                continue
            if mention.canonical_value and mention.canonical_value in document.text:
                issues.append(f"difficulty_violation:{mention.label}")
                continue
            for cue in author.contextual_cues:
                if cue.reveals_label == mention.label and cue.reveals_value and cue.reveals_value in document.text:
                    issues.append(f"difficulty_violation:{mention.label}")
                    break
    return issues


def _unintended_leak_issues(document: DocumentRecord, author: CharacterProfile) -> List[str]:
    issues: List[str] = []
    signal_values: Dict[str, List[str]] = document.metadata.get("signal_values", {})
    grounded_labels = {mention.label for mention in document_surface_grounding(document)}
    allowed_labels = set(signal_values) | grounded_labels
    for label, field_name in LEAK_ATTRIBUTE_LABELS.items():
        if label in allowed_labels:
            continue
        value = getattr(author, field_name, "")
        if value and value in document.text:
            issues.append(f"unintended_leak:{label}")
    for sensitive_label in author.sensitive_attributes:
        if sensitive_label in allowed_labels:
            continue
        if sensitive_label and sensitive_label in document.text:
            issues.append(f"unintended_leak:{sensitive_label}")
    return issues


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
    issues.extend(_span_issues(document))
    issues.extend(_difficulty_issues(document, author))
    issues.extend(_unintended_leak_issues(document, author))
    if any(issue.startswith(("unintended_leak:", "span_boundary:", "span_overlap:", "difficulty_violation:")) for issue in issues):
        document.metadata["human_review_required"] = True
    return issues
