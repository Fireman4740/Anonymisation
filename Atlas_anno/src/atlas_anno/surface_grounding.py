from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from atlas_anno.records import grounded_mention_from_dict
from atlas_anno.schemas import AnnotationSpan, DocumentRecord, GroundedMention


SpanTuple = Tuple[int, int]


def find_occurrences(text: str, value: str) -> List[SpanTuple]:
    matches: List[SpanTuple] = []
    start = 0
    while value:
        start = text.find(value, start)
        if start == -1:
            break
        end = start + len(value)
        matches.append((start, end))
        start = end
    return matches


def normalize_surface_grounding(payload: object) -> List[GroundedMention]:
    if not isinstance(payload, list):
        return []
    mentions: List[GroundedMention] = []
    for item in payload:
        if isinstance(item, GroundedMention):
            mentions.append(item)
        elif isinstance(item, dict):
            mentions.append(grounded_mention_from_dict(item))
    return mentions


def resolve_grounded_mention(text: str, mention: GroundedMention) -> SpanTuple | None:
    occurrences = find_occurrences(text, mention.snippet)
    index = mention.occurrence_hint - 1
    if index < 0 or index >= len(occurrences):
        return None
    return occurrences[index]


def build_spans_from_grounding(
    text: str,
    grounding: Sequence[GroundedMention],
    *,
    confidence: float,
    source: str,
) -> List[AnnotationSpan]:
    spans: List[AnnotationSpan] = []
    for mention in grounding:
        resolved = resolve_grounded_mention(text, mention)
        if resolved is None:
            continue
        start, end = resolved
        spans.append(
            AnnotationSpan(
                start=start,
                end=end,
                label=mention.label,
                text=mention.snippet,
                confidence=confidence,
                source=source,
            )
        )
    return spans


def document_surface_grounding(document: DocumentRecord) -> List[GroundedMention]:
    return normalize_surface_grounding(document.metadata.get("surface_grounding", []))


def canonical_surface_map(document: DocumentRecord) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for mention in document_surface_grounding(document):
        values = mapping.setdefault(mention.label, [])
        if mention.snippet not in values:
            values.append(mention.snippet)
    return mapping


def has_surface_grounding(document: DocumentRecord) -> bool:
    return bool(document_surface_grounding(document))


def unresolved_mentions(text: str, grounding: Iterable[GroundedMention]) -> List[GroundedMention]:
    missing: List[GroundedMention] = []
    for mention in grounding:
        if resolve_grounded_mention(text, mention) is None:
            missing.append(mention)
    return missing
