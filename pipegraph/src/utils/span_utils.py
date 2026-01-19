"""
Span/Entity overlap resolution utilities.
Provides intelligent strategies for merging overlapping entity spans.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger("SpanUtils")

# Source priority: higher = more trusted
SOURCE_PRIORITY = {
    "regex": 10,
    "deterministic": 10,
    "validator": 9,
    "gliner": 5,
    "flair": 5,
    "spacy": 5,
    "ai": 4,
    "unknown": 1,
}


def resolve_overlaps(
    entities: List[Dict[str, Any]],
    strategy: str = "priority_longest",
    source_priority: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Resolve overlapping entities using the specified strategy.

    Strategies:
    - "longest": Keep the longest span when overlapping
    - "highest_score": Keep the entity with highest score/confidence
    - "first": Keep the first entity encountered (by start position)
    - "priority_longest": Use source priority, then longest span (recommended)
    - "priority_score": Use source priority, then highest score

    Args:
        entities: List of entity dicts with at least 'start', 'end' keys
        strategy: Resolution strategy name
        source_priority: Custom source priority dict (optional)

    Returns:
        List of non-overlapping entities
    """
    if not entities:
        return []

    # Use custom or default priorities
    priorities = source_priority or SOURCE_PRIORITY

    # Sort by start position, then by length (desc), then by score (desc)
    sorted_ents = sorted(
        entities,
        key=lambda e: (
            e.get("start", 0),
            -(e.get("end", 0) - e.get("start", 0)),
            -e.get("score", 0),
        ),
    )

    merged: List[Dict[str, Any]] = []

    for ent in sorted_ents:
        start = ent.get("start", 0)
        end = ent.get("end", 0)

        if end <= start:
            continue

        # Check for overlap with the last merged entity
        if not merged:
            merged.append(ent)
            continue

        last = merged[-1]
        last_start = last.get("start", 0)
        last_end = last.get("end", 0)

        # No overlap - just add
        if start >= last_end:
            merged.append(ent)
            continue

        # Overlap detected - resolve using strategy
        should_replace = _should_replace(ent, last, strategy, priorities)

        if should_replace:
            merged[-1] = ent
        # else: keep the existing entity (last)

    return merged


def _should_replace(
    new_ent: Dict[str, Any], existing_ent: Dict[str, Any], strategy: str, priorities: Dict[str, int]
) -> bool:
    """
    Determine if new_ent should replace existing_ent based on strategy.
    """
    new_len = new_ent.get("end", 0) - new_ent.get("start", 0)
    existing_len = existing_ent.get("end", 0) - existing_ent.get("start", 0)

    new_score = new_ent.get("score", 0)
    existing_score = existing_ent.get("score", 0)

    new_source = new_ent.get("source", "unknown").lower()
    existing_source = existing_ent.get("source", "unknown").lower()

    new_priority = priorities.get(new_source, 1)
    existing_priority = priorities.get(existing_source, 1)

    new_label = new_ent.get("type", new_ent.get("entity_group", ""))
    existing_label = existing_ent.get("type", existing_ent.get("entity_group", ""))
    same_label = (
        new_label.upper() == existing_label.upper() if new_label and existing_label else False
    )

    if strategy == "longest":
        return new_len > existing_len

    elif strategy == "highest_score":
        return new_score > existing_score

    elif strategy == "first":
        # Never replace - first wins
        return False

    elif strategy == "priority_longest":
        # Same label: keep longer
        if same_label:
            return new_len > existing_len
        # Different label: use source priority, then length
        if new_priority != existing_priority:
            return new_priority > existing_priority
        return new_len > existing_len

    elif strategy == "priority_score":
        # Same label: keep higher score
        if same_label:
            return new_score > existing_score
        # Different label: use source priority, then score
        if new_priority != existing_priority:
            return new_priority > existing_priority
        return new_score > existing_score

    else:
        logger.warning(f"Unknown strategy: {strategy}, using 'longest'")
        return new_len > existing_len


def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact duplicates (same start, end, type).

    Args:
        entities: List of entity dicts

    Returns:
        Deduplicated list
    """
    seen = set()
    result = []

    for ent in entities:
        key = (
            ent.get("start", 0),
            ent.get("end", 0),
            str(ent.get("type", ent.get("entity_group", ""))).upper(),
        )
        if key not in seen:
            seen.add(key)
            result.append(ent)

    return result


def merge_entity_lists(
    *entity_lists: List[Dict[str, Any]],
    resolve_overlapping: bool = True,
    strategy: str = "priority_longest",
) -> List[Dict[str, Any]]:
    """
    Merge multiple entity lists, optionally resolving overlaps.

    Args:
        *entity_lists: Variable number of entity lists to merge
        resolve_overlapping: Whether to resolve overlaps (default True)
        strategy: Overlap resolution strategy

    Returns:
        Merged and deduplicated entity list
    """
    all_entities = []

    for entities in entity_lists:
        if entities:
            all_entities.extend(entities)

    if not all_entities:
        return []

    # First deduplicate exact matches
    deduped = deduplicate_entities(all_entities)

    # Then resolve overlaps if requested
    if resolve_overlapping:
        return resolve_overlaps(deduped, strategy=strategy)

    # Just sort by position
    return sorted(deduped, key=lambda e: (e.get("start", 0), e.get("end", 0)))
