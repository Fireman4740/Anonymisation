"""Validation helpers for anonymized text."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple
import re

_PLACEHOLDER_RE = re.compile(r"\[(?P<type>[A-Z]+)_[^\]]+\]")


def count_placeholder_types(text: str) -> Dict[str, int]:
    """Return counts per placeholder type (PER, MAIL, DATE, ...)."""
    counts: Counter[str] = Counter()
    for match in _PLACEHOLDER_RE.finditer(text):
        counts[match.group("type")] += 1
    return dict(counts)


def validate_anonymization(
    original: str,
    anonymized: str,
    expected_counts: Dict[str, int] | None = None,
    forbidden_patterns: Iterable[str] | None = None,
) -> List[str]:
    """Validate forbidden patterns and placeholder multiplicity.

    Returns a list of human-readable issues (empty list = validation OK).
    """
    issues: List[str] = []
    placeholder_counts = count_placeholder_types(anonymized)

    for pattern in forbidden_patterns or []:
        if not pattern:
            continue
        if pattern in anonymized:
            issues.append(f"❌ Forbidden pattern visible: {pattern}")

    if expected_counts:
        for key, expected in expected_counts.items():
            if expected is None:
                continue
            try:
                expected_int = int(expected)
            except Exception:
                continue
            actual = _resolve_placeholder_count(key, anonymized, placeholder_counts)
            if actual < expected_int:
                issues.append(
                    f"⚠️ Placeholder {key}: {actual}/{expected_int} occurrences conservées"
                )

    return issues


def _resolve_placeholder_count(
    key: str,
    anonymized: str,
    counts_by_type: Dict[str, int],
) -> int:
    if not key:
        return 0
    if key.startswith("["):
        return anonymized.count(key)
    normalized = key.strip("[]").upper()
    return counts_by_type.get(normalized, 0)


__all__ = ["count_placeholder_types", "validate_anonymization"]
