"""
Unit tests for span_utils.py - Entity overlap resolution.
"""

import pytest
from src.utils.span_utils import (
    resolve_overlaps,
    deduplicate_entities,
    merge_entity_lists,
    SOURCE_PRIORITY,
)


class TestResolveOverlaps:
    """Tests for resolve_overlaps function."""

    def test_empty_list(self):
        """Test with empty input."""
        result = resolve_overlaps([])
        assert result == []

    def test_no_overlaps(self):
        """Test with non-overlapping entities."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON"},
            {"start": 10, "end": 15, "type": "ORG"},
            {"start": 20, "end": 25, "type": "LOC"},
        ]
        result = resolve_overlaps(entities)
        assert len(result) == 3

    def test_overlapping_longest_strategy(self):
        """Test longest strategy keeps longer span."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON"},
            {"start": 0, "end": 10, "type": "PERSON"},  # Longer, should win
        ]
        result = resolve_overlaps(entities, strategy="longest")

        assert len(result) == 1
        assert result[0]["end"] == 10

    def test_overlapping_first_strategy(self):
        """Test first strategy keeps first entity."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON"},
            {"start": 0, "end": 10, "type": "PERSON"},
        ]
        result = resolve_overlaps(entities, strategy="first")

        assert len(result) == 1
        # First by start position should be kept

    def test_overlapping_highest_score_strategy(self):
        """Test highest_score strategy keeps higher score."""
        entities = [
            {"start": 0, "end": 10, "type": "PERSON", "score": 0.7},
            {"start": 0, "end": 8, "type": "PERSON", "score": 0.95},  # Higher score
        ]
        result = resolve_overlaps(entities, strategy="highest_score")

        assert len(result) == 1
        assert result[0]["score"] == 0.95

    def test_priority_longest_same_label(self):
        """Test priority_longest with same label uses length."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON", "source": "regex"},
            {"start": 0, "end": 10, "type": "PERSON", "source": "regex"},  # Longer
        ]
        result = resolve_overlaps(entities, strategy="priority_longest")

        assert len(result) == 1
        assert result[0]["end"] == 10

    def test_priority_longest_different_source(self):
        """Test priority_longest prefers higher priority source."""
        entities = [
            {"start": 0, "end": 10, "type": "PERSON", "source": "ai"},  # Lower priority
            {"start": 0, "end": 5, "type": "ORG", "source": "regex"},  # Higher priority
        ]
        result = resolve_overlaps(entities, strategy="priority_longest")

        assert len(result) == 1
        assert result[0]["source"] == "regex"

    def test_priority_score_strategy(self):
        """Test priority_score uses source priority then score."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON", "source": "ai", "score": 0.99},
            {"start": 0, "end": 10, "type": "ORG", "source": "regex", "score": 0.5},
        ]
        result = resolve_overlaps(entities, strategy="priority_score")

        assert len(result) == 1
        assert result[0]["source"] == "regex"  # Higher priority source

    def test_partial_overlap(self):
        """Test partial overlap resolution."""
        entities = [
            {"start": 0, "end": 10, "type": "PERSON"},
            {"start": 5, "end": 15, "type": "ORG"},  # Overlaps with first
        ]
        result = resolve_overlaps(entities, strategy="longest")

        # Both have same length, but first one is kept (appears first by start)
        assert len(result) == 1

    def test_adjacent_entities_not_overlapping(self):
        """Test that adjacent entities (end == start) are not considered overlapping."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON"},
            {"start": 5, "end": 10, "type": "ORG"},  # Adjacent, not overlapping
        ]
        result = resolve_overlaps(entities)

        assert len(result) == 2

    def test_invalid_entity_skipped(self):
        """Test that invalid entities (end <= start) are skipped."""
        entities = [
            {"start": 10, "end": 5, "type": "INVALID"},  # end < start
            {"start": 0, "end": 5, "type": "VALID"},
        ]
        result = resolve_overlaps(entities)

        assert len(result) == 1
        assert result[0]["type"] == "VALID"

    def test_custom_source_priority(self):
        """Test with custom source priority mapping."""
        custom_priority = {"custom_source": 100, "other_source": 1}

        entities = [
            {"start": 0, "end": 10, "type": "PERSON", "source": "other_source"},
            {"start": 0, "end": 5, "type": "ORG", "source": "custom_source"},
        ]
        result = resolve_overlaps(
            entities, strategy="priority_longest", source_priority=custom_priority
        )

        assert len(result) == 1
        assert result[0]["source"] == "custom_source"

    def test_unknown_strategy_fallback(self):
        """Test that unknown strategy falls back to 'longest'."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON"},
            {"start": 0, "end": 10, "type": "ORG"},
        ]
        result = resolve_overlaps(entities, strategy="invalid_strategy")

        assert len(result) == 1
        assert result[0]["end"] == 10  # Longest wins


class TestDeduplicateEntities:
    """Tests for deduplicate_entities function."""

    def test_empty_list(self):
        """Test with empty input."""
        result = deduplicate_entities([])
        assert result == []

    def test_no_duplicates(self):
        """Test with no duplicates."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON"},
            {"start": 10, "end": 15, "type": "ORG"},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 2

    def test_exact_duplicates(self):
        """Test removal of exact duplicates."""
        entities = [
            {"start": 0, "end": 5, "type": "PERSON", "score": 0.9},
            {"start": 0, "end": 5, "type": "PERSON", "score": 0.8},  # Duplicate
            {"start": 10, "end": 15, "type": "ORG"},
        ]
        result = deduplicate_entities(entities)

        assert len(result) == 2
        # First occurrence is kept
        assert result[0]["score"] == 0.9

    def test_case_insensitive_type_matching(self):
        """Test that type comparison is case-insensitive."""
        entities = [
            {"start": 0, "end": 5, "type": "person"},
            {"start": 0, "end": 5, "type": "PERSON"},  # Same type, different case
        ]
        result = deduplicate_entities(entities)

        assert len(result) == 1

    def test_entity_group_fallback(self):
        """Test that entity_group is used if type is missing."""
        entities = [
            {"start": 0, "end": 5, "entity_group": "PERSON"},
            {"start": 0, "end": 5, "entity_group": "PERSON"},  # Duplicate
        ]
        result = deduplicate_entities(entities)

        assert len(result) == 1


class TestMergeEntityLists:
    """Tests for merge_entity_lists function."""

    def test_empty_lists(self):
        """Test with empty inputs."""
        result = merge_entity_lists([], [], [])
        assert result == []

    def test_single_list(self):
        """Test with single list."""
        entities = [{"start": 0, "end": 5, "type": "PERSON"}]
        result = merge_entity_lists(entities)

        assert len(result) == 1

    def test_merge_multiple_lists(self):
        """Test merging multiple lists."""
        list1 = [{"start": 0, "end": 5, "type": "PERSON"}]
        list2 = [{"start": 10, "end": 15, "type": "ORG"}]

        result = merge_entity_lists(list1, list2)

        assert len(result) == 2

    def test_merge_with_deduplication(self):
        """Test that duplicates are removed during merge."""
        list1 = [{"start": 0, "end": 5, "type": "PERSON"}]
        list2 = [{"start": 0, "end": 5, "type": "PERSON"}]  # Duplicate

        result = merge_entity_lists(list1, list2)

        assert len(result) == 1

    def test_merge_with_overlap_resolution(self):
        """Test that overlaps are resolved during merge."""
        list1 = [{"start": 0, "end": 5, "type": "PERSON", "source": "ai"}]
        list2 = [{"start": 0, "end": 10, "type": "ORG", "source": "regex"}]

        result = merge_entity_lists(list1, list2, resolve_overlapping=True)

        assert len(result) == 1
        assert result[0]["source"] == "regex"  # Higher priority

    def test_merge_without_overlap_resolution(self):
        """Test merge without overlap resolution (just sort)."""
        list1 = [{"start": 0, "end": 5, "type": "PERSON"}]
        list2 = [{"start": 0, "end": 10, "type": "ORG"}]

        result = merge_entity_lists(list1, list2, resolve_overlapping=False)

        # Both kept, just sorted
        assert len(result) == 2

    def test_merge_with_custom_strategy(self):
        """Test merge with custom overlap strategy."""
        list1 = [{"start": 0, "end": 10, "type": "PERSON", "score": 0.7}]
        list2 = [{"start": 0, "end": 5, "type": "ORG", "score": 0.95}]

        result = merge_entity_lists(
            list1, list2, resolve_overlapping=True, strategy="highest_score"
        )

        assert len(result) == 1
        assert result[0]["score"] == 0.95

    def test_merge_none_list_ignored(self):
        """Test that None lists are handled gracefully."""
        list1 = [{"start": 0, "end": 5, "type": "PERSON"}]

        result = merge_entity_lists(list1, None)

        assert len(result) == 1


class TestSourcePriority:
    """Tests for SOURCE_PRIORITY constant."""

    def test_deterministic_higher_than_ai(self):
        """Test that deterministic sources have higher priority than AI."""
        assert SOURCE_PRIORITY["deterministic"] > SOURCE_PRIORITY["ai"]
        assert SOURCE_PRIORITY["regex"] > SOURCE_PRIORITY["gliner"]

    def test_validator_high_priority(self):
        """Test that validator has high priority."""
        assert SOURCE_PRIORITY["validator"] > SOURCE_PRIORITY["ai"]
        assert SOURCE_PRIORITY["validator"] > SOURCE_PRIORITY["gliner"]

    def test_unknown_has_lowest_priority(self):
        """Test that unknown source has lowest priority."""
        for source, priority in SOURCE_PRIORITY.items():
            if source != "unknown":
                assert priority >= SOURCE_PRIORITY["unknown"]
