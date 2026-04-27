# -*- coding: utf-8 -*-
"""Tests for LLMVerificationNode — entity verification and correction."""

import pytest
from unittest.mock import patch, MagicMock

from src.nodes.llm.llm_verification_node import (
    LLMVerificationNode,
    _apply_decisions,
    _build_chunks,
    _entities_in_chunk,
    _format_entities_for_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(start, end, value, etype="PER", source="gliner"):
    return {
        "start": start,
        "end": end,
        "value": value,
        "type": etype,
        "source": source,
    }


def _make_state(text, entities, config=None):
    return {
        "original_text": text,
        "text": text,
        "entities": entities,
        "config": config or {},
    }


# ---------------------------------------------------------------------------
# Unit tests — _apply_decisions
# ---------------------------------------------------------------------------

class TestApplyDecisions:
    """Tests for the _apply_decisions helper."""

    def test_keep_action(self):
        entities = [_entity(0, 4, "Jean")]
        decisions = [{"id": 0, "action": "keep"}]
        result = _apply_decisions(entities, decisions, "Jean Dupont", 0)
        assert len(result) == 1
        assert result[0]["value"] == "Jean"

    def test_remove_action_ignored(self):
        """Remove action is treated as keep — recall-first policy."""
        entities = [_entity(0, 4, "Jean"), _entity(10, 15, "Paris")]
        decisions = [
            {"id": 0, "action": "remove", "reason": "not PII"},
            {"id": 1, "action": "keep"},
        ]
        result = _apply_decisions(entities, decisions, "Jean vit \u00e0 Paris", 0)
        assert len(result) == 2  # Both kept — remove is ignored
        assert result[0]["value"] == "Jean"
        assert result[1]["value"] == "Paris"

    def test_retype_action(self):
        entities = [_entity(0, 5, "Acme", etype="PER")]
        decisions = [{"id": 0, "action": "retype", "new_type": "ORG"}]
        result = _apply_decisions(entities, decisions, "Acme Corp", 0)
        assert len(result) == 1
        assert result[0]["type"] == "ORG"
        assert result[0]["source"] == "llm_verified"

    def test_extend_action_with_search(self):
        text = "Jean Dupont habite ici"
        entities = [_entity(0, 4, "Jean")]
        decisions = [
            {"id": 0, "action": "extend", "new_text": "Jean Dupont"},
        ]
        result = _apply_decisions(entities, decisions, text, 0)
        assert len(result) == 1
        assert result[0]["value"] == "Jean Dupont"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 11
        assert result[0]["source"] == "llm_verified"

    def test_extend_action_with_offsets(self):
        text = "Jean Dupont habite ici"
        entities = [_entity(0, 4, "Jean")]
        decisions = [
            {"id": 0, "action": "extend", "new_start": 0, "new_end": 11, "new_text": "Jean Dupont"},
        ]
        result = _apply_decisions(entities, decisions, text, 0)
        assert len(result) == 1
        assert result[0]["value"] == "Jean Dupont"

    def test_extend_fails_gracefully(self):
        text = "Jean habite ici"
        entities = [_entity(0, 4, "Jean")]
        decisions = [
            {"id": 0, "action": "extend", "new_text": "Jean Dupont"},
        ]
        # "Jean Dupont" not in text → keep original
        result = _apply_decisions(entities, decisions, text, 0)
        assert len(result) == 1
        assert result[0]["value"] == "Jean"

    def test_no_decision_keeps_entity(self):
        entities = [_entity(0, 4, "Jean"), _entity(5, 11, "Dupont")]
        decisions = [{"id": 0, "action": "remove"}]
        # remove is ignored (recall-first), no decision for id=1 → both kept
        result = _apply_decisions(entities, decisions, "Jean Dupont", 0)
        assert len(result) == 2

    def test_unknown_action_keeps_entity(self):
        entities = [_entity(0, 4, "Jean")]
        decisions = [{"id": 0, "action": "INVALID"}]
        result = _apply_decisions(entities, decisions, "Jean Dupont", 0)
        assert len(result) == 1

    def test_empty_decisions(self):
        entities = [_entity(0, 4, "Jean")]
        result = _apply_decisions(entities, [], "Jean Dupont", 0)
        assert len(result) == 1

    def test_retype_with_bad_new_type(self):
        entities = [_entity(0, 4, "Jean")]
        decisions = [{"id": 0, "action": "retype", "new_type": ""}]
        result = _apply_decisions(entities, decisions, "Jean Dupont", 0)
        assert len(result) == 1
        assert result[0]["type"] == "PER"  # Unchanged

    def test_chunk_offset(self):
        text = "Prefix. Jean Dupont habite ici"
        entities = [_entity(8, 12, "Jean")]
        decisions = [
            {"id": 0, "action": "extend", "new_start": 0, "new_end": 11, "new_text": "Jean Dupont"},
        ]
        # chunk_offset=8, so absolute start = 0+8 = 8, end = 11+8 = 19
        # text[8:19] = "Jean Dupon" — that's 11 chars from offset 8
        result = _apply_decisions(entities, decisions, text, 8)
        assert len(result) == 1
        # The offset-based path should try text[8:19] = "Jean Dupon" (case mismatch with "Jean Dupont")
        # Fallback search should find it
        assert result[0]["value"] == "Jean Dupont"


# ---------------------------------------------------------------------------
# Unit tests — chunking helpers
# ---------------------------------------------------------------------------

class TestChunkHelpers:

    def test_build_chunks_short_text(self):
        text = "Hello World"
        chunks = _build_chunks(text, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello World"

    def test_entities_in_chunk(self):
        entities = [
            _entity(0, 5, "Jean"),
            _entity(20, 30, "Paris"),
            _entity(50, 60, "Durand"),
        ]
        result = _entities_in_chunk(entities, 0, 25)
        assert len(result) == 2  # Jean (0-5) and Paris (20-30) overlaps 0-25


# ---------------------------------------------------------------------------
# Node-level tests (with mocked LLMClient)
# ---------------------------------------------------------------------------

class TestLLMVerificationNodeDisabled:
    """Test that the node skips gracefully when disabled."""

    def test_disabled_via_runtime_config(self):
        node = LLMVerificationNode()
        state = _make_state("Jean Dupont", [_entity(0, 4, "Jean")], {"llm_verification": False})
        with patch("src.nodes.llm.llm_verification_node.load_full_config", return_value={"features": {"llm_verification": True}}):
            result = node(state)
        assert result == {}

    def test_disabled_via_feature_flag(self):
        node = LLMVerificationNode()
        state = _make_state("Jean Dupont", [_entity(0, 4, "Jean")])
        with patch("src.nodes.llm.llm_verification_node.load_full_config", return_value={"features": {"llm_verification": False}}):
            result = node(state)
        assert result == {}

    def test_disabled_via_disable_llm(self):
        node = LLMVerificationNode()
        state = _make_state("Jean Dupont", [_entity(0, 4, "Jean")], {"disable_llm": True})
        with patch("src.nodes.llm.llm_verification_node.load_full_config", return_value={"features": {"llm_verification": True}}):
            result = node(state)
        assert result == {}

    def test_empty_entities(self):
        node = LLMVerificationNode()
        state = _make_state("Some text", [])
        with patch("src.nodes.llm.llm_verification_node.load_full_config", return_value={"features": {"llm_verification": True}}):
            result = node(state)
        assert result == {}


class TestLLMVerificationNodeWithMock:
    """Test the node with mocked LLM responses."""

    def _run_node(self, text, entities, llm_response):
        node = LLMVerificationNode()
        state = _make_state(text, entities)

        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.chat_batch.return_value = ["__llm_raw__"]

        parsed = llm_response if isinstance(llm_response, list) else None

        patches = {
            "load_full_config": patch("src.nodes.llm.llm_verification_node.load_full_config",
                                      return_value={"features": {"llm_verification": True}}),
            "max_chars": patch("src.nodes.llm.llm_verification_node.estimate_max_prompt_chars",
                               return_value=8000),
            "client": patch.object(node, "_client", return_value=mock_client),
            "extract": patch("src.nodes.llm.llm_verification_node.LLMClient.extract_json",
                             return_value=parsed),
        }

        for p in patches.values():
            p.start()
        try:
            result = node(state)
        finally:
            for p in patches.values():
                p.stop()
        return result

    def test_remove_ignored_all_kept(self):
        """When LLM says remove, all entities should still be returned."""
        text = "Il travaille chez Google depuis 2018"
        entities = [
            _entity(19, 25, "Google", "ORG"),
            _entity(33, 37, "2018", "DATE"),
        ]
        decisions = [
            {"id": 0, "action": "keep"},
            {"id": 1, "action": "remove", "reason": "year is not PII"},
        ]
        result = self._run_node(text, entities, decisions)
        assert "entities" in result
        assert len(result["entities"]) == 2  # Both kept — recall-first

    def test_type_correction(self):
        text = "Jean est de la société Nexus"
        entities = [
            _entity(0, 4, "Jean"),
            _entity(23, 28, "Nexus", "PER"),  # Mis-typed as PER
        ]
        decisions = [
            {"id": 0, "action": "keep"},
            {"id": 1, "action": "retype", "new_type": "ORG"},
        ]
        result = self._run_node(text, entities, decisions)
        assert "entities" in result
        types = {e["value"]: e["type"] for e in result["entities"]}
        assert types["Nexus"] == "ORG"

    def test_graceful_on_bad_response(self):
        text = "Jean habite Paris"
        entities = [_entity(0, 4, "Jean"), _entity(13, 18, "Paris")]
        # LLM returns garbage
        result = self._run_node(text, entities, "not json at all")
        # All entities should be kept when parse fails
        assert "entities" in result
        assert len(result["entities"]) == 2
