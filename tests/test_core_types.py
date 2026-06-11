from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipegraph"))

from src.core.types import (
    AnonymizationDecision,
    EntitySpan,
    IdentifierType,
    RiskLevel,
    Strategy,
    spans_from_dicts,
)


def test_entity_span_validation():
    with pytest.raises(ValueError):
        EntitySpan(start=-1, end=5, text="x", entity_type="PERSON")
    with pytest.raises(ValueError):
        EntitySpan(start=5, end=5, text="x", entity_type="PERSON")
    with pytest.raises(ValueError):
        EntitySpan(start=0, end=5, text="x", entity_type="")


def test_from_dict_canonical_keys():
    span = EntitySpan.from_dict(
        {"start": 0, "end": 11, "type": "PERSON", "value": "Jean Dupont", "source": "regex", "score": 0.9}
    )
    assert span.text == "Jean Dupont"
    assert span.entity_type == "PERSON"
    assert span.confidence == 0.9
    assert span.identifier_type is IdentifierType.DIRECT  # inferred from type


def test_from_dict_legacy_keys_and_metadata_preserved():
    span = EntitySpan.from_dict(
        {"start": 3, "end": 8, "entity_type": "LOC", "text": "Paris", "source": "gliner",
         "llm_reason": "city name", "validated": True}
    )
    assert span.entity_type == "LOC"
    assert span.identifier_type is IdentifierType.QUASI
    assert span.metadata["llm_reason"] == "city name"
    assert span.metadata["validated"] is True


def test_dict_roundtrip_keeps_provenance():
    raw = {"start": 0, "end": 4, "type": "DATE", "value": "1994", "source": "flair", "score": 0.5}
    out = EntitySpan.from_dict(raw).to_dict()
    assert out["source"] == "flair"
    assert out["type"] == "DATE"
    assert out["identifier_type"] == "quasi"


def test_overlaps():
    a = EntitySpan(start=0, end=10, text="x" * 10, entity_type="PERSON")
    b = EntitySpan(start=5, end=15, text="y" * 10, entity_type="LOC")
    c = EntitySpan(start=10, end=12, text="zz", entity_type="LOC")
    assert a.overlaps(b)
    assert not a.overlaps(c)  # adjacent, not overlapping


def test_spans_from_dicts_skips_malformed():
    spans = spans_from_dicts(
        [
            {"start": 0, "end": 4, "type": "DATE", "value": "1994"},
            {"start": 9, "end": 2, "type": "PERSON", "value": "bad"},
            {"type": "PERSON", "value": "no offsets"},
        ]
    )
    assert len(spans) == 1
    assert spans[0].entity_type == "DATE"


def test_anonymization_decision_serializes():
    span = EntitySpan(start=0, end=5, text="Paris", entity_type="LOC", source="gliner")
    decision = AnonymizationDecision(
        span=span,
        strategy=Strategy.GENERALIZE,
        replacement="[LOC]",
        reason="QUASI identifier, policy LOC->generalize",
        policy_name="config.json:pipeline.nodes.anonymization.policy",
    )
    payload = decision.to_dict()
    assert payload["strategy"] == "generalize"
    assert payload["span"]["value"] == "Paris"
    assert payload["replacement"] == "[LOC]"


def test_risk_level_enum_values():
    assert RiskLevel("high") is RiskLevel.HIGH
