import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

# Ensure project modules are importable when tests run from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.orchestrator import anonymize_text
from src.services.detection.detection import DetectedEntity
from src.services.generalization.generalizer import Generalization
from src.services.llm.llm_pipeline import RuptaResult


class StubDetectionService:
    """Detection stub returning a predefined entity list."""

    def __init__(self, entities: Iterable[DetectedEntity]) -> None:
        self._entities = list(entities)

    def detect_all(
        self,
        text: str,
        skip_regex_tags: Optional[set[str]] = None,
        external_ner: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DetectedEntity]:
        return self._entities


class StubGeneralizationService:
    """Generalization stub that returns the input text unchanged."""

    def __init__(self, generalizations: Optional[List[Generalization]] = None) -> None:
        self._generalizations = generalizations or []

    def apply_all(self, text: str) -> Tuple[str, List[Generalization]]:
        return text, list(self._generalizations)


class StubLLMService:
    """Minimal LLMPipelineService replacement for integration tests."""

    def __init__(self, paraphrase_suffix: str = " | paraphrased", rupta_suffix: str = " | optimized") -> None:
        self._paraphrase_suffix = paraphrase_suffix
        self._rupta_suffix = rupta_suffix
        self.models: Dict[str, str] = {"detect": "stub/model"}

    def paraphrase(
        self,
        text: str,
        temperature: float = 0.3,
        ensure_placeholders_preserved: bool = True,
    ) -> Tuple[str, Optional[str]]:
        return f"{text}{self._paraphrase_suffix}", None

    def audit(self, text: str) -> Tuple[Dict[str, Any], Optional[str]]:
        return {"risk_score": 13, "findings": [], "recommendations": []}, None

    def optimize_with_rupta(
        self,
        original_text: str,
        initial_anonymized_text: str,
        ground_truth_people: Optional[str],
        ground_truth_label: Optional[str],
    ) -> Tuple[RuptaResult, Optional[str]]:
        return (
            RuptaResult(
                final_text=f"{initial_anonymized_text}{self._rupta_suffix}",
                privacy_score={"rank": 5},
                utility_score={"confidence": 87},
                iterations=1,
                converged=True,
                history=[],
            ),
            None,
        )


def _make_detected_entity(text: str, surface: str, etype: str = "PER") -> DetectedEntity:
    start = text.index(surface)
    end = start + len(surface)
    return DetectedEntity(
        start=start,
        end=end,
        surface=surface,
        etype=etype,
        source="stub",
        score=1.0,
        metadata={},
    )


def test_anonymize_text_without_llm() -> None:
    text = "Jean Dupont travaille chez ACME."
    secret = "top_secret"
    entity = _make_detected_entity(text, "Jean Dupont")

    result = anonymize_text(
        text,
        scope_id="ticket-123",
        secret_salt=secret,
        level="L0",
        detection_service=StubDetectionService([entity]),
        generalization_service=StubGeneralizationService(),
        llm_service=None,
    )

    anonymized = result["anonymized_text"]
    assert anonymized.endswith("travaille chez ACME."), "Expected wording preserved after pseudonymization"
    assert re.match(r"^\[PER_[A-Z]{3}\]", anonymized)

    audit = result["audit"]
    assert audit["paraphrase_applied"] is False
    assert audit["rupta_applied"] is False
    assert "llm_audit" not in audit
    assert audit["llm_errors"] == []

    replacements = audit["replacements"]
    assert len(replacements) == 1
    assert replacements[0]["surface"] == "Jean Dupont"

    evaluation = result["evaluation"]
    assert evaluation["is_valid"] is True
    assert evaluation["metrics"]["entities_detected"] == 1
    assert evaluation["warnings"] == []


def test_anonymize_text_with_llm_pipeline() -> None:
    text = "Alice a appelé Bob."
    secret = "another_secret"
    entity = _make_detected_entity(text, "Alice")

    generalization = Generalization(
        start=0,
        end=0,
        surface="",
        replacement="",
        etype="DATE",
        policy_rule="month",
    )

    overrides = {
        "paraphrase_temperature": 0.2,
        "llm_models": {"detect": "custom/model"},
        "rupta_ground_truth_people": "Alice",
        "rupta_ground_truth_label": "caller",
    }

    result = anonymize_text(
        text,
        scope_id="batch-456",
        secret_salt=secret,
        level="L1",
        overrides=overrides,
        detection_service=StubDetectionService([entity]),
        generalization_service=StubGeneralizationService([generalization]),
        llm_service=StubLLMService(),
    )

    anonymized = result["anonymized_text"]
    assert anonymized.endswith(" | paraphrased | optimized"), "Expected LLM steps reflected in final text"
    assert anonymized.startswith("[PER_")

    audit = result["audit"]
    assert audit["paraphrase_applied"] is True
    assert audit["rupta_applied"] is True
    assert audit["llm_audit"]["risk_score"] == 13
    assert audit["rupta_metrics"]["converged"] is True
    assert audit["metadata"]["llm_models"]["detect"] == "stub/model"

    replacements = audit["replacements"]
    assert len(replacements) == 1

    generalizations = audit["generalizations"]
    assert len(generalizations) == 1
    assert generalizations[0]["policy_rule"] == "month"

    evaluation = result["evaluation"]
    assert evaluation["is_valid"] is True
    assert evaluation["metrics"]["entities_detected"] == 1
    assert evaluation["warnings"] == []


@pytest.mark.parametrize(
    "tags,expected",
    [
        (None, []),
        ([], []),
        (["email", "phone"], ["EMAIL", "PHONE"]),
    ],
)
def test_skip_regex_tags_reflected_in_audit_metadata(tags: Optional[List[str]], expected: List[str]) -> None:
    text = "Charlie"
    secret = "secret"
    entity = _make_detected_entity(text, "Charlie")

    result = anonymize_text(
        text,
        scope_id="scope",
        secret_salt=secret,
        level="L0",
        overrides={"skip_regex_tags": tags} if tags is not None else None,
        detection_service=StubDetectionService([entity]),
        generalization_service=StubGeneralizationService(),
        llm_service=None,
    )

    metadata_tags = result["audit"]["metadata"]["skip_regex_tags"]
    assert metadata_tags == expected
