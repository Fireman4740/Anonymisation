from __future__ import annotations

import unittest

from atlas_anno.evaluation.calibration import run_difficulty_calibration
from atlas_anno.schemas import (
    AnnotationBundle,
    AttackPair,
    AuxiliaryKnowledge,
    CandidatePools,
    CharacterProfile,
    DocumentRecord,
    ScenarioSpec,
    StyleProfile,
)


def _character(person_id: str, full_name: str, role: str = "Analyste") -> CharacterProfile:
    return CharacterProfile(
        person_id=person_id,
        full_name=full_name,
        email=f"{person_id}@example.test",
        phone="+33 6 00 00 00 00",
        username=person_id,
        account_id=f"ACC-{person_id}",
        language="fr",
        country="France",
        location="Paris",
        age_range="35-39",
        gender="female",
        nationality="France",
        organization_id="org_01",
        department="Data",
        team="Ops",
        role=role,
        seniority="mid",
        tenure_years=4,
        degrees=["Master"],
        skills=["python"],
        certifications=[],
        rare_traits=[],
        events=[],
        sensitive_attributes=[],
        style_profile=StyleProfile("medium", "thanks_name", "short", "none"),
    )


def _document(doc_id: str, author_id: str, difficulty: str, mode: str, text: str) -> DocumentRecord:
    scenario = ScenarioSpec(
        scenario_id=f"scenario_{doc_id}",
        domain="email",
        unit_type="single_message",
        language="fr",
        author_id=author_id,
        recipient_role="ops",
        document_goal="coordination",
        difficulty=difficulty,
    )
    return DocumentRecord(
        doc_id=doc_id,
        domain="email",
        unit_type="single_message",
        language="fr",
        author_id=author_id,
        target_person_ids=[author_id],
        world_id="world_001",
        split="train",
        text=text,
        scenario=scenario,
        candidate_pools=CandidatePools(),
        annotations=AnnotationBundle(),
        metadata={"difficulty": difficulty, "mention_difficulty": {mode: 1}},
    )


def _pair(document: DocumentRecord, target: str, mode: str) -> AttackPair:
    return AttackPair(
        pair_id=f"{document.doc_id}_none",
        doc_id=document.doc_id,
        target_person_id=target,
        target_attributes={"person_id": target},
        aux_knowledge=AuxiliaryKnowledge("none", {}),
        candidate_pool=["person_001", "person_002"],
        difficulty=document.scenario.difficulty,
        metadata={"difficulty_mode": mode, "aux_level": "none"},
    )


class CalibrationTest(unittest.TestCase):
    def test_calibration_passes_on_monotone_fixture(self) -> None:
        characters = {
            "person_001": _character("person_001", "Alice Martin"),
            "person_002": _character("person_002", "Zoé Bernard"),
        }
        documents = [
            _document("doc_easy", "person_001", "easy", "explicit_easy", "Alice Martin"),
            _document("doc_medium_ok", "person_001", "medium", "explicit_hard", "Alice Martin"),
            _document("doc_medium_fail", "person_002", "medium", "explicit_hard", ""),
            _document("doc_hard", "person_002", "hard", "implicit", ""),
        ]
        pairs = [_pair(document, document.author_id, next(iter(document.metadata["mention_difficulty"]))) for document in documents]

        report = run_difficulty_calibration(documents, characters, pairs)

        self.assertTrue(report["passed"])
        self.assertEqual(report["by_difficulty"]["easy"]["top1"], 1.0)
        self.assertEqual(report["by_difficulty"]["hard"]["top1"], 0.0)

    def test_calibration_fails_when_easy_is_too_low(self) -> None:
        characters = {
            "person_001": _character("person_001", "Alice Martin"),
            "person_002": _character("person_002", "Zoé Bernard"),
        }
        documents = [_document("doc_easy", "person_002", "easy", "explicit_easy", "")]
        pairs = [_pair(documents[0], "person_002", "explicit_easy")]

        report = run_difficulty_calibration(documents, characters, pairs)

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["easy_raw_minimum"]["passed"])


if __name__ == "__main__":
    unittest.main()
