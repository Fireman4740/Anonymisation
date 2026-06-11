from __future__ import annotations

import unittest

from atlas_anno.generation.auditor import audit_document
from atlas_anno.schemas import (
    AnnotationBundle,
    CandidatePools,
    CharacterProfile,
    DocumentRecord,
    GroundedMention,
    ScenarioSpec,
    StyleProfile,
)


def _character() -> CharacterProfile:
    return CharacterProfile(
        person_id="person_001",
        full_name="Camille Martin",
        email="camille.martin@example.test",
        phone="+33 6 00 00 00 00",
        username="cmartin",
        account_id="ACC-01",
        language="fr",
        country="France",
        location="Lyon",
        age_range="35-39",
        gender="female",
        nationality="France",
        organization_id="org_01",
        department="Data",
        team="Ops",
        role="Analyste",
        seniority="mid",
        tenure_years=4,
        degrees=["Master"],
        skills=["python"],
        certifications=[],
        rare_traits=[],
        events=[],
        sensitive_attributes=["HEALTH"],
        style_profile=StyleProfile("medium", "thanks_name", "short", "none"),
    )


def _document(text: str, grounding: list[GroundedMention]) -> DocumentRecord:
    scenario = ScenarioSpec(
        scenario_id="scenario_001",
        domain="email",
        unit_type="single_message",
        language="fr",
        author_id="person_001",
        recipient_role="ops",
        document_goal="coordination",
        difficulty="hard",
        required_signals=["ROLE"],
    )
    return DocumentRecord(
        doc_id="doc_001",
        domain="email",
        unit_type="single_message",
        language="fr",
        author_id="person_001",
        target_person_ids=["person_001"],
        world_id="world_001",
        split="train",
        text=text,
        scenario=scenario,
        candidate_pools=CandidatePools(),
        annotations=AnnotationBundle(),
        metadata={
            "signal_values": {"ROLE": ["Analyste"]},
            "surface_grounding": [mention.__dict__ for mention in grounding],
        },
    )


class AuditorExtendedTest(unittest.TestCase):
    def test_unintended_leak_is_reported(self) -> None:
        author = _character()
        document = _document(
            "Je suis Analyste. Mon email perso est camille.martin@example.test.",
            [GroundedMention("ROLE", "Analyste", "Analyste")],
        )

        issues = audit_document(document, author)

        self.assertIn("unintended_leak:EMAIL", issues)
        self.assertTrue(document.metadata["human_review_required"])

    def test_span_boundary_and_difficulty_violation_are_reported(self) -> None:
        author = _character()
        document = _document(
            "banana Camille Martin",
            [
                GroundedMention("ROLE", "ana", "ana"),
                GroundedMention("PERSON_NAME", "Camille Martin", "Camille Martin", difficulty_mode="explicit_hard"),
            ],
        )

        issues = audit_document(document, author)

        self.assertIn("span_boundary:ROLE", issues)
        self.assertIn("difficulty_violation:PERSON_NAME", issues)


if __name__ == "__main__":
    unittest.main()
