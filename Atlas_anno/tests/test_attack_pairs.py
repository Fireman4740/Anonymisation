from __future__ import annotations

import unittest

from atlas_anno.attacks.pairs import build_attack_pairs
from atlas_anno.schemas import AnnotationBundle, CandidatePools, CharacterProfile, DocumentRecord, ScenarioSpec, StyleProfile


def _character(index: int, team: str = "Ops", role: str = "Analyste") -> CharacterProfile:
    return CharacterProfile(
        person_id=f"person_{index:03d}",
        full_name=f"Personne {index}",
        email=f"personne{index}@example.test",
        phone="+33 6 00 00 00 00",
        username=f"personne{index}",
        account_id=f"ACC-{index:03d}",
        language="fr",
        country="France",
        location="Paris",
        age_range="35-39",
        gender="female",
        nationality="France",
        organization_id="org_01",
        department="Data",
        team=team,
        role=role,
        seniority="mid",
        tenure_years=4,
        degrees=["Master"],
        skills=["python"],
        certifications=[],
        rare_traits=[f"trait_{index}"],
        events=[],
        sensitive_attributes=[],
        style_profile=StyleProfile("medium", "thanks_name", "short", "none"),
    )


def _document(index: int, author_id: str, difficulty: str = "medium") -> DocumentRecord:
    scenario = ScenarioSpec(
        scenario_id=f"scenario_{index:03d}",
        domain="email",
        unit_type="single_message",
        language="fr",
        author_id=author_id,
        recipient_role="ops",
        document_goal="coordination",
        difficulty=difficulty,
    )
    return DocumentRecord(
        doc_id=f"doc_{index:03d}",
        domain="email",
        unit_type="single_message",
        language="fr",
        author_id=author_id,
        target_person_ids=[author_id],
        world_id="world_001",
        split="train",
        text="",
        scenario=scenario,
        candidate_pools=CandidatePools(),
        annotations=AnnotationBundle(),
        metadata={"difficulty": difficulty, "mention_difficulty": {"explicit_easy": 1}},
    )


class AttackPairsTest(unittest.TestCase):
    def test_build_attack_pairs_is_deterministic_and_has_aux_levels(self) -> None:
        characters = [_character(index, team="Ops" if index < 5 else "Support") for index in range(8)]
        documents = [_document(index, characters[index].person_id) for index in range(4)]

        pairs_a = build_attack_pairs(documents, characters, seed=53)
        pairs_b = build_attack_pairs(documents, characters, seed=53)

        self.assertEqual([pair.__dict__ for pair in pairs_a], [pair.__dict__ for pair in pairs_b])
        self.assertEqual(len(pairs_a), len(documents) * 3)
        self.assertEqual({pair.aux_knowledge.level for pair in pairs_a}, {"none", "partial", "strong"})
        self.assertTrue(all(pair.target_person_id in pair.candidate_pool for pair in pairs_a))
        self.assertTrue(all(pair.metadata.get("difficulty_mode") for pair in pairs_a))


if __name__ == "__main__":
    unittest.main()
