from __future__ import annotations

import unittest

from atlas_anno.generation.llm_text_generator import generate_llm_texts
from atlas_anno.io import serialize
from atlas_anno.schemas import (
    AnnotationBundle,
    CandidatePools,
    CharacterProfile,
    DocumentRecord,
    GroundedMention,
    LLMRunMeta,
    ScenarioSpec,
    StyleProfile,
    World,
)


class _Settings:
    atlas_model_creative = "creative-test"


class _FakeClient:
    def __init__(self, payloads):
        self.settings = _Settings()
        self.payloads = list(payloads)
        self.calls = 0

    def complete_json(self, *, step_name, prompt_spec, user_prompt, model, validator, fallback_value, temperature, allow_fallback=True):
        self.calls += 1
        payload = self.payloads.pop(0) if self.payloads else fallback_value
        value = validator(payload)
        return value, LLMRunMeta(
            step_name=step_name,
            model=model,
            prompt_version=prompt_spec.version,
            llm_used=payload is not fallback_value,
            fallback_used=payload is fallback_value,
            retry_count=0,
        )


def _seed_document():
    world = World(
        world_id="world_001",
        language="fr",
        organization_id="org_01",
        organization_name="Atlas Services",
        departments=["Data"],
        teams=["Ops"],
        projects=["migration_sso"],
        products=["AtlasDesk"],
        incidents=[],
        calendar_events=["audit_iso_q2"],
    )
    character = CharacterProfile(
        person_id="person_001",
        full_name="Camille Martin",
        email="camille.martin@example.test",
        phone="+33 6 00 00 00 00",
        username="cmartin",
        account_id="ACC-001",
        language="fr",
        country="France",
        location="Paris",
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
        sensitive_attributes=[],
        style_profile=StyleProfile("medium", "thanks_name", "short", "none"),
    )
    scenario = ScenarioSpec(
        scenario_id="scenario_001",
        domain="email",
        unit_type="single_message",
        language="fr",
        author_id=character.person_id,
        recipient_role="ops",
        document_goal="coordination",
        difficulty="hard",
        required_signals=["ROLE", "TEAM"],
    )
    grounding = [
        GroundedMention("ROLE", "Analyste", "Analyste"),
        GroundedMention("TEAM", "Ops", "Ops"),
    ]
    document = DocumentRecord(
        doc_id="doc_001",
        domain="email",
        unit_type="single_message",
        language="fr",
        author_id=character.person_id,
        target_person_ids=[character.person_id],
        world_id=world.world_id,
        split="train",
        text="Analyste Ops",
        scenario=scenario,
        candidate_pools=CandidatePools(),
        annotations=AnnotationBundle(),
        metadata={
            "difficulty": "hard",
            "signal_values": {"ROLE": ["Analyste"], "TEAM": ["Ops"]},
            "surface_grounding": [mention.__dict__ for mention in grounding],
            "mention_difficulty": {"explicit_easy": 2},
        },
    )
    return [world], [character], document


class LLMTextGeneratorTest(unittest.TestCase):
    def test_missing_snippet_triggers_repair_then_accepts_valid_payload(self) -> None:
        worlds, characters, document = _seed_document()
        planned = document.metadata["surface_grounding"]
        valid_text = " ".join(item["snippet"] for item in planned)
        client = _FakeClient(
            [
                {"text": "texte incomplet", "grounding": []},
                {"text": valid_text, "grounding": planned},
            ]
        )

        generated, runs, stats = generate_llm_texts(
            [document],
            {character.person_id: character for character in characters},
            {world.world_id: world for world in worlds},
            client,
            "primary-fallback",
            {"batch_name": "test", "resume_enabled": False, "checkpoint_every": 1, "creative_workers": 1, "text_repair_retries": 2},
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(generated[0].text, valid_text)
        self.assertEqual(generated[0].metadata["text_repair_retries"], 1)
        self.assertFalse(runs[document.doc_id].fallback_used)
        self.assertEqual(stats["repair_retries"], 1)

    def test_exhausted_repairs_falls_back_to_deterministic_document(self) -> None:
        worlds, characters, document = _seed_document()
        original = serialize(document)
        client = _FakeClient([{"text": "toujours incomplet", "grounding": []}] * 3)

        generated, runs, _ = generate_llm_texts(
            [document],
            {character.person_id: character for character in characters},
            {world.world_id: world for world in worlds},
            client,
            "primary-fallback",
            {"batch_name": "test", "resume_enabled": False, "checkpoint_every": 1, "creative_workers": 1, "text_repair_retries": 1},
        )

        self.assertEqual(generated[0].text, original["text"])
        self.assertTrue(runs[document.doc_id].error)
        self.assertEqual(generated[0].metadata["text_generation_mode"], "error")


if __name__ == "__main__":
    unittest.main()
