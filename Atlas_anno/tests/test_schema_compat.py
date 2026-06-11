from __future__ import annotations

import unittest

from atlas_anno.records import (
    character_from_dict,
    dataset_batch_manifest_from_dict,
    document_from_dict,
    grounded_mention_from_dict,
    scenario_from_dict,
)
from atlas_anno.schemas import SCHEMA_VERSION

# Lignes figées au format v1 (avant l'ajout des champs v2) : elles doivent
# rester chargeables telles quelles.
V1_STYLE_PROFILE = {
    "formality": "medium",
    "signature_pattern": "thanks_name",
    "verbosity": "short",
    "emoji_usage": "none",
    "favorite_connectors": ["donc"],
    "jargon_pattern": "support_ops",
}

V1_CHARACTER = {
    "person_id": "person_w01_001",
    "full_name": "Nadia Mercier",
    "email": "nadia.mercier@atlas-services.example",
    "phone": "+33 6 00 12 34",
    "username": "nadia.mercier",
    "account_id": "ACC-01-0001",
    "language": "fr",
    "country": "France",
    "location": "Paris",
    "age_range": "25-29",
    "gender": "female",
    "nationality": "France",
    "organization_id": "org_01",
    "department": "AI Solutions",
    "team": "LLM Ops",
    "role": "AI Support Engineer",
    "seniority": "mid",
    "tenure_years": 1,
    "degrees": ["Master"],
    "skills": ["python"],
    "certifications": ["azure-ai-architect"],
    "rare_traits": [],
    "events": ["incident_1178_mars"],
    "sensitive_attributes": [],
    "style_profile": V1_STYLE_PROFILE,
}

V1_SCENARIO = {
    "scenario_id": "scn_0001",
    "domain": "support_ticket",
    "unit_type": "single_message",
    "language": "fr",
    "author_id": "person_w01_001",
    "recipient_role": "service_desk",
    "document_goal": "request_help",
    "difficulty": "easy",
    "required_signals": ["ROLE", "TEAM"],
    "implicit_signals": ["JARGON_PATTERN"],
    "include_signature": False,
    "include_direct_identifiers": True,
    "include_sensitive": False,
    "urgency": "medium",
    "noise_level": "medium",
    "split": "train",
}

V1_MENTION = {
    "label": "ROLE",
    "canonical_value": "AI Support Engineer",
    "snippet": "AI Support Engineer",
    "occurrence_hint": 1,
}

V1_DOCUMENT = {
    "doc_id": "doc_0001",
    "domain": "support_ticket",
    "unit_type": "single_message",
    "language": "fr",
    "author_id": "person_w01_001",
    "target_person_ids": ["person_w01_001"],
    "world_id": "world_01",
    "split": "train",
    "text": "Bonjour, je suis AI Support Engineer dans l'equipe LLM Ops.",
    "scenario": V1_SCENARIO,
    "candidate_pools": {
        "public": ["person_w01_001"],
        "org_internal": ["person_w01_001"],
        "insider": ["person_w01_001"],
    },
    "annotations": {
        "spans": [
            {"start": 17, "end": 36, "label": "ROLE", "text": "AI Support Engineer"}
        ],
        "relations": [],
        "doc_labels": {},
        "human_review_required": False,
    },
    "metadata": {"difficulty": "easy"},
}

V1_MANIFEST = {
    "batch_name": "pilot_100",
    "worlds_total": 3,
    "characters_total": 36,
    "documents_total": 100,
    "llm_mode": "disabled",
    "artifacts": {"raw_docs": "data/raw_docs/raw_docs.jsonl"},
    "stats": {"splits": {"train": 60}},
}


class SchemaCompatTest(unittest.TestCase):
    def test_schema_version_defined(self) -> None:
        self.assertTrue(SCHEMA_VERSION)

    def test_v1_character_round_trips(self) -> None:
        character = character_from_dict(V1_CHARACTER)
        self.assertEqual(character.person_id, "person_w01_001")
        self.assertEqual(character.style_profile.formality, "medium")

    def test_v1_scenario_round_trips(self) -> None:
        scenario = scenario_from_dict(V1_SCENARIO)
        self.assertEqual(scenario.scenario_id, "scn_0001")
        self.assertEqual(scenario.required_signals, ["ROLE", "TEAM"])

    def test_v1_grounded_mention_round_trips(self) -> None:
        mention = grounded_mention_from_dict(V1_MENTION)
        self.assertEqual(mention.label, "ROLE")
        self.assertEqual(mention.occurrence_hint, 1)

    def test_v1_document_round_trips(self) -> None:
        document = document_from_dict(V1_DOCUMENT)
        self.assertEqual(document.doc_id, "doc_0001")
        self.assertEqual(document.scenario.domain, "support_ticket")
        self.assertEqual(len(document.annotations.spans), 1)

    def test_v1_manifest_round_trips(self) -> None:
        manifest = dataset_batch_manifest_from_dict(V1_MANIFEST)
        self.assertEqual(manifest.batch_name, "pilot_100")

    def test_unknown_fields_are_tolerated(self) -> None:
        payload = dict(V1_DOCUMENT)
        payload["future_field"] = {"anything": 1}
        document = document_from_dict(payload)
        self.assertEqual(
            document.metadata["_unknown_fields"], {"future_field": {"anything": 1}}
        )

        scenario_payload = dict(V1_SCENARIO)
        scenario_payload["future_field"] = "x"
        scenario = scenario_from_dict(scenario_payload)
        self.assertEqual(scenario.scenario_id, "scn_0001")

        mention_payload = dict(V1_MENTION)
        mention_payload["future_field"] = "x"
        mention = grounded_mention_from_dict(mention_payload)
        self.assertEqual(mention.label, "ROLE")


if __name__ == "__main__":
    unittest.main()
