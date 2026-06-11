from __future__ import annotations

import unittest
from unittest.mock import patch

from atlas_anno.anonymization.baselines import anonymize_documents
from atlas_anno.annotation.preannotator import build_gold_annotations, build_predicted_annotations
from atlas_anno.generation.character_builder import build_characters
from atlas_anno.generation.scenario_planner import build_candidate_pools, build_scenarios
from atlas_anno.generation.text_generator import build_documents
from atlas_anno.generation.world_builder import build_worlds
from atlas_anno.schemas import LLMRunMeta


class AnonymizationTest(unittest.TestCase):
    def test_masking_removes_direct_identifier(self) -> None:
        worlds = build_worlds(1, seed=1)
        characters = build_characters(worlds, per_world=4, seed=2)
        scenarios = build_scenarios(characters, documents=4, seed=3)
        candidate_pools = {character.person_id: build_candidate_pools(character, characters) for character in characters}
        documents = build_documents(worlds, characters, scenarios, candidate_pools)
        document = documents[0]
        gold = build_gold_annotations(document)
        document.annotations = build_predicted_annotations(document, gold, mode="disabled")
        results = anonymize_documents([document], "masking")
        self.assertNotIn(characters[0].email, results[0].anonymized_text)
        self.assertGreaterEqual(results[0].estimated_privacy_gain, 0.0)

    def test_rewrite_balanced_llm_mode_uses_structured_response(self) -> None:
        worlds = build_worlds(1, seed=1)
        characters = build_characters(worlds, per_world=4, seed=2)
        scenarios = build_scenarios(characters, documents=4, seed=3)
        candidate_pools = {character.person_id: build_candidate_pools(character, characters) for character in characters}
        documents = build_documents(worlds, characters, scenarios, candidate_pools)
        document = documents[0]
        gold = build_gold_annotations(document)
        predicted = build_predicted_annotations(document, gold, mode="disabled")
        document.annotations.spans = predicted.spans
        document.metadata["predicted_annotations"] = {
            "spans": [{"start": span.start, "end": span.end, "label": span.label, "text": span.text, "confidence": span.confidence, "source": span.source} for span in predicted.spans]
        }

        def fake_complete_json(self, *, step_name, prompt_spec, user_prompt, model, validator, fallback_value, temperature, allow_fallback=True):
            payload = validator(
                {
                    "anonymized_text": "Texte réécrit",
                    "actions_performed": ["rewrite_balanced:EMAIL"],
                    "rationale": "llm rewrite",
                    "estimated_privacy_gain": 0.9,
                    "estimated_utility_loss": 0.1,
                }
            )
            return payload, LLMRunMeta(
                step_name=step_name,
                model=model,
                prompt_version=prompt_spec.version,
                llm_used=True,
                fallback_used=False,
                retry_count=0,
                validation_errors=[],
                latency_ms=1,
                estimated_cost=0.0,
            )

        with patch("atlas_anno.llm.OpenRouterClient.complete_json", new=fake_complete_json):
            results = anonymize_documents([document], "rewrite_balanced", mode="llm")
        self.assertEqual(results[0].anonymized_text, "Texte réécrit")
        self.assertIn("llm_run", results[0].metadata)
