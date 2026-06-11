from __future__ import annotations

import unittest
from unittest.mock import patch

from atlas_anno.generation.llm_text_generator import generate_llm_texts
from atlas_anno.generation.naturalization import (
    build_persona_block,
    build_style_directives,
    variant_order,
)
from tests.test_llm_text_generator import _FakeClient, _seed_document


def _runtime(repairs: int = 1):
    return {
        "batch_name": "test-nat",
        "resume_enabled": False,
        "checkpoint_every": 1,
        "creative_workers": 1,
        "text_repair_retries": repairs,
    }


def _patched_naturalization(mode: str, variants: int = 3):
    from atlas_anno.config import load_config

    config = load_config()
    config.defaults["generation"]["naturalization"] = {
        "mode": mode,
        "variants": variants,
        "temperature": 0.7,
    }
    return patch("atlas_anno.generation.llm_text_generator.load_config", return_value=config)


class StyleDirectivesTest(unittest.TestCase):
    def test_directives_use_effective_scenario_values(self) -> None:
        _, characters, document = _seed_document()
        document.scenario.register = "familier"
        document.scenario.address_form = "tu"
        directives = build_style_directives(characters[0], document.scenario)
        self.assertEqual(directives["register"], "familier")
        self.assertEqual(directives["address_form"], "tu")
        self.assertIn("typo_propensity", directives)

    def test_persona_block_only_exposes_planned_attributes(self) -> None:
        _, characters, document = _seed_document()
        persona = build_persona_block(characters[0], document)
        self.assertIn("role", persona)
        self.assertIn("team", persona)
        # Identifiants directs hors plan : absents du payload.
        self.assertNotIn("email", persona)
        self.assertNotIn("phone", persona)
        self.assertNotIn("full_name", persona)

    def test_directives_present_in_user_prompt(self) -> None:
        worlds, characters, document = _seed_document()
        document.scenario.register = "soutenu"
        document.scenario.address_form = "vous"
        captured = {}

        class _CapturingClient(_FakeClient):
            def complete_json(self, **kwargs):
                captured["user_prompt"] = kwargs["user_prompt"]
                return super().complete_json(**kwargs)

        planned = document.metadata["surface_grounding"]
        valid_text = " ".join(item["snippet"] for item in planned)
        client = _CapturingClient([{"text": valid_text, "grounding": planned}])
        generate_llm_texts(
            [document],
            {character.person_id: character for character in characters},
            {world.world_id: world for world in worlds},
            client,
            "primary-fallback",
            _runtime(),
        )
        self.assertIn("style_directives", captured["user_prompt"])
        self.assertIn("soutenu", captured["user_prompt"])
        self.assertIn("forbidden_unplanned_attribute_labels", captured["user_prompt"])


class VariantOrderTest(unittest.TestCase):
    def test_order_is_deterministic_per_doc_id(self) -> None:
        self.assertEqual(variant_order("doc_001", 5), variant_order("doc_001", 5))
        self.assertEqual(sorted(variant_order("doc_001", 5)), [0, 1, 2, 3, 4])

    def test_order_varies_across_doc_ids(self) -> None:
        orders = {tuple(variant_order(f"doc_{index:03d}", 6)) for index in range(20)}
        self.assertGreater(len(orders), 1)


class VerbalizedSamplingTest(unittest.TestCase):
    def test_invalid_variant_skipped_for_valid_one(self) -> None:
        worlds, characters, document = _seed_document()
        planned = document.metadata["surface_grounding"]
        valid_text = " ".join(item["snippet"] for item in planned)
        payload = {
            "variants": [
                {"probability": 0.6, "text": "variante invalide", "grounding": []},
                {"probability": 0.4, "text": valid_text, "grounding": planned},
            ]
        }
        with _patched_naturalization("verbalized", variants=2):
            client = _FakeClient([payload])
            generated, runs, _ = generate_llm_texts(
                [document],
                {character.person_id: character for character in characters},
                {world.world_id: world for world in worlds},
                client,
                "primary-fallback",
                _runtime(),
            )
        self.assertEqual(client.calls, 1)
        self.assertEqual(generated[0].text, valid_text)
        self.assertEqual(generated[0].metadata["text_generation_mode"], "llm")
        self.assertFalse(runs[document.doc_id].error)

    def test_verbalized_pick_is_stable_across_runs(self) -> None:
        texts = []
        for _ in range(2):
            worlds, characters, document = _seed_document()
            planned = document.metadata["surface_grounding"]
            base = " ".join(item["snippet"] for item in planned)
            payload = {
                "variants": [
                    {"probability": 0.5, "text": f"{base} — variante A", "grounding": planned},
                    {"probability": 0.3, "text": f"{base} — variante B", "grounding": planned},
                    {"probability": 0.2, "text": f"{base} — variante C", "grounding": planned},
                ]
            }
            with _patched_naturalization("verbalized", variants=3):
                client = _FakeClient([payload])
                generated, _, _ = generate_llm_texts(
                    [document],
                    {character.person_id: character for character in characters},
                    {world.world_id: world for world in worlds},
                    client,
                    "primary-fallback",
                    _runtime(),
                )
            texts.append(generated[0].text)
        self.assertEqual(texts[0], texts[1])

    def test_all_variants_invalid_marks_error(self) -> None:
        worlds, characters, document = _seed_document()
        payload = {
            "variants": [
                {"probability": 0.5, "text": "rien", "grounding": []},
                {"probability": 0.5, "text": "toujours rien", "grounding": []},
            ]
        }
        with _patched_naturalization("verbalized", variants=2):
            client = _FakeClient([payload, payload])
            generated, runs, _ = generate_llm_texts(
                [document],
                {character.person_id: character for character in characters},
                {world.world_id: world for world in worlds},
                client,
                "primary-fallback",
                _runtime(repairs=1),
            )
        self.assertEqual(generated[0].metadata["text_generation_mode"], "error")
        self.assertTrue(runs[document.doc_id].error)


if __name__ == "__main__":
    unittest.main()
