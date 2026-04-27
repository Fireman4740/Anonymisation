from __future__ import annotations

import unittest

from atlas_anno.annotation.preannotator import build_gold_annotations
from atlas_anno.generation.character_builder import build_characters
from atlas_anno.generation.scenario_planner import build_scenarios
from atlas_anno.generation.text_generator import build_documents
from atlas_anno.generation.world_builder import build_worlds
from atlas_anno.generation.scenario_planner import build_candidate_pools
from atlas_anno.surface_grounding import document_surface_grounding, resolve_grounded_mention


class GenerationTest(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        worlds_a = build_worlds(1, seed=5)
        worlds_b = build_worlds(1, seed=5)
        self.assertEqual(worlds_a[0].organization_name, worlds_b[0].organization_name)

        characters_a = build_characters(worlds_a, per_world=5, seed=9)
        characters_b = build_characters(worlds_b, per_world=5, seed=9)
        self.assertEqual(characters_a[0].full_name, characters_b[0].full_name)

        scenarios = build_scenarios(characters_a, documents=12, seed=11)
        candidate_pools = {character.person_id: build_candidate_pools(character, characters_a) for character in characters_a}
        documents = build_documents(worlds_a, characters_a, scenarios, candidate_pools)
        self.assertEqual(len(documents), 12)
        self.assertTrue(all(document.text for document in documents))
        self.assertTrue(all(document.metadata.get("surface_grounding") for document in documents))
        self.assertTrue(any(document.split == "test_hard" for document in documents))

    def test_generated_text_is_humanized_and_grounded(self) -> None:
        worlds = build_worlds(1, seed=5)
        characters = build_characters(worlds, per_world=6, seed=9)
        scenarios = build_scenarios(characters, documents=12, seed=11)
        candidate_pools = {character.person_id: build_candidate_pools(character, characters) for character in characters}
        documents = build_documents(worlds, characters, scenarios, candidate_pools)

        self.assertTrue(any(document.domain == "support_ticket" for document in documents))
        self.assertTrue(any(document.domain == "email" for document in documents))
        self.assertTrue(all("request_help" not in document.text for document in documents))
        self.assertTrue(all("incident_" not in document.text for document in documents))
        self.assertTrue(all("support_ops" not in document.text for document in documents))
        self.assertTrue(all("only_" not in document.text for document in documents))

        for document in documents:
            grounding = document_surface_grounding(document)
            self.assertTrue(grounding)
            for mention in grounding:
                self.assertIsNotNone(resolve_grounded_mention(document.text, mention))

            gold = build_gold_annotations(document)
            self.assertTrue(gold.spans)

    def test_scenarios_cover_more_problem_variety(self) -> None:
        worlds = build_worlds(2, seed=5)
        characters = build_characters(worlds, per_world=8, seed=9)
        scenarios = build_scenarios(characters, documents=48, seed=11)

        support_goals = {scenario.document_goal for scenario in scenarios if scenario.domain == "support_ticket"}
        email_goals = {scenario.document_goal for scenario in scenarios if scenario.domain == "email"}

        self.assertGreaterEqual(len(support_goals), 6)
        self.assertGreaterEqual(len(email_goals), 6)
