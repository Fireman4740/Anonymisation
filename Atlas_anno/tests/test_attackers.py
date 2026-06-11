from __future__ import annotations

import unittest

from atlas_anno.anonymization.baselines import anonymize_documents
from atlas_anno.annotation.preannotator import build_gold_annotations, build_predicted_annotations
from atlas_anno.attacks.structured import run_attack
from atlas_anno.generation.character_builder import build_characters
from atlas_anno.generation.scenario_planner import build_candidate_pools, build_scenarios
from atlas_anno.generation.text_generator import build_documents
from atlas_anno.generation.world_builder import build_worlds


class AttackersTest(unittest.TestCase):
    def test_structured_attack_returns_ranked_candidates(self) -> None:
        worlds = build_worlds(1, seed=1)
        characters = build_characters(worlds, per_world=6, seed=2)
        scenarios = build_scenarios(characters, documents=6, seed=3)
        candidate_pools = {character.person_id: build_candidate_pools(character, characters) for character in characters}
        documents = build_documents(worlds, characters, scenarios, candidate_pools)
        for document in documents:
            gold = build_gold_annotations(document)
            document.annotations = build_predicted_annotations(document, gold, mode="disabled")
        from atlas_anno.storage import save_documents, save_anonymization_results

        save_documents(documents, annotated=True)
        save_anonymization_results("masking", anonymize_documents(documents, "masking"))
        results = run_attack(documents, {character.person_id: character for character in characters}, "masking")
        self.assertIn(len(results), {len(documents), len(documents) * 3})
        self.assertTrue(all(result.top_k for result in results))

