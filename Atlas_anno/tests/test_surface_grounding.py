from __future__ import annotations

import unittest

from atlas_anno.annotation.preannotator import build_gold_annotations, build_predicted_annotations
from atlas_anno.evaluation.privacy import evaluate_privacy
from atlas_anno.evaluation.utility import evaluate_utility
from atlas_anno.generation.character_builder import build_characters
from atlas_anno.generation.scenario_planner import build_candidate_pools, build_scenarios
from atlas_anno.generation.text_generator import build_documents
from atlas_anno.generation.world_builder import build_worlds
from atlas_anno.schemas import AnnotationBundle, AnonymizationResult, CandidatePools, DocumentRecord, ScenarioSpec
from atlas_anno.surface_grounding import document_surface_grounding


class SurfaceGroundingTest(unittest.TestCase):
    def _documents(self):
        worlds = build_worlds(1, seed=5)
        characters = build_characters(worlds, per_world=6, seed=9)
        scenarios = build_scenarios(characters, documents=6, seed=11)
        candidate_pools = {character.person_id: build_candidate_pools(character, characters) for character in characters}
        return build_documents(worlds, characters, scenarios, candidate_pools)

    def test_gold_annotations_fallback_to_legacy_signal_values(self) -> None:
        document = self._documents()[0]
        surface_grounding = document.metadata.pop("surface_grounding")
        self.assertTrue(surface_grounding)

        gold = build_gold_annotations(document)

        self.assertTrue(gold.spans)
        self.assertTrue(any(span.label == "ROLE" for span in gold.spans))

    def test_privacy_eval_reads_surface_grounding_mentions(self) -> None:
        document = next(document for document in self._documents() if "EMAIL" in document.metadata.get("signal_values", {}))
        grounding = document_surface_grounding(document)
        self.assertTrue(grounding)

        privacy = evaluate_privacy(
            [document],
            {
                document.doc_id: AnonymizationResult(
                    doc_id=document.doc_id,
                    strategy="masking",
                    anonymized_text="document entier remplace",
                    actions_performed=["masking"],
                    rationale="unit test",
                    estimated_privacy_gain=1.0,
                    estimated_utility_loss=1.0,
                    metadata={},
                )
            },
        )

        self.assertEqual(privacy["summary"]["direct_removal"], 1.0)

    def test_utility_eval_handles_humanized_texts(self) -> None:
        documents = self._documents()
        results = {}
        for document in documents:
            results[document.doc_id] = AnonymizationResult(
                doc_id=document.doc_id,
                strategy="copy",
                anonymized_text=document.text,
                actions_performed=[],
                rationale="identity",
                estimated_privacy_gain=0.0,
                estimated_utility_loss=0.0,
                metadata={},
            )

        report = evaluate_utility(documents, results)

        self.assertGreater(report["summary"]["domain_accuracy"], 0.5)

    def test_preannotation_lexical_fallback_captures_business_entities(self) -> None:
        document = DocumentRecord(
            doc_id="doc_lexical",
            domain="support_ticket",
            unit_type="single_message",
            language="fr",
            author_id="p_0001",
            target_person_ids=["p_0001"],
            world_id="world_01",
            split="train",
            text=(
                "Bonjour, MergeFlow bloque dans OpsConsole chez Atlas Services. "
                "Escalade Support a relance Identity Access et Data Platform. "
                "Meridian Cloud a aussi recu le meme signal."
            ),
            scenario=ScenarioSpec(
                scenario_id="scenario_lexical",
                domain="support_ticket",
                unit_type="single_message",
                language="fr",
                author_id="p_0001",
                recipient_role="service_desk",
                document_goal="request_help",
                difficulty="medium",
                required_signals=["TEAM"],
                implicit_signals=["JARGON_PATTERN"],
                include_signature=False,
                include_direct_identifiers=False,
                include_sensitive=False,
                urgency="medium",
                noise_level="low",
                split="train",
            ),
            candidate_pools=CandidatePools(),
            annotations=AnnotationBundle(),
            metadata={
                "difficulty": "medium",
                "signal_values": {
                    "ORG_NAME_STRONG": ["Atlas Services", "Meridian Cloud"],
                    "TEAM": ["Escalade Support", "Identity Access", "Data Platform"],
                    "PRODUCT_CONTEXT": ["MergeFlow", "OpsConsole"],
                },
            },
        )

        gold = build_gold_annotations(document)
        predicted = build_predicted_annotations(document, gold, mode="disabled")
        spans = {(span.label, span.text) for span in predicted.spans}

        self.assertIn(("ORG_NAME_STRONG", "Atlas Services"), spans)
        self.assertIn(("ORG_NAME_STRONG", "Meridian Cloud"), spans)
        self.assertIn(("TEAM", "Escalade Support"), spans)
        self.assertIn(("TEAM", "Identity Access"), spans)
        self.assertIn(("TEAM", "Data Platform"), spans)
        self.assertIn(("PRODUCT_CONTEXT", "MergeFlow"), spans)
        self.assertIn(("PRODUCT_CONTEXT", "OpsConsole"), spans)
