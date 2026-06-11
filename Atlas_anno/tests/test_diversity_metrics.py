from __future__ import annotations

import unittest
from typing import List

from atlas_anno.evaluation.diversity import (
    cell_coverage,
    distinct_n,
    embedding_metrics,
    evaluate_diversity,
    find_near_duplicates,
    self_bleu,
)
from atlas_anno.schemas import (
    AnnotationBundle,
    CandidatePools,
    DocumentRecord,
    ScenarioSpec,
    StyleProfile,
)


def _doc(
    doc_id: str,
    text: str,
    domain: str = "email",
    difficulty: str = "easy",
    register: str = "courant",
    goal: str = "info",
) -> DocumentRecord:
    scenario = ScenarioSpec(
        scenario_id=f"scenario_{doc_id}",
        domain=domain,
        unit_type="single_message",
        language="fr",
        author_id="p1",
        recipient_role="ops",
        document_goal=goal,
        difficulty=difficulty,
        register=register,
    )
    return DocumentRecord(
        doc_id=doc_id,
        domain=domain,
        unit_type="single_message",
        language="fr",
        author_id="p1",
        target_person_ids=["p1"],
        world_id="world_01",
        split="train",
        text=text,
        scenario=scenario,
        candidate_pools=CandidatePools(),
        annotations=AnnotationBundle(),
        metadata={"difficulty": difficulty, "register": register},
    )


# ---------------------------------------------------------------------------
# distinct_n
# ---------------------------------------------------------------------------

class DistinctNTest(unittest.TestCase):
    def test_identical_corpus_low_distinct(self) -> None:
        texts = ["bonjour monde"] * 20
        d1 = distinct_n(texts, 1)
        self.assertLess(d1["corpus"], 0.10)

    def test_diverse_corpus_high_distinct(self) -> None:
        texts = [f"token_{i} word_{i * 2} phrase_{i * 3}" for i in range(50)]
        d2 = distinct_n(texts, 2)
        self.assertGreater(d2["corpus"], 0.70)

    def test_empty_returns_zero(self) -> None:
        d = distinct_n([], 1)
        self.assertEqual(d["corpus"], 0.0)
        self.assertEqual(d["mean_per_doc"], 0.0)


# ---------------------------------------------------------------------------
# self_bleu
# ---------------------------------------------------------------------------

class SelfBleuTest(unittest.TestCase):
    def test_identical_corpus_score_high(self) -> None:
        texts = ["le chat mange la souris tous les jours depuis longtemps"] * 20
        score = self_bleu(texts)
        self.assertGreater(score, 0.90)

    def test_disjoint_corpus_score_low(self) -> None:
        texts = [
            f"zyx_{i} qwt_{i} vbn_{i} xyz_{i} abc_{i} def_{i} ghi_{i} jkl_{i}"
            for i in range(30)
        ]
        score = self_bleu(texts)
        self.assertLess(score, 0.15)

    def test_deterministic(self) -> None:
        texts = [f"mot_{i} autre_{i} chose_{i} truc_{i}" for i in range(50)]
        self.assertEqual(self_bleu(texts, seed=17), self_bleu(texts, seed=17))

    def test_single_text_returns_zero(self) -> None:
        self.assertEqual(self_bleu(["texte unique"]), 0.0)


# ---------------------------------------------------------------------------
# MinHash / find_near_duplicates
# ---------------------------------------------------------------------------

class MinHashTest(unittest.TestCase):
    def test_identical_texts_flagged(self) -> None:
        text = "le renard rapide saute par dessus le chien paresseux assis"
        texts = [text] * 5
        pairs = find_near_duplicates(texts, jaccard_threshold=0.80)
        self.assertGreater(len(pairs), 0)

    def test_different_texts_not_flagged(self) -> None:
        texts = [
            "le renard rapide saute par dessus le chien paresseux au soleil",
            "la lune brille intensément dans le ciel étoilé de la nuit",
            "le programmeur écrit du code Python tous les matins très tôt",
            "le médecin soigne ses patients à l'hôpital central de la ville",
        ]
        pairs = find_near_duplicates(texts, jaccard_threshold=0.80)
        self.assertEqual(pairs, [])

    def test_near_duplicate_flagged_non_related_not(self) -> None:
        base = "le renard rapide saute par dessus le chien paresseux au soleil brillant"
        near = base + " vers le nord"
        other = "la pluie tombe sur la ville de Paris pendant les mois d'automne"
        texts = [base, near, other]
        pairs = find_near_duplicates(texts, jaccard_threshold=0.70)
        doc_ids_in_pairs = {(i, j) for i, j, _ in pairs}
        self.assertIn((0, 1), doc_ids_in_pairs)
        self.assertNotIn((0, 2), doc_ids_in_pairs)
        self.assertNotIn((1, 2), doc_ids_in_pairs)

    def test_deterministic(self) -> None:
        texts = [f"texte_{i} contenu_{i * 3} mot_{i}" for i in range(20)]
        p1 = find_near_duplicates(texts)
        p2 = find_near_duplicates(texts)
        self.assertEqual(p1, p2)


# ---------------------------------------------------------------------------
# cell_coverage
# ---------------------------------------------------------------------------

class CellCoverageTest(unittest.TestCase):
    def test_single_cell_zero_entropy(self) -> None:
        docs = [_doc(f"d{i}", f"texte {i}") for i in range(10)]
        result = cell_coverage(docs)
        self.assertEqual(result["filled_cells"], 1)
        self.assertAlmostEqual(result["normalized_entropy"], 0.0, places=5)

    def test_multiple_cells_higher_entropy(self) -> None:
        docs = [
            _doc("d1", "ta", domain="email", difficulty="easy", register="courant", goal="info"),
            _doc("d2", "tb", domain="email", difficulty="hard", register="soutenu", goal="demande"),
            _doc("d3", "tc", domain="support_ticket", difficulty="easy", register="familier", goal="info"),
            _doc("d4", "td", domain="support_ticket", difficulty="hard", register="courant", goal="demande"),
        ]
        result = cell_coverage(docs)
        self.assertGreaterEqual(result["filled_cells"], 3)
        self.assertGreater(result["normalized_entropy"], 0.5)

    def test_unknown_register_when_absent(self) -> None:
        scenario = ScenarioSpec(
            scenario_id="s1",
            domain="email",
            unit_type="single_message",
            language="fr",
            author_id="p1",
            recipient_role="ops",
            document_goal="info",
            difficulty="easy",
        )
        doc = DocumentRecord(
            doc_id="no_register",
            domain="email",
            unit_type="single_message",
            language="fr",
            author_id="p1",
            target_person_ids=["p1"],
            world_id="w1",
            split="train",
            text="texte",
            scenario=scenario,
            candidate_pools=CandidatePools(),
            annotations=AnnotationBundle(),
            metadata={},
        )
        result = cell_coverage([doc])
        self.assertEqual(result["filled_cells"], 1)


# ---------------------------------------------------------------------------
# embedding_metrics (skipped hors env avec sentence-transformers)
# ---------------------------------------------------------------------------

class EmbeddingMetricsTest(unittest.TestCase):
    def test_skipped_without_deps(self) -> None:
        result = embedding_metrics(["texte a", "texte b", "texte c"])
        # Dans l'environnement de test, sentence-transformers/numpy sont absents.
        if result.get("skipped"):
            self.assertIn("reason", result)
        else:
            # Si disponibles, vérifier la structure.
            self.assertIn("dispersion", result)
            self.assertIn("vendi_score", result)


# ---------------------------------------------------------------------------
# evaluate_diversity (rapport + porte)
# ---------------------------------------------------------------------------

class EvaluateDiversityTest(unittest.TestCase):
    def test_identical_corpus_fails_thresholds(self) -> None:
        text = "le chat mange la souris et dort sur le canapé tous les jours"
        docs = [_doc(f"d{i}", text) for i in range(20)]
        report = evaluate_diversity(
            docs,
            {
                "enforce": True,
                "self_bleu_max": 0.90,
                "distinct_2_min": 0.15,
                "max_duplicate_rate": 0.02,
                "min_cell_coverage": 0.60,
            },
        )
        self.assertFalse(report["summary"]["passed"])
        self.assertGreater(len(report["summary"]["failures"]), 0)
        # duplicate_rate = 1.0 pour corpus identique
        self.assertAlmostEqual(report["summary"]["duplicate_rate"], 1.0, places=1)

    def test_diverse_corpus_distinct2_above_min(self) -> None:
        texts = [
            f"utilisateur_{i} rencontre un problème avec le ticket numéro {i * 13} "
            f"dans le système version {i + 1} pour la demande {i * 7} du département"
            for i in range(30)
        ]
        docs = [_doc(f"d{i}", texts[i]) for i in range(30)]
        report = evaluate_diversity(docs, {"distinct_2_min": 0.05})
        self.assertGreater(report["summary"]["distinct_2"], 0.05)

    def test_determinism_across_two_runs(self) -> None:
        texts = [f"texte_{i} contenu_{i * 3} mot_{i}" for i in range(30)]
        docs = [_doc(f"d{i}", texts[i]) for i in range(30)]
        r1 = evaluate_diversity(docs, {})
        r2 = evaluate_diversity(docs, {})
        self.assertEqual(r1["summary"]["self_bleu"], r2["summary"]["self_bleu"])
        self.assertEqual(r1["summary"]["duplicate_rate"], r2["summary"]["duplicate_rate"])
        self.assertEqual(r1["summary"]["distinct_2"], r2["summary"]["distinct_2"])

    def test_report_structure(self) -> None:
        docs = [_doc(f"d{i}", f"texte {i} test {i}") for i in range(5)]
        report = evaluate_diversity(docs, {})
        self.assertIn("summary", report)
        self.assertIn("details", report)
        self.assertIn("distinct_2", report["summary"])
        self.assertIn("self_bleu", report["summary"])
        self.assertIn("duplicate_rate", report["summary"])
        self.assertIn("cell_coverage", report["summary"])
        self.assertIn("passed", report["summary"])
        self.assertIn("embeddings", report["details"])


if __name__ == "__main__":
    unittest.main()
