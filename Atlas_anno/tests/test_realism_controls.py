from __future__ import annotations

import unittest
from typing import List
from unittest.mock import MagicMock, patch

from atlas_anno.evaluation.register_conformance import apply_register_conformance, check_document
from atlas_anno.evaluation.realism_judge import apply_human_realism_sample, run_judge_realism_command
from atlas_anno.generation.style_sampler import load_style_factors
from atlas_anno.schemas import (
    AnnotationBundle,
    CandidatePools,
    DocumentRecord,
    ScenarioSpec,
)


def _doc(
    doc_id: str,
    text: str,
    address_form: str = "vous",
    register: str = "courant",
    domain: str = "email",
) -> DocumentRecord:
    scenario = ScenarioSpec(
        scenario_id=f"scenario_{doc_id}",
        domain=domain,
        unit_type="single_message",
        language="fr",
        author_id="p1",
        recipient_role="ops",
        document_goal="info",
        difficulty="easy",
        address_form=address_form,
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
        metadata={"address_form": address_form, "register": register},
    )


# ---------------------------------------------------------------------------
# Détecteur tu / vous
# ---------------------------------------------------------------------------

class TuVousDetectorTest(unittest.TestCase):
    def test_vous_doc_with_tutoiement_flagged(self) -> None:
        doc = _doc("d1", "Bonjour, tu peux m'aider avec ce ticket ?", address_form="vous")
        result = check_document(doc)
        self.assertFalse(result["passed"])
        self.assertTrue(any("tutoiement" in flag for flag in result["flags"]))

    def test_vous_doc_without_tutoiement_passes(self) -> None:
        doc = _doc("d2", "Bonjour, pouvez-vous m'aider avec ce problème ?", address_form="vous")
        result = check_document(doc)
        self.assertTrue(result["passed"])

    def test_tu_doc_with_vouvoiement_flagged(self) -> None:
        doc = _doc("d3", "Salut, veuillez confirmer la réservation.", address_form="tu")
        result = check_document(doc)
        self.assertFalse(result["passed"])
        self.assertTrue(any("vouvoiement" in flag for flag in result["flags"]))

    def test_tu_doc_without_vouvoiement_passes(self) -> None:
        doc = _doc("d4", "Salut, tu peux me confirmer ça rapidement ?", address_form="tu")
        result = check_document(doc)
        self.assertTrue(result["passed"])

    def test_vous_pluriel_ambigu_non_flagge_en_doc_tu(self) -> None:
        # « vous » seul = ambigu pluriel → ne doit PAS déclencher de flag vouvoiement
        doc = _doc("d5", "Nous vous informons que vous avez reçu un message.", address_form="tu")
        result = check_document(doc)
        self.assertTrue(result["passed"])

    def test_peux_tu_flagged_in_vous_doc(self) -> None:
        doc = _doc("d6", "Peux-tu m'envoyer ce document dès que possible ?", address_form="vous")
        result = check_document(doc)
        self.assertFalse(result["passed"])


# ---------------------------------------------------------------------------
# Tabous par registre
# ---------------------------------------------------------------------------

class RegisterTabooTest(unittest.TestCase):
    def test_soutenu_taboo_flagged(self) -> None:
        taboos = load_style_factors().get("register_taboos", {}).get("soutenu", [])
        if not taboos:
            self.skipTest("Pas de tabous soutenu configurés")
        taboo_word = taboos[0]
        text = f"Je vous informe que {taboo_word} le dossier est en cours de traitement."
        doc = _doc("d7", text, address_form="vous", register="soutenu")
        result = check_document(doc)
        self.assertFalse(result["passed"])
        self.assertTrue(any("register_taboo" in flag for flag in result["flags"]))

    def test_non_taboo_in_soutenu_passes(self) -> None:
        doc = _doc("d8", "Je vous informe que le dossier est en cours de traitement.", address_form="vous", register="soutenu")
        flags = [f for f in check_document(doc).get("flags", []) if "register_taboo" in f]
        self.assertEqual(flags, [])

    def test_familier_taboo_flagged(self) -> None:
        taboos = load_style_factors().get("register_taboos", {}).get("familier", [])
        if not taboos:
            self.skipTest("Pas de tabous familier configurés")
        taboo_word = taboos[0]
        text = f"Salut, {taboo_word} j'ai besoin d'aide pour mon ticket."
        doc = _doc("d9", text, address_form="tu", register="familier")
        result = check_document(doc)
        self.assertFalse(result["passed"])


# ---------------------------------------------------------------------------
# apply_register_conformance
# ---------------------------------------------------------------------------

class ApplyRegisterConformanceTest(unittest.TestCase):
    def test_applies_to_documents_with_register(self) -> None:
        docs = [
            _doc("d1", "tu peux m'aider ?", address_form="vous", register="courant"),
            _doc("d2", "pouvez-vous confirmer ?", address_form="vous", register="courant"),
        ]
        result = apply_register_conformance(docs)
        self.assertEqual(result["checked"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertTrue(docs[0].metadata.get("human_review_required"))
        self.assertIn("register_conformance", docs[0].metadata.get("review_reasons", []))

    def test_reasons_coexist_with_existing_flags(self) -> None:
        doc = _doc("d1", "tu peux m'aider ?", address_form="vous", register="courant")
        doc.metadata["review_reasons"] = ["previous_reason"]
        apply_register_conformance([doc])
        reasons = doc.metadata.get("review_reasons", [])
        self.assertIn("previous_reason", reasons)
        self.assertIn("register_conformance", reasons)

    def test_skips_doc_without_register(self) -> None:
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
            text="tu peux venir ?",
            scenario=scenario,
            candidate_pools=CandidatePools(),
            annotations=AnnotationBundle(),
            metadata={},
        )
        result = apply_register_conformance([doc])
        self.assertEqual(result["checked"], 0)


# ---------------------------------------------------------------------------
# Échantillonnage humain déterministe
# ---------------------------------------------------------------------------

class HumanRealismSampleTest(unittest.TestCase):
    def _make_docs(self, n: int) -> List[DocumentRecord]:
        return [_doc(f"doc_{i:03d}", f"texte du document numéro {i}") for i in range(n)]

    def test_sample_count_at_5pct(self) -> None:
        docs = self._make_docs(100)
        n = apply_human_realism_sample(docs, {"human_sample_rate": 0.05, "sample_seed": 53})
        self.assertEqual(n, 5)
        marked = [d for d in docs if d.metadata.get("human_review_required")]
        self.assertEqual(len(marked), 5)

    def test_stable_across_two_calls(self) -> None:
        docs1 = self._make_docs(100)
        docs2 = self._make_docs(100)
        apply_human_realism_sample(docs1, {"human_sample_rate": 0.05, "sample_seed": 53})
        apply_human_realism_sample(docs2, {"human_sample_rate": 0.05, "sample_seed": 53})
        marked1 = {d.doc_id for d in docs1 if d.metadata.get("human_review_required")}
        marked2 = {d.doc_id for d in docs2 if d.metadata.get("human_review_required")}
        self.assertEqual(marked1, marked2)

    def test_realism_sample_reason_appended(self) -> None:
        docs = self._make_docs(20)
        apply_human_realism_sample(docs, {"human_sample_rate": 0.25, "sample_seed": 53})
        for doc in docs:
            if doc.metadata.get("human_review_required"):
                self.assertIn("realism_sample", doc.metadata.get("review_reasons", []))

    def test_existing_review_reasons_preserved(self) -> None:
        docs = self._make_docs(10)
        for doc in docs:
            doc.metadata["review_reasons"] = ["previous_reason"]
        apply_human_realism_sample(docs, {"human_sample_rate": 0.50, "sample_seed": 53})
        for doc in docs:
            if doc.metadata.get("human_review_required"):
                reasons = doc.metadata.get("review_reasons", [])
                self.assertIn("previous_reason", reasons)
                self.assertIn("realism_sample", reasons)

    def test_zero_rate_marks_nothing(self) -> None:
        docs = self._make_docs(10)
        n = apply_human_realism_sample(docs, {"human_sample_rate": 0.0, "sample_seed": 53})
        self.assertEqual(n, 0)
        self.assertFalse(any(d.metadata.get("human_review_required") for d in docs))


# ---------------------------------------------------------------------------
# run_judge_realism_command — mode disabled
# ---------------------------------------------------------------------------

class JudgeRealismDisabledTest(unittest.TestCase):
    def test_disabled_saves_empty_report(self) -> None:
        with patch("atlas_anno.evaluation.realism_judge.save_report") as mock_save:
            run_judge_realism_command("disabled")
        mock_save.assert_called_once()
        _strategy, name, report = mock_save.call_args[0]
        self.assertEqual(name, "realism")
        self.assertEqual(report["summary"]["mode"], "disabled")
        self.assertEqual(report["summary"]["judged"], 0)
        self.assertEqual(report["judgments"], [])

    def test_primary_fallback_with_fake_client(self) -> None:
        fake_judgment = {
            "rationale": "Texte naturel et correctement rédigé.",
            "scores": {"naturalness": 4, "register": 4, "plausibility": 4},
            "overall": 4,
        }
        docs = [_doc("d1", "Bonjour, pouvez-vous m'aider avec ce problème ?", address_form="vous")]

        meta_mock = MagicMock()
        meta_mock.cache_hit = False
        meta_mock.llm_used = True
        meta_mock.error = False

        with (
            patch("atlas_anno.evaluation.realism_judge.load_documents", return_value=docs),
            patch("atlas_anno.evaluation.realism_judge.save_report") as mock_save,
            patch("atlas_anno.evaluation.realism_judge.OpenRouterClient") as MockClient,
        ):
            instance = MockClient.return_value
            instance.complete_json.return_value = (fake_judgment, meta_mock)
            run_judge_realism_command("primary-fallback")

        mock_save.assert_called_once()
        _strategy, name, report = mock_save.call_args[0]
        self.assertEqual(name, "realism")
        self.assertEqual(report["summary"]["mode"], "primary-fallback")
        self.assertEqual(report["summary"]["judged"], 1)
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertEqual(report["judgments"][0]["avg_overall"], 4.0)


# ---------------------------------------------------------------------------
# Export Label Studio filtre les docs échantillonnés
# ---------------------------------------------------------------------------

class LabelStudioRealismSampleTest(unittest.TestCase):
    def test_review_required_selection_includes_sampled(self) -> None:
        from atlas_anno.review.label_studio import _selected_documents

        docs = [_doc(f"d{i}", f"texte {i}") for i in range(20)]
        n = apply_human_realism_sample(docs, {"human_sample_rate": 0.25, "sample_seed": 53})
        self.assertGreater(n, 0)
        required_ids = {d.doc_id for d in docs if d.metadata.get("human_review_required")}

        with patch("atlas_anno.review.label_studio.load_documents", return_value=docs):
            selected = _selected_documents("review-required")

        selected_ids = {d.doc_id for d in selected}
        self.assertEqual(selected_ids, required_ids)


if __name__ == "__main__":
    unittest.main()
