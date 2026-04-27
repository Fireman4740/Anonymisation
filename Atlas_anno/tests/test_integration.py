from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from atlas_anno.anonymization.baselines import run_anonymization_command
from atlas_anno.annotation.preannotator import run_preannotation_command
from atlas_anno.attacks.structured import run_structured_attack_command
from atlas_anno.evaluation.aggregate import (
    run_eval_privacy_command,
    run_eval_reid_command,
    run_eval_spans_command,
    run_eval_utility_command,
)
from atlas_anno.io import serialize
from atlas_anno.generation.pipeline import run_generate_dataset_command
from atlas_anno.reporting.builder import run_build_report_command
from atlas_anno.storage import load_documents, load_report, report_html_path, report_markdown_path


class IntegrationTest(unittest.TestCase):
    def test_end_to_end_mini_run(self) -> None:
        run_generate_dataset_command(100, "disabled", resume_enabled=False, cache_enabled=False)
        run_preannotation_command("hybrid-llm")
        run_anonymization_command("masking")
        run_structured_attack_command("masking")
        run_eval_spans_command("masking")
        run_eval_privacy_command("masking")
        run_eval_reid_command("masking")
        run_eval_utility_command("masking")
        run_build_report_command("masking")

        documents = load_documents(annotated=True)
        privacy = load_report("masking", "privacy")
        utility = load_report("masking", "utility")
        self.assertEqual(len(documents), 100)
        self.assertIn("privacy_score", privacy["summary"])
        self.assertIn("utility_score", utility["summary"])
        self.assertTrue(report_markdown_path("masking").exists())
        self.assertTrue(report_html_path("masking").exists())

    def test_generate_dataset_with_mocked_llm(self) -> None:
        from atlas_anno.schemas import LLMRunMeta

        def fake_complete_json(self, *, step_name, prompt_spec, user_prompt, model, validator, fallback_value, temperature):
            validated = validator(fallback_value)
            return validated, LLMRunMeta(
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
            run_generate_dataset_command(
                100,
                "primary-fallback",
                reasoning_workers=1,
                creative_workers=1,
                resume_enabled=False,
                cache_enabled=False,
            )
            sequential_documents = [serialize(document) for document in load_documents(annotated=False)]
            run_generate_dataset_command(
                100,
                "primary-fallback",
                reasoning_workers=4,
                creative_workers=4,
                resume_enabled=False,
                cache_enabled=False,
            )

        documents = load_documents(annotated=False)
        parallel_documents = [serialize(document) for document in documents]
        self.assertEqual(len(documents), 100)
        self.assertEqual(sequential_documents, parallel_documents)
        self.assertTrue(all("llm_audit" in document.metadata for document in documents))
        self.assertTrue(all(document.metadata.get("surface_grounding") for document in documents))

    @unittest.skipUnless(os.environ.get("OPENROUTER_API_KEY"), "requires OPENROUTER_API_KEY")
    def test_real_llm_smoke_sub_batch(self) -> None:
        run_generate_dataset_command(5, "primary-fallback", resume_enabled=False, cache_enabled=False)
        documents = load_documents(annotated=False)
        self.assertEqual(len(documents), 5)
