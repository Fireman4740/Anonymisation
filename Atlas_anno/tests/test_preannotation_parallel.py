from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from atlas_anno.annotation.preannotator import _annotate_document
from atlas_anno.annotation.preannotator import run_preannotation_command
from atlas_anno.generation.pipeline import run_generate_dataset_command
from atlas_anno.io import serialize
from atlas_anno.schemas import LLMRunMeta
from atlas_anno.settings import AtlasSettings, load_settings
from atlas_anno.llm import OpenRouterClient
from atlas_anno.storage import load_documents


class ParallelPreannotationTest(unittest.TestCase):
    def setUp(self) -> None:
        run_generate_dataset_command(100, "disabled", resume_enabled=False, cache_enabled=False)

    def _client(self) -> OpenRouterClient:
        return OpenRouterClient(
            AtlasSettings(
                openrouter_api_key="test-key",
                openrouter_base_url="https://example.test",
                atlas_model_reasoning="reasoning-model",
                atlas_model_creative="creative-model",
                http_timeout_seconds=5,
            ),
            runtime_overrides={"reasoning_workers": 4, "cache_enabled": False, "resume_enabled": False, "checkpoint_every": 1},
        )

    def test_preannotation_parallel_preserves_output(self) -> None:
        active = {"current": 0, "peak": 0}
        lock = threading.Lock()

        def fake_complete_json(self, *, step_name, prompt_spec, user_prompt, model, validator, fallback_value, temperature):
            with lock:
                active["current"] += 1
                active["peak"] = max(active["peak"], active["current"])
            try:
                time.sleep(0.01)
                payload = validator(
                    {
                        "spans": [{"snippet": "ACC-", "label": "ACCOUNT_ID", "confidence": 0.88}],
                        "relations": [],
                        "doc_labels": {"llm_confirmed": True},
                        "human_review_required": False,
                    }
                )
                return payload, LLMRunMeta(
                    step_name=step_name,
                    model=model,
                    prompt_version=prompt_spec.version,
                    llm_used=True,
                    fallback_used=False,
                    retry_count=0,
                    attempt_count=1,
                    queue_wait_ms=0,
                    cache_hit=False,
                    validation_errors=[],
                    latency_ms=1,
                    estimated_cost=0.0,
                )
            finally:
                with lock:
                    active["current"] -= 1

        with patch("atlas_anno.llm.OpenRouterClient.complete_json", new=fake_complete_json), patch(
            "atlas_anno.runtime.load_stage_checkpoints",
            return_value=[],
        ), patch("atlas_anno.runtime.append_stage_checkpoint", return_value=None):
            run_preannotation_command("hybrid-llm", reasoning_workers=1, resume_enabled=False, cache_enabled=False)
            sequential_documents = [serialize(document) for document in load_documents(annotated=True)]
            run_preannotation_command("hybrid-llm", reasoning_workers=4, resume_enabled=False, cache_enabled=False)

        parallel_documents = [serialize(document) for document in load_documents(annotated=True)]
        self.assertEqual(sequential_documents, parallel_documents)
        self.assertGreater(active["peak"], 1)

    def test_preannotation_resume_skips_checkpointed_documents(self) -> None:
        client = OpenRouterClient(load_settings(), runtime_overrides={"reasoning_workers": 4, "cache_enabled": False, "resume_enabled": True, "checkpoint_every": 1})
        documents = load_documents(annotated=False)
        checkpoint_rows = []

        def fake_complete_json(self, *, step_name, prompt_spec, user_prompt, model, validator, fallback_value, temperature):
            payload = validator(
                {
                    "spans": [],
                    "relations": [],
                    "doc_labels": {"llm_confirmed": True},
                    "human_review_required": False,
                }
            )
            return payload, LLMRunMeta(
                step_name=step_name,
                model=model,
                prompt_version=prompt_spec.version,
                llm_used=True,
                fallback_used=False,
                retry_count=0,
                attempt_count=1,
                queue_wait_ms=0,
                cache_hit=False,
                validation_errors=[],
                latency_ms=1,
                estimated_cost=0.0,
            )

        with patch("atlas_anno.llm.OpenRouterClient.complete_json", new=fake_complete_json):
            for document in documents[:2]:
                annotated_document, meta = _annotate_document(document, "hybrid-llm", client)
                checkpoint_rows.append(
                    {
                        "item_id": document.doc_id,
                        "step_name": "preannotation",
                        "result": serialize(annotated_document),
                        "llm_run": serialize(meta),
                        "completed_at": "2026-03-10T00:00:00+00:00",
                    }
                )

        call_count = 0

        def fake_counting_complete_json(self, *, step_name, prompt_spec, user_prompt, model, validator, fallback_value, temperature):
            nonlocal call_count
            call_count += 1
            payload = validator(
                {
                    "spans": [],
                    "relations": [],
                    "doc_labels": {"llm_confirmed": True},
                    "human_review_required": False,
                }
            )
            return payload, LLMRunMeta(
                step_name=step_name,
                model=model,
                prompt_version=prompt_spec.version,
                llm_used=True,
                fallback_used=False,
                retry_count=0,
                attempt_count=1,
                queue_wait_ms=0,
                cache_hit=False,
                validation_errors=[],
                latency_ms=1,
                estimated_cost=0.0,
            )

        with patch("atlas_anno.llm.OpenRouterClient.complete_json", new=fake_counting_complete_json), patch(
            "atlas_anno.runtime.load_stage_checkpoints",
            return_value=checkpoint_rows,
        ), patch("atlas_anno.runtime.append_stage_checkpoint", return_value=None):
            run_preannotation_command("hybrid-llm", reasoning_workers=4, resume_enabled=True, cache_enabled=False)

        self.assertEqual(call_count, len(documents) - 2)
