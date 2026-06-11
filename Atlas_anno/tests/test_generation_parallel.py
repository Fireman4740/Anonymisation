from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from atlas_anno.generation.character_builder import build_characters
from atlas_anno.generation.llm_generation import refine_characters
from atlas_anno.generation.world_builder import build_worlds
from atlas_anno.io import serialize
from atlas_anno.prompts import load_prompt_spec
from atlas_anno.schemas import LLMRunMeta
from atlas_anno.settings import AtlasSettings
from atlas_anno.llm import OpenRouterClient


class ParallelGenerationTest(unittest.TestCase):
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

    def _characters(self, count: int = 6):
        worlds = build_worlds(count=1)
        return build_characters(worlds, per_world=count)[:count]

    def test_refine_characters_preserves_order_under_parallel_completion(self) -> None:
        characters = self._characters(6)
        client = self._client()
        delays = {character.full_name: float(index % 3) * 0.02 for index, character in enumerate(reversed(characters))}
        active = {"current": 0, "peak": 0}
        lock = threading.Lock()

        def fake_complete_json(self, *, step_name, prompt_spec, user_prompt, model, validator, fallback_value, temperature, allow_fallback=True):
            with lock:
                active["current"] += 1
                active["peak"] = max(active["peak"], active["current"])
            try:
                time.sleep(delays[fallback_value["full_name"]])
                validated = validator(fallback_value)
                return validated, LLMRunMeta(
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
            refined, _, stage_stats = refine_characters(
                characters,
                client,
                "primary-fallback",
                {"batch_name": "unit_parallel_order", "reasoning_workers": 4, "resume_enabled": False, "checkpoint_every": 1},
            )

        self.assertEqual([character.person_id for character in refined], [character.person_id for character in characters])
        self.assertGreater(active["peak"], 1)
        self.assertGreater(stage_stats["peak_concurrency"], 1)

    def test_refine_characters_resume_skips_checkpointed_items(self) -> None:
        characters = self._characters(4)
        client = self._client()
        prompt_spec = load_prompt_spec("character_builder")
        checkpoint_rows = []
        for character in characters[:2]:
            checkpoint_rows.append(
                {
                    "item_id": character.person_id,
                    "step_name": "character_generation",
                    "result": serialize(character),
                    "llm_run": serialize(
                        LLMRunMeta(
                            step_name="character_generation",
                            model="reasoning-model",
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
                    ),
                    "completed_at": "2026-03-10T00:00:00+00:00",
                }
            )

        call_count = 0

        def fake_complete_json(self, *, step_name, prompt_spec, user_prompt, model, validator, fallback_value, temperature, allow_fallback=True):
            nonlocal call_count
            call_count += 1
            validated = validator(fallback_value)
            return validated, LLMRunMeta(
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

        with patch("atlas_anno.llm.OpenRouterClient.complete_json", new=fake_complete_json), patch(
            "atlas_anno.runtime.load_stage_checkpoints",
            return_value=checkpoint_rows,
        ), patch("atlas_anno.runtime.append_stage_checkpoint", return_value=None):
            refined, _, stage_stats = refine_characters(
                characters,
                client,
                "primary-fallback",
                {"batch_name": "unit_parallel_resume", "reasoning_workers": 4, "resume_enabled": True, "checkpoint_every": 1},
            )

        self.assertEqual(len(refined), len(characters))
        self.assertEqual(call_count, len(characters) - 2)
        self.assertEqual(stage_stats["resumed_items"], 2)
        self.assertEqual(stage_stats["processed_items"], len(characters) - 2)
