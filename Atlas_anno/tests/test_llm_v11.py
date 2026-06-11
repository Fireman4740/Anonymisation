from __future__ import annotations

import unittest
from unittest.mock import patch

from atlas_anno.io import serialize
from atlas_anno.llm import OpenRouterClient, RetryableLLMError
from atlas_anno.prompts import load_prompt_spec
from atlas_anno.records import world_draft_from_dict
from atlas_anno.schemas import LLMRunMeta
from atlas_anno.settings import AtlasSettings


class LLMFallbackTest(unittest.TestCase):
    def _client(self, **runtime_overrides) -> OpenRouterClient:
        return OpenRouterClient(
            AtlasSettings(
                openrouter_api_key="test-key",
                openrouter_base_url="https://example.test",
                atlas_model_reasoning="reasoning-model",
                atlas_model_creative="creative-model",
                http_timeout_seconds=5,
            ),
            runtime_overrides=runtime_overrides,
        )

    def test_complete_json_falls_back_on_invalid_response(self) -> None:
        client = self._client(cache_enabled=False)

        with patch.object(client, "_perform_http_request", return_value={"choices": [{"message": {"content": "not json"}}]}):
            payload, meta = client.complete_json(
                step_name="unit-test",
                prompt_spec=load_prompt_spec("preannotation"),
                user_prompt="return json",
                model="reasoning-model",
                validator=lambda payload: payload,
                fallback_value={"ok": True},
                temperature=0.0,
            )
        self.assertEqual(payload, {"ok": True})
        self.assertTrue(meta.fallback_used)
        self.assertFalse(meta.llm_used)

    def test_complete_json_falls_back_on_missing_content(self) -> None:
        client = self._client(cache_enabled=False)

        with patch.object(
            client,
            "_perform_http_request",
            return_value={"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        ):
            payload, meta = client.complete_json(
                step_name="unit-test-missing-content",
                prompt_spec=load_prompt_spec("preannotation"),
                user_prompt="return json",
                model="reasoning-model",
                validator=lambda payload: payload,
                fallback_value={"ok": True},
                temperature=0.0,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertTrue(meta.fallback_used)
        self.assertFalse(meta.llm_used)
        self.assertTrue(any("missing text content" in error for error in meta.validation_errors))

    def test_complete_json_accepts_dataclass_validator_output(self) -> None:
        client = self._client(cache_enabled=False)

        with patch.object(
            client,
            "_perform_http_request",
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": '{"organization_name":"Atlas Services","departments":["AI Solutions"],"teams":["LLM Ops"],"projects":["connecteur_x"],"products":["AtlasDesk"],"incidents":["incident_1178"],"calendar_events":["audit_iso_q2"]}'
                        }
                    }
                ]
            },
        ):
            payload, meta = client.complete_json(
                step_name="unit-test-dataclass",
                prompt_spec=load_prompt_spec("world_builder"),
                user_prompt="return world draft json",
                model="reasoning-model",
                validator=world_draft_from_dict,
                fallback_value={
                    "organization_name": "Fallback Org",
                    "departments": ["AI Solutions"],
                    "teams": ["LLM Ops"],
                    "projects": ["connecteur_x"],
                    "products": ["AtlasDesk"],
                    "incidents": ["incident_1178"],
                    "calendar_events": ["audit_iso_q2"],
                },
                temperature=0.0,
            )
        self.assertEqual(payload.organization_name, "Atlas Services")
        self.assertTrue(meta.llm_used)

    def test_complete_json_uses_cache_without_http_request(self) -> None:
        client = self._client(cache_enabled=True)
        cached_meta = LLMRunMeta(
            step_name="cached-step",
            model="reasoning-model",
            prompt_version=load_prompt_spec("preannotation").version,
            llm_used=True,
            fallback_used=False,
            retry_count=0,
            attempt_count=1,
            queue_wait_ms=4,
            cache_hit=False,
            validation_errors=[],
            latency_ms=12,
            estimated_cost=0.1,
        )

        with patch(
            "atlas_anno.llm.load_llm_cache_entry",
            return_value={"result": {"ok": True}, "llm_run": serialize(cached_meta)},
        ), patch.object(client, "_perform_http_request", side_effect=AssertionError("http should not be called")):
            payload, meta = client.complete_json(
                step_name="cached-step",
                prompt_spec=load_prompt_spec("preannotation"),
                user_prompt="return json",
                model="reasoning-model",
                validator=lambda payload: payload,
                fallback_value={"ok": False},
                temperature=0.0,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertTrue(meta.cache_hit)
        self.assertEqual(meta.attempt_count, 0)

    def test_complete_json_retries_retryable_http_request(self) -> None:
        client = self._client(cache_enabled=False, backoff_initial_seconds=0, backoff_max_seconds=0)
        responses = [
            RetryableLLMError("temporary failure"),
            {"choices": [{"message": {"content": '{"ok": true}'}}]},
        ]

        def fake_request(model, messages, temperature=0.0, use_json_format=False):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        with patch.object(client, "_perform_http_request", side_effect=fake_request):
            payload, meta = client.complete_json(
                step_name="retry-step",
                prompt_spec=load_prompt_spec("preannotation"),
                user_prompt="return json",
                model="reasoning-model",
                validator=lambda payload: payload,
                fallback_value={"ok": False},
                temperature=0.0,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertTrue(meta.llm_used)
        self.assertEqual(meta.attempt_count, 2)
        self.assertEqual(meta.retry_count, 1)
