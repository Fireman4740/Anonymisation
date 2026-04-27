from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from atlas_anno.settings import load_settings, parse_dotenv


class SettingsTest(unittest.TestCase):
    def test_parse_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OPENROUTER_API_KEY=test-key\nATLAS_HTTP_TIMEOUT_SECONDS=12\n", encoding="utf-8")
            values = parse_dotenv(env_path)
            self.assertEqual(values["OPENROUTER_API_KEY"], "test-key")
            self.assertEqual(values["ATLAS_HTTP_TIMEOUT_SECONDS"], "12")

    def test_load_settings_from_custom_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY=test-key\n"
                "OPENROUTER_BASE_URL=https://example.test\n"
                "ATLAS_MODEL_REASONING=reasoning-model\n"
                "ATLAS_MODEL_CREATIVE=creative-model\n"
                "ATLAS_HTTP_TIMEOUT_SECONDS=33\n",
                encoding="utf-8",
            )
            settings = load_settings(str(env_path))
            self.assertEqual(settings.openrouter_api_key, "test-key")
            self.assertEqual(settings.openrouter_base_url, "https://example.test")
            self.assertEqual(settings.atlas_model_reasoning, "reasoning-model")
            self.assertEqual(settings.atlas_model_creative, "creative-model")
            self.assertEqual(settings.http_timeout_seconds, 33)

    def test_load_settings_uses_versioned_model_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY=test-key\n"
                "OPENROUTER_BASE_URL=https://example.test\n"
                "ATLAS_HTTP_TIMEOUT_SECONDS=21\n",
                encoding="utf-8",
            )
            settings = load_settings(str(env_path))
            self.assertEqual(settings.openrouter_api_key, "test-key")
            self.assertEqual(settings.openrouter_base_url, "https://example.test")
            self.assertEqual(settings.atlas_model_reasoning, "aion-labs/aion-2.0")
            self.assertEqual(settings.atlas_model_creative, "mistralai/mistral-small-creative")
            self.assertEqual(settings.http_timeout_seconds, 21)
