# -*- coding: utf-8 -*-
"""Tests for LLM provider/model resolution.

These tests inject the config through ``_CONFIG_CACHE`` instead of reading the
real config.json: the file is user-editable (provider switches between
openrouter and ollama depending on the machine), so asserting its content made
the test fail whenever the local setup changed.
"""

from src.nodes.llm import llm_client
from src.nodes.llm.llm_client import LLMClient


def _inject_config(config):
    llm_client._CONFIG_CACHE = config


def test_ollama_provider_resolution():
    """With an ollama config, the client targets the local endpoint."""
    _inject_config(
        {
            "llm": {
                "provider": "ollama",
                "model": "gemma4:26b",
                "base_url": "http://localhost:11434/v1",
            }
        }
    )
    try:
        client = LLMClient(role="detect")

        assert client.provider == "ollama"
        assert client.base_url == "http://localhost:11434/v1"
        assert client.model == "gemma4:26b"
    finally:
        _inject_config(None)


def test_runtime_model_override_still_wins():
    """Ablation/runtime overrides remain the highest-priority model source."""
    _inject_config(
        {
            "llm": {"provider": "ollama", "model": "gemma4:26b"},
            "openrouter": {"model": "google/gemma-4-26b-a4b-it"},
        }
    )
    try:
        client = LLMClient.create(
            role="detect",
            state_config={
                "llm_provider": "openrouter",
                "llm_model": "openai/gpt-4o-mini",
            },
        )

        assert client.provider == "openrouter"
        assert client.model == "openai/gpt-4o-mini"
    finally:
        _inject_config(None)
