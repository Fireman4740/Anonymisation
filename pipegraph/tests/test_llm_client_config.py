# -*- coding: utf-8 -*-
"""Tests for LLM provider/model resolution."""

from src.nodes.llm import llm_client
from src.nodes.llm.llm_client import LLMClient


def test_default_provider_uses_ollama_config():
    """The default LLM backend is local Ollama."""
    llm_client._CONFIG_CACHE = None

    client = LLMClient(role="detect")

    assert client.provider == "ollama"
    assert client.base_url == "http://localhost:11434/v1"
    assert client.model == "gemma4:26b"


def test_runtime_model_override_still_wins():
    """Ablation/runtime overrides remain the highest-priority model source."""
    llm_client._CONFIG_CACHE = None

    client = LLMClient.create(
        role="detect",
        state_config={
            "llm_provider": "openrouter",
            "llm_model": "openai/gpt-4o-mini",
        },
    )

    assert client.provider == "openrouter"
    assert client.model == "openai/gpt-4o-mini"
