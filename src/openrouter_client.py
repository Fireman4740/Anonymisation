"""Compatibility wrapper for legacy imports."""

from __future__ import annotations

from .llm.openrouter_client import (
    OpenRouterClient,
    load_llm_settings,
    JSONParseError,
)

__all__ = [
    "OpenRouterClient",
    "load_llm_settings",
    "JSONParseError",
]
