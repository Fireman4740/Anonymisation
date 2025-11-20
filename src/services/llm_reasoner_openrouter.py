"""Compatibility wrapper mapping to the LLM reasoner implementation."""

from __future__ import annotations

from ..llm.reasoner import LLMReasoner, SeedSpan, DetectionPlan

__all__ = [
    "LLMReasoner",
    "SeedSpan",
    "DetectionPlan",
]
