"""Compatibility shim mapping legacy import path to the GLiNER ensemble module."""

from __future__ import annotations

from .ner.ensemble import (
    run_gliner,
    merge_ner_lists,
    GLINER_ALL_LABELS,
)

__all__ = [
    "run_gliner",
    "merge_ner_lists",
    "GLINER_ALL_LABELS",
]
