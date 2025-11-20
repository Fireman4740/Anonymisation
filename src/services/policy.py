"""Compatibility shim exposing anonymization policy to services package."""

from __future__ import annotations

from ..core.policy import AnonymizationPolicy, preset

__all__ = ["AnonymizationPolicy", "preset"]
