"""Compatibility wrapper forwarding to the core RUPTA privacy evaluator."""

from __future__ import annotations

from ...rupta.privacy_evaluator import evaluate_reidentification_risk

__all__ = ["evaluate_reidentification_risk"]
