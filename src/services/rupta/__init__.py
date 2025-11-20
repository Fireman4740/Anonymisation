"""Compatibility package for RUPTA services."""

from __future__ import annotations

from ...rupta.optimizer import optimize_anonymization
from ...rupta.privacy_evaluator import evaluate_reidentification_risk
from ...rupta.utility_evaluator import evaluate_utility_preservation

__all__ = [
    "optimize_anonymization",
    "evaluate_reidentification_risk",
    "evaluate_utility_preservation",
]
