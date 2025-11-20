"""LLM Pipeline Service - Encapsulates LLM reasoning and RUPTA optimization.

This module provides a clean interface for:
- LLM-based entity detection and planning
- Text paraphrasing for stylometric reduction
- Risk auditing
- RUPTA privacy-utility optimization
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...llm.openrouter_client import OpenRouterClient
from ...llm.reasoner import DetectionPlan, LLMReasoner, SeedSpan
from ...rupta.optimizer import optimize_anonymization
from ...rupta.privacy_evaluator import evaluate_reidentification_risk
from ...rupta.utility_evaluator import evaluate_utility_preservation
from ...core.policy import AnonymizationPolicy


@dataclass
class LLMDetectionResult:
    """Result from LLM-based detection."""

    entities: List[Dict[str, Any]]
    generalizations: List[Dict[str, Any]]
    edits: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    notes: List[str]


@dataclass
class RuptaResult:
    """Result from RUPTA optimization."""

    final_text: str
    privacy_score: Dict[str, Any]
    utility_score: Dict[str, Any]
    iterations: int
    converged: bool
    history: List[Dict[str, Any]]


class LLMPipelineService:
    """Service for LLM-based operations (detection, paraphrase, audit, RUPTA)."""

    def __init__(
        self,
        client: OpenRouterClient,
        policy: AnonymizationPolicy,
        models: Optional[Dict[str, str]] = None,
    ) -> None:
        self.client = client
        self.policy = policy
        self.models = models or {}

        detect_model = self.models.get("detect", "openai/gpt-4.1-mini")
        paraphrase_model = self.models.get("paraphrase", detect_model)
        audit_model = self.models.get("audit", detect_model)

        self.reasoner = LLMReasoner(
            client=client,
            model_detect=detect_model,
            model_paraphrase=paraphrase_model,
            model_audit=audit_model,
        )

    def detect_and_plan(self, text: str, seeds: List[SeedSpan]) -> LLMDetectionResult:
        """Use LLM to detect additional entities and create anonymization plan."""

        try:
            plan: DetectionPlan = self.reasoner.detect_and_plan(
                text,
                seeds,
                self.policy.to_dict(),
            )
            return LLMDetectionResult(
                entities=plan.entities,
                generalizations=plan.generalizations,
                edits=plan.edits,
                relations=plan.relations,
                notes=plan.notes,
            )
        except Exception as exc:  # pragma: no cover - LLM failure fallback
            return LLMDetectionResult(
                entities=[],
                generalizations=[],
                edits=[],
                relations=[],
                notes=[f"LLM detection error: {exc}"],
            )

    def paraphrase(
        self,
        text: str,
        temperature: float = 0.3,
        ensure_placeholders_preserved: bool = True,
        preserve_multiplicity: bool = False,
        expected_counts: Optional[Dict[str, int]] = None,
        intensity: int = 1,
    ) -> Tuple[str, Optional[str]]:
        """Paraphrase text for stylometric reduction."""

        try:
            paraphrased = self.reasoner.paraphrase(
                text,
                temperature=temperature,
                ensure_placeholders_preserved=ensure_placeholders_preserved,
                preserve_multiplicity=preserve_multiplicity,
                expected_counts=expected_counts,
                intensity=intensity,
            )
            return paraphrased, None
        except Exception as exc:  # pragma: no cover - LLM failure fallback
            return text, f"Paraphrase error: {exc}"

    def audit(self, text: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Audit anonymized text for re-identification risk."""

        try:
            report = self.reasoner.audit(text)
            return report, None
        except Exception as exc:  # pragma: no cover - LLM failure fallback
            return {
                "risk_score": 100,
                "findings": [],
                "recommendations": [],
            }, f"Audit error: {exc}"

    def optimize_with_rupta(
        self,
        original_text: str,
        initial_anonymized_text: str,
        ground_truth_people: Optional[str],
        ground_truth_label: Optional[str],
    ) -> Tuple[RuptaResult, Optional[str]]:
        """Optimize anonymization using RUPTA iterative refinement."""

        try:
            result = optimize_anonymization(
                client=self.client,
                original_text=original_text,
                initial_anonymized_text=initial_anonymized_text,
                ground_truth_people=ground_truth_people,
                ground_truth_label=ground_truth_label,
                max_iterations=self.policy.rupta_max_iterations,
                p_threshold=self.policy.rupta_p_threshold,
                privacy_target_rank=self.policy.rupta_privacy_threshold
                or (self.policy.rupta_p_threshold + 1),
                utility_min_confidence=self.policy.rupta_utility_threshold,
                model=self.models.get("detect"),
            )

            return (
                RuptaResult(
                    final_text=result["final_text"],
                    privacy_score=result["privacy_score"],
                    utility_score=result["utility_score"],
                    iterations=result["iterations"],
                    converged=result["converged"],
                    history=result["history"],
                ),
                None,
            )
        except Exception as exc:  # pragma: no cover - RUPTA failure fallback
            return (
                RuptaResult(
                    final_text=initial_anonymized_text,
                    privacy_score={},
                    utility_score={},
                    iterations=0,
                    converged=False,
                    history=[],
                ),
                f"RUPTA optimization error: {exc}",
            )

    def evaluate_privacy(
        self,
        anonymized_text: str,
        ground_truth_people: str,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Evaluate privacy (re-identification risk) of anonymized text."""

        try:
            evaluation = evaluate_reidentification_risk(
                client=self.client,
                anonymized_text=anonymized_text,
                ground_truth_people=ground_truth_people,
                p_threshold=self.policy.rupta_p_threshold,
                model=self.models.get("audit") or self.models.get("detect"),
            )
            return evaluation, None
        except Exception as exc:  # pragma: no cover - evaluation fallback
            return {}, f"Privacy evaluation error: {exc}"

    def evaluate_utility(
        self,
        anonymized_text: str,
        ground_truth_label: str,
        original_text: str = "",
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Evaluate utility (classification preservation) of anonymized text."""

        try:
            evaluation = evaluate_utility_preservation(
                client=self.client,
                anonymized_text=anonymized_text,
                ground_truth_label=ground_truth_label,
                model=self.models.get("paraphrase") or self.models.get("detect"),
            )
            return evaluation, None
        except Exception as exc:  # pragma: no cover - evaluation fallback
            return {}, f"Utility evaluation error: {exc}"


def create_llm_pipeline(
    policy: AnonymizationPolicy,
    openrouter_models: Optional[Dict[str, str]] = None,
) -> Optional[LLMPipelineService]:
    """Factory helper to create ``LLMPipelineService``."""

    if not (
        policy.llm_detection
        or policy.llm_paraphrase
        or policy.llm_audit
        or policy.rupta_enabled
    ):
        return None

    try:
        client = OpenRouterClient.from_config()
    except Exception as exc:  # pragma: no cover - configuration failure
        print(f"[LLMPipeline] Failed to create OpenRouter client: {exc}")
        return None

    models: Dict[str, str] = {
        "detect": "openai/gpt-4.1-mini",
        "paraphrase": "openai/gpt-4.1-mini",
        "audit": "openai/gpt-4.1-mini",
    }

    if getattr(client, "config_models", None):
        models.update(client.config_models)

    if openrouter_models:
        models.update({k: v for k, v in openrouter_models.items() if isinstance(v, str) and v})

    return LLMPipelineService(
        client=client,
        policy=policy,
        models=models,
    )


__all__ = [
    "LLMDetectionResult",
    "RuptaResult",
    "LLMPipelineService",
    "create_llm_pipeline",
]
