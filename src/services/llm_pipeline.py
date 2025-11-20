"""
LLM Pipeline Service - Encapsulates LLM reasoning and RUPTA optimization

This module provides a clean interface for:
- LLM-based entity detection and planning
- Text paraphrasing for stylometric reduction
- Risk auditing
- RUPTA privacy-utility optimization
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .openrouter_client import OpenRouterClient
from .llm_reasoner_openrouter import LLMReasoner, SeedSpan, DetectionPlan
from .rupta.optimizer import optimize_anonymization
from .rupta.privacy_evaluator import evaluate_reidentification_risk
from .rupta.utility_evaluator import evaluate_utility_preservation
from .policy import AnonymizationPolicy


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
    """
    Service for LLM-based operations (detection, paraphrase, audit, RUPTA).
    
    This service encapsulates all LLM interactions and provides a clean
    interface for the orchestrator.
    """
    
    def __init__(
        self,
        client: OpenRouterClient,
        policy: AnonymizationPolicy,
        models: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize LLM pipeline service.
        
        Args:
            client: OpenRouter client instance
            policy: Anonymization policy
            models: Dict of model names for different tasks
                    {"detect": ..., "paraphrase": ..., "audit": ...}
        """
        self.client = client
        self.policy = policy
        self.models = models or {}
        
        # Create LLM reasoner
        detect_model = self.models.get("detect", "openai/gpt-4.1-mini")
        paraphrase_model = self.models.get("paraphrase", detect_model)
        audit_model = self.models.get("audit", detect_model)
        
        self.reasoner = LLMReasoner(
            client=client,
            model_detect=detect_model,
            model_paraphrase=paraphrase_model,
            model_audit=audit_model,
        )
    
    def detect_and_plan(
        self,
        text: str,
        seeds: List[SeedSpan],
    ) -> LLMDetectionResult:
        """
        Use LLM to detect additional entities and create anonymization plan.
        
        Args:
            text: Input text
            seeds: Initial seed spans from regex/NER
            
        Returns:
            LLMDetectionResult with detected entities and plan
        """
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
        except Exception as e:
            # Return empty result on error
            return LLMDetectionResult(
                entities=[],
                generalizations=[],
                edits=[],
                relations=[],
                notes=[f"LLM detection error: {e}"],
            )
    
    def paraphrase(
        self,
        text: str,
        temperature: float = 0.3,
        ensure_placeholders_preserved: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """
        Paraphrase text for stylometric reduction.
        
        Args:
            text: Input text
            temperature: LLM temperature
            ensure_placeholders_preserved: Whether to verify placeholders remain
            
        Returns:
            Tuple of (paraphrased_text, error_message)
        """
        try:
            paraphrased = self.reasoner.paraphrase(
                text,
                temperature=temperature,
                ensure_placeholders_preserved=ensure_placeholders_preserved,
            )
            return paraphrased, None
        except Exception as e:
            return text, f"Paraphrase error: {e}"
    
    def audit(self, text: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Audit anonymized text for re-identification risk.
        
        Args:
            text: Anonymized text to audit
            
        Returns:
            Tuple of (audit_report, error_message)
        """
        try:
            report = self.reasoner.audit(text)
            return report, None
        except Exception as e:
            return {
                "risk_score": 100,
                "findings": [],
                "recommendations": [],
            }, f"Audit error: {e}"
    
    def optimize_with_rupta(
        self,
        original_text: str,
        initial_anonymized_text: str,
        ground_truth_people: str,
        ground_truth_label: str,
    ) -> Tuple[RuptaResult, Optional[str]]:
        """
        Optimize anonymization using RUPTA iterative refinement.
        
        Args:
            original_text: Original text before anonymization
            initial_anonymized_text: Text after initial anonymization
            ground_truth_people: Real person name (for privacy eval)
            ground_truth_label: Real classification label (for utility eval)
            
        Returns:
            Tuple of (RuptaResult, error_message)
        """
        try:
            result = optimize_anonymization(
                client=self.client,
                original_text=original_text,
                initial_anonymized_text=initial_anonymized_text,
                ground_truth_people=ground_truth_people,
                ground_truth_label=ground_truth_label,
                max_iterations=self.policy.rupta_max_iterations,
                p_threshold=self.policy.rupta_p_threshold,
                privacy_target_rank=self.policy.rupta_privacy_threshold or (self.policy.rupta_p_threshold + 1),
                utility_min_confidence=self.policy.rupta_utility_threshold,
                model=self.models.get("detect"),
            )
            
            return RuptaResult(
                final_text=result["final_text"],
                privacy_score=result["privacy_score"],
                utility_score=result["utility_score"],
                iterations=result["iterations"],
                converged=result["converged"],
                history=result["history"],
            ), None
        except Exception as e:
            # Return initial text on error
            return RuptaResult(
                final_text=initial_anonymized_text,
                privacy_score={},
                utility_score={},
                iterations=0,
                converged=False,
                history=[],
            ), f"RUPTA optimization error: {e}"
    
    def evaluate_privacy(
        self,
        anonymized_text: str,
        ground_truth_people: str,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Evaluate privacy (re-identification risk) of anonymized text.
        
        Args:
            anonymized_text: Text to evaluate
            ground_truth_people: Real person name
            
        Returns:
            Tuple of (privacy_eval, error_message)
        """
        try:
            eval_result = evaluate_reidentification_risk(
                client=self.client,
                anonymized_text=anonymized_text,
                ground_truth_people=ground_truth_people,
                p_threshold=self.policy.rupta_p_threshold,
                model=self.models.get("audit") or self.models.get("detect"),
            )
            return eval_result, None
        except Exception as e:
            return {}, f"Privacy evaluation error: {e}"
    
    def evaluate_utility(
        self,
        anonymized_text: str,
        ground_truth_label: str,
        original_text: str = "",
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Evaluate utility (classification preservation) of anonymized text.
        
        Args:
            anonymized_text: Text to evaluate
            ground_truth_label: Real classification label
            original_text: Original text for comparison
            
        Returns:
            Tuple of (utility_eval, error_message)
        """
        try:
            eval_result = evaluate_utility_preservation(
                client=self.client,
                anonymized_text=anonymized_text,
                ground_truth_label=ground_truth_label,
                model=self.models.get("paraphrase") or self.models.get("detect"),
            )
            return eval_result, None
        except Exception as e:
            return {}, f"Utility evaluation error: {e}"


def create_llm_pipeline(
    policy: AnonymizationPolicy,
    openrouter_models: Optional[Dict[str, str]] = None,
) -> Optional[LLMPipelineService]:
    """
    Factory function to create LLM pipeline service.
    
    Args:
        policy: Anonymization policy
        openrouter_models: Optional model overrides
        
    Returns:
        LLMPipelineService instance or None if LLM disabled
    """
    if not policy.llm_detection:
        return None
    
    try:
        client = OpenRouterClient.from_config()
        
        # Resolve models
        default_models = {
            "detect": "openai/gpt-4.1-mini",
            "paraphrase": "openai/gpt-4.1-mini",
            "audit": "openai/gpt-4.1-mini",
        }
        
        # Update with config models
        if hasattr(client, "config_models") and client.config_models:
            default_models.update(client.config_models)
        
        # Update with overrides
        if openrouter_models:
            default_models.update(openrouter_models)
        
        return LLMPipelineService(
            client=client,
            policy=policy,
            models=default_models,
        )
    except Exception as e:
        print(f"[LLMPipeline] Failed to create LLM pipeline: {e}")
        return None


__all__ = [
    "LLMDetectionResult",
    "RuptaResult",
    "LLMPipelineService",
    "create_llm_pipeline",
]
