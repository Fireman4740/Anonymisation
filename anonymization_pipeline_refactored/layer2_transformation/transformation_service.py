"""
Service de Transformation Unifié

Ce service encapsule toute la logique de transformation d'entités :
- Remplacement par placeholders
- Généralisation policy-driven (dates, orgs, IPs)
- Paraphrase LLM (stylométrique)
- Optimisation RUPTA (privacy-utility)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re

try:
    from .generalization.generalizer import GeneralizationService, Generalization, escalate_policy
    from ..utils.pseudo_mapper import PseudoMapper
    from ..api.policy import AnonymizationPolicy
except Exception:
    # Imports directs pour tests
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from generalization.generalizer import GeneralizationService, Generalization, escalate_policy
    from utils.pseudo_mapper import PseudoMapper
    from api.policy import AnonymizationPolicy


@dataclass
class TransformationResult:
    """
    Résultat d'une transformation complète.
    
    Attributes:
        anonymized_text: Texte anonymisé final
        replacements: Liste des remplacements effectués
        generalizations: Liste des généralisations effectuées
        paraphrase_applied: Paraphrase appliquée ou non
        rupta_applied: RUPTA appliqué ou non
        metadata: Métadonnées additionnelles
    """
    anonymized_text: str
    replacements: List[Dict[str, Any]] = field(default_factory=list)
    generalizations: List[Generalization] = field(default_factory=list)
    paraphrase_applied: bool = False
    rupta_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransformationService:
    """
    Service de transformation unifié.
    
    Ce service coordonne toutes les transformations nécessaires pour
    anonymiser un texte selon une policy donnée.
    """
    
    def __init__(
        self,
        policy: AnonymizationPolicy,
        pseudo_mapper: PseudoMapper,
        llm_service: Optional[Any] = None,  # LLMPipelineService optionnel
    ):
        """
        Initialise le service de transformation.
        
        Args:
            policy: Policy d'anonymisation
            pseudo_mapper: Mapper pour générer les placeholders
            llm_service: Service LLM optionnel pour paraphrase et RUPTA
        """
        self.policy = policy
        self.pseudo_mapper = pseudo_mapper
        self.llm_service = llm_service
        self.generalization_service = GeneralizationService(policy)
    
    def apply_replacements(
        self,
        text: str,
        entities: List[Any],  # List[DetectedEntity]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Applique les remplacements pour les entités détectées.
        
        Args:
            text: Texte original
            entities: Liste d'entités détectées
        
        Returns:
            Tuple (texte transformé, liste des remplacements)
        """
        replacements = []
        
        # Trier les entités par position (de la fin vers le début pour préserver les offsets)
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        
        result_text = text
        for entity in sorted_entities:
            # Générer le placeholder selon la policy
            placeholder = self._generate_placeholder(entity)
            
            # Appliquer le remplacement
            result_text = (
                result_text[:entity.start] +
                placeholder +
                result_text[entity.end:]
            )
            
            replacements.append({
                "start": entity.start,
                "end": entity.end,
                "original": entity.surface,
                "replacement": placeholder,
                "etype": entity.etype,
                "source": entity.source,
            })
        
        # Inverser pour avoir l'ordre original
        replacements.reverse()
        
        return result_text, replacements
    
    def _generate_placeholder(self, entity: Any) -> str:
        """
        Génère un placeholder pour une entité selon la policy.
        
        Args:
            entity: Entité détectée
        
        Returns:
            Placeholder généré
        """
        etype = entity.etype.upper()
        surface = entity.surface
        
        if self.policy.placeholder_style == "generic":
            return "[REDACTED]"
        
        # Style typed: utiliser le pseudo_mapper pour générer un placeholder stable
        return self.pseudo_mapper.placeholder(etype, surface)
    
    def apply_generalization(
        self,
        text: str,
    ) -> Tuple[str, List[Generalization]]:
        """
        Applique les généralisations policy-driven.
        
        Args:
            text: Texte avec placeholders
        
        Returns:
            Tuple (texte généralisé, liste des généralisations)
        """
        return self.generalization_service.apply_all(text)
    
    def apply_paraphrase(
        self,
        text: str,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Applique la paraphrase LLM si activée.
        
        Args:
            text: Texte à paraphraser
            temperature: Température LLM (None = calculée depuis policy)
        
        Returns:
            Tuple (texte paraphrasé, erreur éventuelle)
        """
        if not self.llm_service:
            return text, None
        
        if not self.policy.llm_paraphrase or self.policy.paraphrase_intensity <= 0:
            return text, None
        
        if temperature is None:
            temperature = 0.2 + 0.1 * self.policy.paraphrase_intensity
        
        try:
            paraphrased, error = self.llm_service.paraphrase(text, temperature)
            return paraphrased if not error else text, error
        except Exception as e:
            return text, f"Paraphrase error: {e}"
    
    def apply_audit_and_hardening(
        self,
        text: str,
        original_text: str,
    ) -> Tuple[str, Dict[str, Any], int]:
        """
        Applique l'audit de risque et le hardening loop si nécessaire.
        
        Args:
            text: Texte anonymisé
            original_text: Texte original (pour contexte)
        
        Returns:
            Tuple (texte durci, rapport d'audit, nombre de tours)
        """
        if not self.llm_service or not self.policy.llm_audit:
            return text, {"risk_score": 0, "findings": [], "recommendations": []}, 0
        
        # Audit initial
        risk_report, error = self.llm_service.audit(text)
        if error:
            risk_report = {"risk_score": 0, "findings": [], "recommendations": [], "error": error}
        
        # Hardening loop
        rounds = 0
        max_rounds = int(self.policy.max_hardening_rounds or 0)
        
        while (
            isinstance(risk_report.get("risk_score"), int) and
            risk_report["risk_score"] > self.policy.risk_threshold and
            rounds < max_rounds
        ):
            rounds += 1
            
            # Escalader la policy (immutable)
            escalated_policy = escalate_policy(self.policy)
            
            # Recréer le service de généralisation avec la policy escaladée
            temp_gen_service = GeneralizationService(escalated_policy)
            
            # Appliquer généralisation org plus agressive
            text, org_gens = temp_gen_service.generalize_org_placeholders(text)
            
            # Reparaphraser si possible
            if self.llm_service and escalated_policy.llm_paraphrase:
                temp = 0.2 + 0.1 * escalated_policy.paraphrase_intensity
                text, _ = self.llm_service.paraphrase(text, temperature=temp)
            
            # Réauditer
            risk_report, _ = self.llm_service.audit(text)
        
        return text, risk_report, rounds
    
    def apply_rupta(
        self,
        original_text: str,
        anonymized_text: str,
        ground_truth_people: Optional[str] = None,
        ground_truth_label: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Applique l'optimisation RUPTA (Risk-Utility Privacy Trade-off Analysis).
        
        Args:
            original_text: Texte original
            anonymized_text: Texte anonymisé initial
            ground_truth_people: Personnes dans le texte (pour évaluation)
            ground_truth_label: Label attendu (pour évaluation)
        
        Returns:
            Tuple (texte optimisé, métriques RUPTA)
        """
        if not self.llm_service or not self.policy.rupta_enabled:
            return anonymized_text, {}
        
        if not ground_truth_people or not ground_truth_label:
            return anonymized_text, {"error": "RUPTA requires ground truth data"}
        
        try:
            rupta_result, error = self.llm_service.optimize_with_rupta(
                original_text=original_text,
                initial_anonymized_text=anonymized_text,
                ground_truth_people=ground_truth_people,
                ground_truth_label=ground_truth_label,
            )
            
            if error:
                return anonymized_text, {"error": error}
            
            return rupta_result.final_text, {
                "privacy": rupta_result.privacy_score,
                "utility": rupta_result.utility_score,
                "iterations": rupta_result.iterations,
                "converged": rupta_result.converged,
            }
        except Exception as e:
            return anonymized_text, {"error": f"RUPTA error: {e}"}
    
    def transform(
        self,
        text: str,
        entities: List[Any],
        original_text: Optional[str] = None,
        ground_truth_people: Optional[str] = None,
        ground_truth_label: Optional[str] = None,
    ) -> TransformationResult:
        """
        Applique toutes les transformations nécessaires.
        
        Args:
            text: Texte à transformer
            entities: Entités détectées
            original_text: Texte original (pour RUPTA)
            ground_truth_people: Ground truth pour RUPTA
            ground_truth_label: Ground truth label pour RUPTA
        
        Returns:
            TransformationResult complet
        """
        result = TransformationResult(anonymized_text=text)
        
        # 1. Remplacements
        text, replacements = self.apply_replacements(text, entities)
        result.anonymized_text = text
        result.replacements = replacements
        
        # 2. Généralisation
        text, generalizations = self.apply_generalization(text)
        result.anonymized_text = text
        result.generalizations = generalizations
        
        # 3. Paraphrase (si L1+)
        if self.policy.llm_paraphrase:
            text, para_error = self.apply_paraphrase(text)
            result.anonymized_text = text
            result.paraphrase_applied = para_error is None
            if para_error:
                result.metadata["paraphrase_error"] = para_error
        
        # 4. Audit & Hardening (si L1+)
        if self.policy.llm_audit:
            text, audit_report, hardening_rounds = self.apply_audit_and_hardening(
                text, original_text or result.anonymized_text
            )
            result.anonymized_text = text
            result.metadata["audit"] = audit_report
            result.metadata["hardening_rounds"] = hardening_rounds
        
        # 5. RUPTA (si L1+ et activé)
        if self.policy.rupta_enabled:
            text, rupta_metrics = self.apply_rupta(
                original_text or result.anonymized_text,
                text,
                ground_truth_people,
                ground_truth_label,
            )
            result.anonymized_text = text
            result.rupta_applied = "error" not in rupta_metrics
            result.metadata["rupta"] = rupta_metrics
        
        return result


__all__ = ["TransformationService", "TransformationResult", "Generalization"]
