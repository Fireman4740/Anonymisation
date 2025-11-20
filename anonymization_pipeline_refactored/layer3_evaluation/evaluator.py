
"""
Évaluateur - Métriques et Validation

Ce module fournit des métriques de base et de la validation pour
évaluer la qualité de l'anonymisation.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
import re


@dataclass
class EvaluationResult:
    """
    Résultat d'évaluation.
    
    Attributes:
        is_valid: Le texte anonymisé est valide
        metrics: Métriques calculées
        validation_errors: Erreurs de validation
        warnings: Avertissements
    """
    is_valid: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Evaluator:
    """
    Évaluateur pour les résultats d'anonymisation.
    
    Fournit des métriques de base et de la validation pour s'assurer
    que l'anonymisation est correcte.
    """
    
    def __init__(self):
        """Initialise l'évaluateur."""
        self.placeholder_pattern = re.compile(r'\[[A-Z_]+(?:_[A-Z0-9]+)?\]')
    
    def evaluate(
        self,
        original_text: str,
        anonymized_text: str,
        entities_detected: int,
        entities_replaced: int,
    ) -> EvaluationResult:
        """
        Évalue un résultat d'anonymisation.
        
        Args:
            original_text: Texte original
            anonymized_text: Texte anonymisé
            entities_detected: Nombre d'entités détectées
            entities_replaced: Nombre d'entités remplacées
        
        Returns:
            EvaluationResult avec métriques et validation
        """
        result = EvaluationResult()
        
        # Métriques de base
        result.metrics["original_length"] = len(original_text)
        result.metrics["anonymized_length"] = len(anonymized_text)
        result.metrics["length_ratio"] = len(anonymized_text) / len(original_text) if original_text else 0
        result.metrics["entities_detected"] = entities_detected
        result.metrics["entities_replaced"] = entities_replaced
        result.metrics["replacement_rate"] = entities_replaced / entities_detected if entities_detected > 0 else 0
        
        # Compter les placeholders
        placeholders = self.placeholder_pattern.findall(anonymized_text)
        result.metrics["placeholder_count"] = len(placeholders)
        result.metrics["unique_placeholders"] = len(set(placeholders))
        
        # Validation : vérifier qu'il n'y a pas de placeholders mal formés
        malformed = re.findall(r'\[[A-Z_]+[^\]]*(?!\])', anonymized_text)
        if malformed:
            result.validation_errors.append(f"Placeholders mal formés détectés: {malformed[:5]}")
            result.is_valid = False
        
        # Validation : vérifier que le texte n'est pas vide
        if not anonymized_text.strip():
            result.validation_errors.append("Le texte anonymisé est vide")
            result.is_valid = False
        
        # Warning : si beaucoup de placeholders
        if len(placeholders) > len(original_text.split()) * 0.5:
            result.warnings.append("Plus de 50% des mots ont été remplacés par des placeholders")
        
        # Warning : si le texte a trop changé de taille
        if result.metrics["length_ratio"] < 0.5 or result.metrics["length_ratio"] > 2.0:
            result.warnings.append(f"Changement de taille important (ratio: {result.metrics['length_ratio']:.2f})")
        
        return result
    
    def validate_placeholders(self, text: str) -> Tuple[bool, List[str]]:
        """
        Valide que tous les placeholders sont bien formés.
        
        Args:
            text: Texte à valider
        
        Returns:
            Tuple (is_valid, errors)
        """
        errors = []
        
        # Vérifier les placeholders mal fermés
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        if open_brackets != close_brackets:
            errors.append(f"Nombre de crochets non équilibré: {open_brackets} [ vs {close_brackets} ]")
        
        # Vérifier les placeholders mal formés
        malformed = re.findall(r'\[[^\]]{50,}\]', text)  # Placeholders trop longs
        if malformed:
            errors.append(f"Placeholders suspicieusement longs: {len(malformed)}")
        
        return len(errors) == 0, errors


__all__ = ["Evaluator", "EvaluationResult"]
