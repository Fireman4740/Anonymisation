"""
API Publique du Pipeline d'Anonymisation

Interface simple et claire pour utiliser le pipeline d'anonymisation.
"""

from typing import Dict, Any, Optional, List

try:
    from ..orchestrator import anonymize_text_refactored
    from .policy import AnonymizationPolicy, preset
except Exception:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from orchestrator import anonymize_text_refactored
    from api.policy import AnonymizationPolicy, preset


class AnonymizationPipeline:
    """
    Pipeline d'anonymisation avec interface orientée objet.
    
    Cette classe fournit une interface pratique pour anonymiser du texte
    avec une configuration réutilisable.
    
    Exemple:
        >>> pipeline = AnonymizationPipeline(level="L1", secret_salt="my_secret")
        >>> result = pipeline.anonymize("Jean Dupont habite à Paris")
        >>> print(result["anonymized_text"])
        [PER_ABC] habite à [LOC_XYZ]
    """
    
    def __init__(
        self,
        level: str = "L0",
        secret_salt: str = "default_secret",
        policy: Optional[AnonymizationPolicy] = None,
        **overrides
    ):
        """
        Initialise le pipeline.
        
        Args:
            level: Niveau d'anonymisation ("L0", "L1", "L2")
            secret_salt: Secret HMAC pour pseudonymisation
            policy: Policy personnalisée (optionnel)
            **overrides: Overrides de policy additionnels
        """
        self.level = level
        self.secret_salt = secret_salt
        self.policy = policy or preset(level)
        self.overrides = overrides
        
        # Appliquer les overrides à la policy
        for k, v in overrides.items():
            if hasattr(self.policy, k):
                try:
                    setattr(self.policy, k, v)
                except Exception:
                    pass
    
    def anonymize(
        self,
        text: str,
        scope_id: str = "default",
        ner_results: Optional[List[dict]] = None,
        **additional_overrides
    ) -> Dict[str, Any]:
        """
        Anonymise un texte.
        
        Args:
            text: Texte à anonymiser
            scope_id: ID de scope pour pseudonymisation
            ner_results: Résultats NER pré-calculés
            **additional_overrides: Overrides additionnels pour cet appel
        
        Returns:
            Dictionnaire avec:
                - anonymized_text: Texte anonymisé
                - audit: Informations de détection et transformation
                - evaluation: Métriques et validation
                - policy: Policy utilisée
        """
        # Fusionner les overrides
        all_overrides = {**self.overrides, **additional_overrides}
        
        return anonymize_text_refactored(
            value=text,
            scope_id=scope_id,
            secret_salt=self.secret_salt,
            level=self.level,
            ner_results=ner_results,
            overrides=all_overrides if all_overrides else None,
        )
    
    def anonymize_batch(
        self,
        texts: List[str],
        scope_id: str = "default",
        **overrides
    ) -> List[Dict[str, Any]]:
        """
        Anonymise un lot de textes.
        
        Args:
            texts: Liste de textes à anonymiser
            scope_id: ID de scope pour pseudonymisation
            **overrides: Overrides additionnels
        
        Returns:
            Liste de résultats d'anonymisation
        """
        return [
            self.anonymize(text, scope_id=scope_id, **overrides)
            for text in texts
        ]


def anonymize_text(
    text: str,
    level: str = "L0",
    scope_id: str = "default",
    secret_salt: str = "default_secret",
    ner_results: Optional[List[dict]] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Fonction pratique pour anonymiser un texte (API fonctionnelle).
    
    Args:
        text: Texte à anonymiser
        level: Niveau d'anonymisation ("L0", "L1", "L2")
        scope_id: ID de scope pour pseudonymisation
        secret_salt: Secret HMAC pour pseudonymisation
        ner_results: Résultats NER pré-calculés
        **overrides: Overrides de policy
    
    Returns:
        Dictionnaire avec:
            - anonymized_text: Texte anonymisé
            - audit: Informations de détection et transformation
            - evaluation: Métriques et validation
            - policy: Policy utilisée
    
    Exemple:
        >>> result = anonymize_text(
        ...     "Jean Dupont habite à Paris",
        ...     level="L0",
        ...     secret_salt="my_secret"
        ... )
        >>> print(result["anonymized_text"])
        [PER_ABC] habite à [LOC_XYZ]
    """
    return anonymize_text_refactored(
        value=text,
        scope_id=scope_id,
        secret_salt=secret_salt,
        level=level,
        ner_results=ner_results,
        overrides=overrides if overrides else None,
    )


__all__ = ["AnonymizationPipeline", "anonymize_text", "AnonymizationPolicy", "preset"]
