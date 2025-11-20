
"""
Couche 2 : Transformation et Pseudonymisation

Cette couche applique les transformations pour anonymiser le texte:
- Remplacement par placeholders
- Généralisation policy-driven
- Paraphrase LLM
- Optimisation RUPTA (privacy-utility)
"""

from .transformation_service import TransformationService, Generalization

__all__ = ["TransformationService", "Generalization"]
