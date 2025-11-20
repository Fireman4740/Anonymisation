
"""
Pipeline d'Anonymisation - Architecture Refactorisée

Architecture en 3 couches :
- Couche 1 : Détection (Regex, NER, LLM)
- Couche 2 : Transformation (Remplacement, Généralisation, Paraphrase, RUPTA)
- Couche 3 : Évaluation (Métriques, Validation)

Version: 2.0
"""

from .api import anonymize_text, AnonymizationPipeline
from .api import AnonymizationPolicy, preset

__version__ = "2.0.0"
__all__ = ["anonymize_text", "AnonymizationPipeline", "AnonymizationPolicy", "preset"]
