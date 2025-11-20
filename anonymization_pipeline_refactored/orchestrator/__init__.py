
"""
Orchestrateur - Coordination des Couches

L'orchestrateur coordonne le flux entre les couches sans contenir
de logique métier. Il utilise l'injection de dépendances pour
faciliter les tests et la composition.
"""

from .orchestrator import anonymize_text_refactored

__all__ = ["anonymize_text_refactored"]
