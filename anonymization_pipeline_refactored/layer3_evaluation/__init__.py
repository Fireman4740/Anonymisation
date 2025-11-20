
"""
Couche 3 : Évaluation

Cette couche fournit des métriques et de la validation:
- Métriques de base
- Validation des outputs
- Logs et traces
"""

from .evaluator import Evaluator, EvaluationResult

__all__ = ["Evaluator", "EvaluationResult"]
