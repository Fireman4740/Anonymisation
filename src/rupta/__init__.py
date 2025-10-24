"""
Module RUPTA (Robust Utility-Preserving Text Anonymization)

Implémentation de la méthode RUPTA adaptée au système d'anonymisation.
Basé sur : https://github.com/UKPLab/acl2025-rupta

Composants :
- privacy_evaluator : Évalue le risque de ré-identification
- utility_evaluator : Mesure la préservation de l'utilité
- optimizer : Boucle de raffinement itératif
"""

__version__ = "0.1.0"

from .privacy_evaluator import evaluate_reidentification_risk
from .utility_evaluator import evaluate_classification_utility
from .optimizer import optimize_anonymization

__all__ = [
    "evaluate_reidentification_risk",
    "evaluate_classification_utility", 
    "optimize_anonymization",
]
