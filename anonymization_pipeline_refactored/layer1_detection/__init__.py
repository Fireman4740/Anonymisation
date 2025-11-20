
"""
Couche 1 : Détection d'Entités

Cette couche encapsule toute la logique de détection d'entités sensibles:
- Détection basée sur regex (patterns PII)
- Détection NER (GLiNER)  
- Détection LLM avancée (clustering, co-référence)
"""

from .detection_service import DetectionService, DetectedEntity

__all__ = ["DetectionService", "DetectedEntity"]
