
"""
API Publique

Interface simple et claire pour utiliser le pipeline d'anonymisation.
"""

from .pipeline import AnonymizationPipeline, anonymize_text
from .policy import AnonymizationPolicy, preset

__all__ = ["AnonymizationPipeline", "anonymize_text", "AnonymizationPolicy", "preset"]
