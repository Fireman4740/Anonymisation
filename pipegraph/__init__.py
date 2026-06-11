"""PipeGraph — hybrid text anonymization pipeline (regex + NER + LLM).

Public API::

    from pipegraph import anonymize, anonymize_file, AnonymizationResult
"""

from pipegraph.api import AnonymizationResult, anonymize, anonymize_file

__all__ = ["anonymize", "anonymize_file", "AnonymizationResult"]
