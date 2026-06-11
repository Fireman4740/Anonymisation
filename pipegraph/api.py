"""Public API for the PipeGraph anonymization pipeline.

Minimal usage (no knowledge of internal modules required)::

    from pipegraph.api import anonymize

    result = anonymize("Je m'appelle Jean Dupont, jean.dupont@example.com")
    print(result.anonymized_text)
    print(result.entities)

The heavy pipeline (langgraph, GLiNER, torch, …) is imported lazily on the
first call, so importing this module is cheap and test-friendly.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

_PIPEGRAPH_DIR = os.path.dirname(os.path.abspath(__file__))

ConfigLike = Union[Dict[str, Any], str, "os.PathLike[str]", None]

NO_LLM_OVERRIDES: Dict[str, Any] = {
    "disable_llm": True,
    "llm_detection": False,
    "llm_verification": False,
    "llm_audit": False,
    "llm_paraphrase": False,
    "rupta_enabled": False,
}


@dataclass(frozen=True)
class AnonymizationResult:
    """Auditable result of one pipeline run.

    Offsets in ``entities`` always reference ``original_text``.
    """

    original_text: str
    anonymized_text: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    privacy_score: Optional[int] = None
    llm_feedback: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "anonymized_text": self.anonymized_text,
            "entities": self.entities,
            "privacy_score": self.privacy_score,
            "llm_feedback": self.llm_feedback,
            "iterations": self.iterations,
            "errors": self.errors,
            "metadata": self.metadata,
            "config_snapshot": self.config_snapshot,
        }


def load_config(config: ConfigLike) -> Dict[str, Any]:
    """Resolve a runtime config: dict passthrough, or JSON/YAML file path.

    Keys starting with ``_`` (e.g. ``_doc``) are stripped — they are comments.
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        return {k: v for k, v in config.items() if not str(k).startswith("_")}

    path = os.fspath(config)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            import yaml  # optional dependency, only needed for YAML configs

            raw = yaml.safe_load(handle)
        else:
            raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain an object/mapping: {path}")
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def _build_pipeline() -> Tuple[Any, Callable[..., Dict[str, Any]]]:
    """Lazy import of the LangGraph pipeline.

    Internal modules use ``from src.X import …``, so the pipegraph directory
    must be on sys.path (same convention as eval/core/bootstrap.py).
    """
    if _PIPEGRAPH_DIR not in sys.path:
        sys.path.insert(0, _PIPEGRAPH_DIR)

    from src.graph import create_pipeline_graph  # type: ignore
    from src.state import create_initial_state  # type: ignore

    return create_pipeline_graph(), create_initial_state


def anonymize(
    text: str,
    config: ConfigLike = None,
    *,
    no_llm: bool = False,
) -> AnonymizationResult:
    """Anonymize a text through the full PipeGraph pipeline.

    Args:
        text: Input text. Entity offsets in the result reference this text.
        config: Runtime config — either a dict of ``state.config`` keys
            (``enable_deterministic``, ``enable_ai``, ``llm_audit``,
            ``anon_strategy``, ``anon_policy``, ``scope_id``, …) or a path to
            a JSON/YAML file (e.g. ``configs/baselines/no_llm.json``).
        no_llm: Force-disable every LLM node and the RUPTA loop. The pipeline
            then runs fully offline (regex + NER only).

    Raises:
        ValueError: If ``text`` is not a non-empty string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("anonymize() requires a non-empty text")

    runtime_config: Dict[str, Any] = load_config(config)
    if no_llm:
        runtime_config.update(NO_LLM_OVERRIDES)

    pipeline, create_initial_state = _build_pipeline()
    initial_state = create_initial_state(text, runtime_config)
    final_state = pipeline.invoke(initial_state)

    return AnonymizationResult(
        original_text=final_state.get("original_text", text),
        anonymized_text=final_state.get("text", text),
        entities=list(final_state.get("entities", [])),
        privacy_score=final_state.get("privacy_score"),
        llm_feedback=dict(final_state.get("llm_feedback", {})),
        iterations=int(final_state.get("iteration", 0)),
        errors=list(final_state.get("errors", [])),
        metadata=dict(final_state.get("metadata", {})),
        config_snapshot=runtime_config,
    )


def anonymize_file(
    input_path: str,
    output_path: str,
    config: ConfigLike = None,
    *,
    no_llm: bool = False,
    encoding: str = "utf-8",
) -> AnonymizationResult:
    """Anonymize a text file and write the anonymized version.

    Returns the same :class:`AnonymizationResult` as :func:`anonymize`.
    """
    with open(input_path, "r", encoding=encoding) as handle:
        text = handle.read()

    result = anonymize(text, config, no_llm=no_llm)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding=encoding) as handle:
        handle.write(result.anonymized_text)

    return result
