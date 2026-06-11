"""LLM provider seam — single entry point for nodes, mockable in tests.

Every LLM node resolves its client through :func:`get_llm_client` instead of
instantiating :class:`LLMClient` conditionally. This gives one place to:

- inject :class:`MockLLMProvider` (``state.config["llm_mock"] = True`` or
  ``PIPEGRAPH_LLM_MOCK=1``) so the full pipeline runs offline in tests,
- apply runtime provider/model overrides (ablations, Streamlit, benchmarks).

The mock returns safe no-op responses per role: detection adds no entities,
verification keeps everything, audit scores 0 (no RUPTA loop), paraphrase
returns ``None`` (nodes keep the original text on missing responses).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from src.nodes.llm.llm_client import LLMClient

LLM_MOCK_ENV_VAR = "PIPEGRAPH_LLM_MOCK"

# Safe no-op canned responses, keyed by LLMClient role
_DEFAULT_MOCK_RESPONSES: Dict[str, Optional[str]] = {
    "detect": json.dumps({"entities": []}),
    "verify": json.dumps([]),
    "audit": json.dumps(
        {"privacy_score": 0, "leaked_attributes": [], "assessment": "mock audit"}
    ),
    "paraphrase": None,
}


@runtime_checkable
class LLMProvider(Protocol):
    """Interface implemented by LLMClient and MockLLMProvider."""

    role: str
    provider: str
    model: str

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Optional[str]: ...

    def chat_batch(
        self,
        message_sets: Sequence[List[Dict[str, str]]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_workers: Optional[int] = None,
    ) -> List[Optional[str]]: ...

    def is_available(self) -> bool: ...


class MockLLMProvider:
    """In-memory LLM stand-in. Records every call, never touches the network.

    Args:
        role: LLM role ("detect", "verify", "audit", "paraphrase").
        responses: Optional list of canned responses, consumed in order then
            repeated. Defaults to the safe no-op response for the role.
    """

    provider = "mock"
    model = "mock"

    def __init__(self, role: str = "detect", responses: Optional[List[Optional[str]]] = None):
        self.role = role
        self._responses = list(responses) if responses else None
        self._call_index = 0
        self.calls: List[Dict[str, Any]] = []

    def _next_response(self) -> Optional[str]:
        if self._responses:
            response = self._responses[min(self._call_index, len(self._responses) - 1)]
            self._call_index += 1
            return response
        return _DEFAULT_MOCK_RESPONSES.get(self.role)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self._next_response()

    def chat_batch(
        self,
        message_sets: Sequence[List[Dict[str, str]]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_workers: Optional[int] = None,
    ) -> List[Optional[str]]:
        return [
            self.chat(messages, temperature=temperature, max_tokens=max_tokens)
            for messages in message_sets
        ]

    def is_available(self) -> bool:
        return True

    # Nodes call LLMClient.extract_json as a static method; mirror it so the
    # mock can be used as a drop-in class substitute too.
    extract_json = staticmethod(LLMClient.extract_json)


def mock_enabled(runtime: Optional[Dict[str, Any]] = None) -> bool:
    if runtime and runtime.get("llm_mock"):
        return True
    return os.environ.get(LLM_MOCK_ENV_VAR, "").strip() in ("1", "true", "yes")


def get_llm_client(
    role: str,
    runtime: Optional[Dict[str, Any]] = None,
    default: Optional[Any] = None,
) -> Any:
    """Resolve the LLM client for a node call.

    Resolution order:
    1. Mock (``runtime["llm_mock"]`` or env ``PIPEGRAPH_LLM_MOCK``) — offline.
    2. Runtime override (``runtime["llm_provider"]``) — fresh client per call.
    3. ``default`` — the node's eagerly-created client (config.json settings).
    4. New ``LLMClient`` from config.json if no default was provided.
    """
    runtime = runtime or {}
    if mock_enabled(runtime):
        return MockLLMProvider(role=role)
    if runtime.get("llm_provider"):
        return LLMClient.create(role=role, state_config=runtime)
    if default is not None:
        return default
    return LLMClient(role=role)
