from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipegraph"))

from src.nodes.llm.provider import (
    LLM_MOCK_ENV_VAR,
    LLMProvider,
    MockLLMProvider,
    get_llm_client,
    mock_enabled,
)


def test_mock_default_responses_are_safe_noops():
    detect = MockLLMProvider(role="detect").chat([{"role": "user", "content": "x"}])
    assert json.loads(detect) == {"entities": []}

    audit = MockLLMProvider(role="audit").chat([{"role": "user", "content": "x"}])
    assert json.loads(audit)["privacy_score"] == 0

    verify = MockLLMProvider(role="verify").chat([{"role": "user", "content": "x"}])
    assert json.loads(verify) == []

    paraphrase = MockLLMProvider(role="paraphrase").chat([{"role": "user", "content": "x"}])
    assert paraphrase is None  # nodes keep original text on missing response


def test_mock_records_calls_and_cycles_responses():
    mock = MockLLMProvider(role="audit", responses=['{"privacy_score": 80}', '{"privacy_score": 10}'])
    first = mock.chat([{"role": "user", "content": "round 1"}])
    second = mock.chat([{"role": "user", "content": "round 2"}])
    third = mock.chat([{"role": "user", "content": "round 3"}])

    assert json.loads(first)["privacy_score"] == 80
    assert json.loads(second)["privacy_score"] == 10
    assert json.loads(third)["privacy_score"] == 10  # last response repeats
    assert len(mock.calls) == 3
    assert mock.calls[0]["messages"][0]["content"] == "round 1"


def test_mock_chat_batch_preserves_order():
    mock = MockLLMProvider(role="verify")
    out = mock.chat_batch([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]])
    assert len(out) == 2
    assert len(mock.calls) == 2


def test_mock_satisfies_provider_protocol():
    assert isinstance(MockLLMProvider(role="detect"), LLMProvider)


def test_get_llm_client_returns_mock_from_runtime_flag():
    client = get_llm_client(role="audit", runtime={"llm_mock": True})
    assert isinstance(client, MockLLMProvider)


def test_get_llm_client_returns_mock_from_env(monkeypatch):
    monkeypatch.setenv(LLM_MOCK_ENV_VAR, "1")
    assert mock_enabled() is True
    client = get_llm_client(role="detect", runtime={})
    assert isinstance(client, MockLLMProvider)


def test_get_llm_client_falls_back_to_default(monkeypatch):
    monkeypatch.delenv(LLM_MOCK_ENV_VAR, raising=False)
    sentinel = object()
    assert get_llm_client(role="detect", runtime={}, default=sentinel) is sentinel
