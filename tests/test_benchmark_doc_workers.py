from __future__ import annotations

import time
import sys
import types
from typing import Any, Dict, List

from eval.core.pipeline import build_report, resolve_doc_workers


def test_run_pipegraph_eval_passes_doc_workers(monkeypatch):
    fake_streamlit = types.SimpleNamespace(
        cache_resource=lambda func=None, **kwargs: func if func is not None else (lambda wrapped: wrapped)
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    from eval.streamlit_app.anonymization_error_analysis.services import pipegraph_service

    captured: Dict[str, Any] = {}

    monkeypatch.setattr(
        pipegraph_service,
        "load_pipegraph_cached",
        lambda: (lambda: object(), lambda text, config: {"text": text, "config": config}),
    )
    monkeypatch.setattr("eval.core.datasets.normalize_dataset_key", lambda dataset_kind: "tab")
    monkeypatch.setattr(
        "eval.core.datasets.load_local_dataset_docs",
        lambda **kwargs: [("doc-1", "John", [(0, 4, "PERSON")])],
    )
    monkeypatch.setattr("eval.core.datasets.get_allowed_labels", lambda *args, **kwargs: frozenset({"PERSON"}))

    def fake_build_report(*args, **kwargs):
        captured["max_workers"] = kwargs.get("max_workers")
        return []

    monkeypatch.setattr("eval.core.pipeline.build_report", fake_build_report)

    pipegraph_service.run_pipegraph_eval(
        dataset_kind="tab",
        dataset_path="unused.jsonl",
        split="test",
        limit=1,
        config={},
        doc_workers=4,
    )

    assert captured["max_workers"] == 4


def test_build_ratbench_report_passes_doc_workers(monkeypatch):
    from eval.core import ratbench

    captured: Dict[str, Any] = {}

    def fake_build_report(*args, **kwargs):
        captured["max_workers"] = kwargs.get("max_workers")
        return []

    monkeypatch.setattr(ratbench, "build_report", fake_build_report)

    ratbench.build_ratbench_report(
        [("ratbench_1_L1", "John", [(0, 4, "PERSON")])],
        [],
        object(),
        lambda text, config: {"text": text, "config": config},
        max_workers=2,
    )

    assert captured["max_workers"] == 2


def test_resolve_doc_workers_auto_openrouter(monkeypatch):
    monkeypatch.setattr(
        "eval.core.pipeline._load_pipegraph_config_safe",
        lambda: {
            "llm": {"provider": "openrouter"},
            "pipeline": {"nodes": {"detection": {"ai_ner": {"gliner": {"use_gpu": False}}}}},
            "ner_gpu": {"enabled": False},
        },
    )

    assert resolve_doc_workers({"llm_provider": "openrouter", "llm_detection": True, "enable_ai": False}, doc_count=10) == 4


def test_resolve_doc_workers_auto_local_provider(monkeypatch):
    monkeypatch.setattr(
        "eval.core.pipeline._load_pipegraph_config_safe",
        lambda: {"llm": {"provider": "ollama"}},
    )

    assert resolve_doc_workers({"llm_detection": True}, doc_count=10) == 1


def test_resolve_doc_workers_auto_openrouter_gpu_caps_to_two(monkeypatch):
    monkeypatch.setattr(
        "eval.core.pipeline._load_pipegraph_config_safe",
        lambda: {
            "llm": {"provider": "openrouter"},
            "pipeline": {"nodes": {"detection": {"ai_ner": {"gliner": {"use_gpu": True}}}}},
            "ner_gpu": {"enabled": True},
        },
    )

    assert resolve_doc_workers({"llm_provider": "openrouter", "llm_audit": True, "enable_ai": True}, doc_count=10) == 2


def test_resolve_doc_workers_explicit_caps_to_doc_count():
    assert resolve_doc_workers({"llm_provider": "ollama"}, requested_workers=8, doc_count=3) == 3


def test_build_report_parallel_preserves_document_order():
    class FakePipeline:
        def invoke(self, state):
            if state["text"] == "slow":
                time.sleep(0.03)
            return {"text": state["text"], "entities": []}

    def create_initial_state(text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": text, "original_text": text, "config": config}

    docs: List[tuple[str, str, list[tuple[int, int, str]]]] = [
        ("doc-slow", "slow", []),
        ("doc-fast", "fast", []),
    ]

    report = build_report(
        docs,
        FakePipeline(),
        create_initial_state,
        config={"disable_llm": True, "doc_timeout_seconds": 5},
        max_workers=2,
    )

    assert [doc["doc_id"] for doc in report] == ["doc-slow", "doc-fast"]
