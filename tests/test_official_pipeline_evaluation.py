from __future__ import annotations

import argparse
import json

from eval.core.metrics import relaxed_overlap_metrics, strict_span_metrics


class _FakePipeline:
    def invoke(self, state):
        text = state["text"]
        return {
            "text": text.replace("John", "[PER]"),
            "entities": [{"start": 0, "end": 4, "type": "PERSON", "source": "test"}],
        }


def _fake_initial_state(text, config):
    return {"text": text, "original_text": text, "config": config}


def _args(tmp_path, **overrides):
    values = {
        "candidate": None,
        "datasets": ["tab"],
        "split": "test",
        "limit": 2,
        "output": str(tmp_path / "official-eval"),
        "save_runs": False,
        "doc_workers": 1,
        "profile": "auto",
        "eval_mode": "both",
        "masking_mode": "benchmark",
        "language": "english",
        "ratbench_languages": None,
        "ratbench_levels": [1],
        "llm_provider": None,
        "llm_model": None,
        "llm_attacker_model": None,
        "skip_risk": False,
        "require_risk": False,
        "risk_limit": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_strict_and_relaxed_metrics_are_distinct():
    truth = [(0, 4, "PER")]
    prediction = [(0, 5, "PER")]

    strict = strict_span_metrics(truth, prediction, typed=True)
    relaxed = relaxed_overlap_metrics(truth, prediction, typed=True)

    assert strict["recall"] == 0.0
    assert relaxed["recall"] == 1.0


def test_official_runner_writes_summary_and_documents(monkeypatch, tmp_path):
    from eval import run_pipeline_evaluation as runner

    monkeypatch.setattr(runner, "_load_pipeline", lambda: (_FakePipeline(), _fake_initial_state))
    monkeypatch.setattr(
        runner,
        "load_benchmark_docs",
        lambda **kwargs: ([("doc-1", "John met Alice.", [(0, 4, "SENSITIVE")])], "TAB/test"),
    )

    payload = runner.run_evaluation(_args(tmp_path))

    assert payload["status"] == "ok"
    assert payload["primary_metric_status"] == "full"
    assert "tab" in payload["datasets"]

    output_dir = tmp_path / "official-eval"
    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "manifest.json").exists()

    docs_path = output_dir / "datasets" / "tab" / "documents.jsonl"
    rows = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["doc_id"] == "doc-1"


def test_ratbench_without_openrouter_degrades_unless_required(monkeypatch, tmp_path):
    from eval import run_pipeline_evaluation as runner

    profile = {
        "id": 1,
        "difficulty": 1,
        "scenario": "memo",
        "text": "John lives in Texas.",
        "direct_identifiers": {"name": "John"},
        "indirect_identifiers": {"state of residence": "Texas"},
        "profile": {"name": "John", "state of residence": "Texas"},
    }

    monkeypatch.setattr(runner, "_load_pipeline", lambda: (_FakePipeline(), _fake_initial_state))
    monkeypatch.setattr(
        runner,
        "load_benchmark_docs",
        lambda **kwargs: ([("ratbench_1_L1", profile["text"], [(0, 4, "PERSON")])], "RAT-Bench/english/L1"),
    )
    monkeypatch.setattr(runner, "load_ratbench_profiles", lambda **kwargs: [profile])
    monkeypatch.setattr(runner, "_openrouter_key_available", lambda root: False)

    payload = runner.run_evaluation(_args(tmp_path, datasets=["ratbench"]))

    result = payload["datasets"]["ratbench/english/L1"]
    assert payload["status"] == "ok"
    assert result["score_status"] == "degraded"
    assert result["axes"]["ratbench_reid_risk"]["status"] == "risk_degraded"


def test_ratbench_require_risk_fails_dataset_when_key_missing(monkeypatch, tmp_path):
    from eval import run_pipeline_evaluation as runner

    monkeypatch.setattr(runner, "_load_pipeline", lambda: (_FakePipeline(), _fake_initial_state))
    monkeypatch.setattr(
        runner,
        "load_benchmark_docs",
        lambda **kwargs: ([("ratbench_1_L1", "John lives in Texas.", [(0, 4, "PERSON")])], "RAT-Bench/english/L1"),
    )
    monkeypatch.setattr(runner, "load_ratbench_profiles", lambda **kwargs: [])
    monkeypatch.setattr(runner, "_openrouter_key_available", lambda root: False)

    payload = runner.run_evaluation(_args(tmp_path, datasets=["ratbench"], require_risk=True))

    assert payload["status"] == "error"
    assert payload["datasets"]["ratbench/english/L1"]["status"] == "error"
    assert payload["errors"]
