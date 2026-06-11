from __future__ import annotations

import argparse
import json

import pytest

# arc_pipegraph (AutoResearchClaw harness) is gitignored and only present on
# the harness machine — skip cleanly elsewhere instead of breaking collection.
pytest.importorskip("arc_pipegraph")

from arc_pipegraph.objective import compute_primary_metric, score_dataset
from arc_pipegraph.pipeline_adapter import (
    candidate_from_payload,
    build_pipegraph_runtime_config,
)


def test_candidate_validation_clamps_and_forces_llm_runtime():
    candidate = candidate_from_payload(
        {
            "candidate_id": "bad-but-safe",
            "config": {
                "gliner_threshold": 9.0,
                "ner_min_len": -4,
                "rupta_max_iterations": 99,
                "disable_llm": True,
                "llm_detection": False,
                "unknown_key": "ignored",
            },
        }
    )

    assert candidate.config["gliner_threshold"] == 0.95
    assert candidate.config["ner_min_len"] == 1
    assert candidate.config["rupta_max_iterations"] == 5
    assert "disable_llm" in candidate.ignored_keys
    assert "llm_detection" in candidate.ignored_keys

    runtime = build_pipegraph_runtime_config(candidate, dataset_key="tab")
    assert runtime["disable_llm"] is False
    assert runtime["llm_detection"] is True
    assert runtime["llm_verification"] is True
    assert runtime["llm_audit"] is True
    assert runtime["llm_paraphrase"] is True
    assert runtime["rupta_enabled"] is True


def test_primary_metric_is_bounded():
    scored = score_dataset(
        {
            "n_documents": 10,
            "micro_f2": 3.0,
            "micro_recall": 1.2,
            "micro_precision": -1.0,
            "micro_exact_label_recall": 0.5,
            "total_ground_truth": 5,
            "total_leaks": 2,
        },
        elapsed_s=10.0,
    )

    assert 0.0 <= scored["score"] <= 1.0
    assert 0.0 <= scored["components"]["privacy_score"] <= 1.0

    primary, aggregate = compute_primary_metric({"tab": scored})
    assert 0.0 <= primary <= 1.0
    assert aggregate["weights"]["tab"] == 1.0


def test_dataset_error_keeps_partial_score():
    primary, aggregate = compute_primary_metric(
        {
            "tab": {"score": 0.8, "status": "ok"},
            "dbbio": {"score": 0.0, "status": "error"},
        }
    )

    assert 0.0 < primary < 0.8
    assert set(aggregate["weights"]) == {"tab", "dbbio"}


def test_cli_main_prints_final_json(monkeypatch, capsys, tmp_path):
    from arc_pipegraph import evaluate_candidate

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps({"candidate_id": "stub", "config": {"gliner_threshold": 0.2}}),
        encoding="utf-8",
    )

    def fake_run_evaluation(args: argparse.Namespace):
        assert args.candidate == str(candidate_path)
        return {
            "status": "ok",
            "primary_metric": 0.42,
            "metric_direction": "maximize",
            "candidate_id": "stub",
            "datasets": {"tab": {"score": 0.42}},
            "aggregate": {},
            "errors": [],
        }

    monkeypatch.setattr(evaluate_candidate, "run_evaluation", fake_run_evaluation)

    rc = evaluate_candidate.main(["--candidate", str(candidate_path), "--datasets", "tab"])

    assert rc == 0
    final_line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(final_line)
    assert payload["primary_metric"] == 0.42
    assert payload["metric_direction"] == "maximize"
