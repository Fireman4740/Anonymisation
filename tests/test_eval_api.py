"""Tests for the unified evaluation API (eval.api) and CLI (eval.cli.main).

No real pipeline, no LLM, no API key: the engine's pipeline loader and
dataset loaders are monkeypatched, same pattern as
test_official_pipeline_evaluation.py.
"""

from __future__ import annotations

import json

import pytest

from eval import run_pipeline_evaluation as engine
from eval.api import (
    EvaluationRunner,
    compare_runs,
    list_available_configs,
    list_available_datasets,
    load_ablation_plan,
    load_eval_config,
    load_metrics,
    load_predictions,
)
from eval.cli.main import main as cli_main
from eval.registry import DatasetAdapter, DatasetRegistry, get_registry


class _FakePipeline:
    def invoke(self, state):
        text = state["text"]
        return {
            "text": text.replace("John", "[PER]"),
            "entities": [{"start": 0, "end": 4, "type": "PERSON", "source": "test"}],
        }


def _fake_initial_state(text, config):
    return {"text": text, "original_text": text, "config": config}


_FAKE_DOCS = [
    ("doc-1", "John met Alice in Paris.", [(0, 4, "PERSON")]),
    ("doc-2", "John called the office.", [(0, 4, "PERSON")]),
]


@pytest.fixture
def fake_engine(monkeypatch):
    captured = {}

    def fake_load_benchmark_docs(**kwargs):
        captured.setdefault("load_calls", []).append(kwargs)
        return list(_FAKE_DOCS), f"{kwargs.get('dataset')}/fake"

    monkeypatch.setattr(engine, "_load_pipeline", lambda: (_FakePipeline(), _fake_initial_state))
    monkeypatch.setattr(engine, "load_benchmark_docs", fake_load_benchmark_docs)
    monkeypatch.setattr(
        engine,
        "load_ratbench_profiles",
        lambda **kwargs: [
            {
                "id": 1,
                "difficulty": kwargs.get("level", 1),
                "scenario": "memo",
                "text": "John met Alice in Paris.",
                "direct_identifiers": {"name": "John"},
                "indirect_identifiers": {},
                "profile": {"name": "John"},
            }
        ],
    )
    return captured


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

def test_registry_lists_known_datasets():
    names = get_registry().list()
    for expected in ("tab", "dbbio", "ratbench", "conll2003", "anonymization", "personalreddit"):
        assert expected in names


def test_registry_resolves_aliases():
    registry = get_registry()
    assert registry.get("rat-bench").name == "ratbench"
    assert registry.get("DB-Bio").name == "dbbio"
    assert "reddit" in registry
    with pytest.raises(KeyError):
        registry.get("nope")


def test_registry_register_and_describe():
    registry = DatasetRegistry()
    adapter = DatasetAdapter(
        name="mockset",
        description="test dataset",
        supports={"span_metrics": True},
        loader=lambda split, limit, **kw: (list(_FAKE_DOCS), "mockset/test"),
    )
    registry.register(adapter)
    docs, name = registry.get("mockset").load(split="test", limit=None)
    assert len(docs) == 2 and name == "mockset/test"
    with pytest.raises(ValueError):
        registry.register(adapter)  # duplicate
    info = registry.get("mockset").describe()
    assert info["supports"]["span_metrics"] is True


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------

def test_load_eval_config_strips_comment_keys(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"_doc": "x", "limit": 7}), encoding="utf-8")
    config = load_eval_config(str(path))
    assert config == {"limit": 7}


def test_load_eval_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_eval_config("does/not/exist.json")


def test_shipped_eval_configs_are_valid():
    configs = list_available_configs()
    assert any("no_llm" in path for path in configs)
    for path in configs:
        if "ablations" in path:
            load_ablation_plan(path)
        else:
            load_eval_config(path)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def test_runner_evaluates_single_dataset(fake_engine, tmp_path):
    runner = EvaluationRunner({"run_name": "t", "limit": 2, "no_llm": True, "skip_risk": True})
    payload = runner.run(dataset="tab", output=str(tmp_path / "run"))

    assert payload["status"] == "ok"
    assert "tab" in payload["datasets"]
    assert payload["datasets"]["tab"]["n_documents"] == 2
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "run_config.json").exists()


def test_runner_all_uses_config_dataset_list(fake_engine, tmp_path):
    runner = EvaluationRunner(
        {"datasets": ["tab", "dbbio"], "limit": 1, "no_llm": True, "skip_risk": True}
    )
    payload = runner.run(dataset="all", output=str(tmp_path / "run"))

    assert set(payload["datasets"]) == {"tab", "dbbio"}


def test_runner_default_all_accepts_every_registry_dataset(fake_engine, tmp_path):
    runner = EvaluationRunner({"limit": 1, "no_llm": True, "skip_risk": True})
    payload = runner.run(dataset="all", output=str(tmp_path / "run"))

    expected = set(get_registry().list())
    expected.remove("ratbench")
    expected.add("ratbench/english/L1")
    assert set(payload["datasets"]) == expected
    assert "personalreddit" in payload["datasets"]


def test_dataset_adapters_expand_requests():
    import argparse

    registry = get_registry()
    args = argparse.Namespace(
        split="test",
        limit=5,
        language="english",
        ratbench_languages=["english", "spanish"],
        ratbench_levels=[1, 3],
    )

    for dataset in ("tab", "dbbio", "anonymization", "conll2003", "personalreddit"):
        requests = registry.get(dataset).expand_run_requests(args)
        assert [request.dataset_key for request in requests] == [dataset]

    ratbench_requests = registry.get("ratbench").expand_run_requests(args)
    assert [request.dataset_key for request in ratbench_requests] == [
        "ratbench/english/L1",
        "ratbench/english/L3",
        "ratbench/spanish/L1",
        "ratbench/spanish/L3",
    ]


def test_runner_pipeline_overrides_become_candidate(fake_engine, tmp_path):
    runner = EvaluationRunner({"limit": 1, "no_llm": True, "skip_risk": True})
    payload = runner.run(
        dataset="tab",
        output=str(tmp_path / "run"),
        pipeline_overrides={"gliner_threshold": 0.4},
    )

    effective = json.loads(
        (tmp_path / "run" / "candidate_effective_config.json").read_text(encoding="utf-8")
    )
    assert effective["config"].get("gliner_threshold") == 0.4
    assert payload["status"] == "ok"


def test_run_artifacts_are_loadable(fake_engine, tmp_path):
    runner = EvaluationRunner({"limit": 1, "no_llm": True, "skip_risk": True})
    run_dir = tmp_path / "run"
    runner.run(dataset="tab", output=str(run_dir))

    metrics = load_metrics(run_dir)
    assert metrics["status"] == "ok"
    predictions = load_predictions(run_dir)
    assert predictions and predictions[0]["doc_id"] == "doc-1"
    assert (run_dir / "errors.jsonl").exists()  # empty file when no errors


def test_errors_jsonl_records_dataset_failures(fake_engine, tmp_path):
    runner = EvaluationRunner({"limit": 1, "no_llm": True, "skip_risk": True})
    run_dir = tmp_path / "run"
    payload = runner.run(dataset=["tab", "unknown_ds"], output=str(run_dir))

    assert payload["status"] in ("partial", "error")
    lines = (run_dir / "errors.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["dataset"] == "unknown_ds"


# ----------------------------------------------------------------------
# Ablations
# ----------------------------------------------------------------------

def test_ablation_plan_validation():
    with pytest.raises(ValueError):
        load_ablation_plan({"ablations": []})
    with pytest.raises(ValueError):
        load_ablation_plan({"ablations": [{"overrides": {}}]})  # missing name
    with pytest.raises(ValueError):
        load_ablation_plan({"ablations": [{"name": "a"}, {"name": "a"}]})  # duplicate

    plan = load_ablation_plan(
        {"ablations": [{"name": "x", "overrides": {"k": 1}, "runner_overrides": {"no_llm": True}}]}
    )
    assert plan[0]["overrides"] == {"k": 1}
    assert plan[0]["runner_overrides"] == {"no_llm": True}


def test_ablation_runs_variants_and_writes_summary(fake_engine, tmp_path):
    runner = EvaluationRunner({"limit": 1, "no_llm": True, "skip_risk": True})
    summary = runner.run_ablation(
        dataset="tab",
        ablation_config={
            "ablations": [
                {"name": "variant_a", "overrides": {"enable_ai": False}},
                {"name": "variant_b", "overrides": {}},
            ]
        },
        output=str(tmp_path / "abl"),
    )

    assert len(summary["variants"]) == 2
    assert (tmp_path / "abl" / "ablation_summary.csv").exists()
    assert (tmp_path / "abl" / "ablation_report.md").exists()
    assert (tmp_path / "abl" / "variant_a" / "summary.json").exists()
    assert (tmp_path / "abl" / "variant_b" / "summary.json").exists()
    effective = json.loads(
        (tmp_path / "abl" / "variant_a" / "candidate_effective_config.json").read_text()
    )
    assert effective["config"].get("enable_ai") is False


# ----------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------

def test_compare_runs_produces_artifacts(fake_engine, tmp_path):
    runner = EvaluationRunner({"limit": 1, "no_llm": True, "skip_risk": True})
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    runner.run(dataset="tab", output=str(run_a))
    runner.run(dataset="tab", output=str(run_b))

    comparison = compare_runs([str(run_a), str(run_b)], output_dir=str(tmp_path / "cmp"))

    assert len(comparison["runs"]) == 2
    assert comparison["runs"][1]["delta_vs_first"] == pytest.approx(0.0)
    assert (tmp_path / "cmp" / "comparison.csv").exists()
    assert (tmp_path / "cmp" / "comparison_report.md").exists()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def test_cli_list_datasets(capsys):
    rc = cli_main(["list-datasets"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tab" in out and "ratbench" in out


def test_cli_run(fake_engine, tmp_path, capsys):
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps({"limit": 1, "no_llm": True, "skip_risk": True}), encoding="utf-8"
    )
    rc = cli_main(
        ["run", "--dataset", "tab", "--config", str(config), "--output", str(tmp_path / "run")]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "tab" in out
    assert str(tmp_path / "run") in out


def test_streamlit_eval_imports_and_ratbench_risk_module_smoke():
    from eval.api import build_eval_config
    from eval.cli.evaluate_ratbench_risk import evaluate_ratbench_risk_from_pipeline
    from eval.streamlit_app.anonymization_error_analysis.services import ratbench_service

    cfg = build_eval_config(dataset="tab", no_llm=True)
    assert cfg["disable_llm"] is True
    assert callable(evaluate_ratbench_risk_from_pipeline)
    assert callable(ratbench_service.run_ratbench_eval)


def test_cli_ablation_and_report(fake_engine, tmp_path, capsys):
    ablation = tmp_path / "abl.json"
    ablation.write_text(
        json.dumps({"ablations": [{"name": "only", "overrides": {}}]}), encoding="utf-8"
    )
    rc = cli_main(
        ["ablation", "--dataset", "tab", "--ablation-config", str(ablation),
         "--output", str(tmp_path / "abl")]
    )
    assert rc == 0
    capsys.readouterr()

    rc = cli_main(["report", "--run", str(tmp_path / "abl" / "only"), "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["status"] == "ok"


def test_ui_helpers():
    datasets = list_available_datasets()
    assert any(info["name"] == "ratbench" and info["supports"]["risk_metrics"] for info in datasets)
