from __future__ import annotations

import json
import os
from typing import Any, Dict

import pytest

from pipegraph import api as pipegraph_api
from pipegraph.api import AnonymizationResult, anonymize, anonymize_file
from pipegraph.cli import main as cli_main


class _FakePipeline:
    """Stands in for the compiled LangGraph — no ML dependency needed."""

    def __init__(self) -> None:
        self.last_state: Dict[str, Any] = {}

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.last_state = state
        text = state["text"]
        return {
            **state,
            "text": text.replace("Jean Dupont", "[PERSON]"),
            "entities": [
                {"start": 0, "end": 11, "type": "PERSON", "value": "Jean Dupont", "source": "regex", "score": 1.0}
            ],
            "privacy_score": 12,
            "iteration": 1,
        }


def _fake_initial_state(text: str, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "text": text,
        "original_text": text,
        "entities": [],
        "config": config or {},
        "metadata": {},
        "errors": [],
        "privacy_score": 100,
        "llm_feedback": {},
        "iteration": 0,
    }


@pytest.fixture
def fake_pipeline(monkeypatch) -> _FakePipeline:
    pipeline = _FakePipeline()
    monkeypatch.setattr(
        pipegraph_api, "_build_pipeline", lambda: (pipeline, _fake_initial_state)
    )
    return pipeline


def test_anonymize_returns_auditable_result(fake_pipeline):
    result = anonymize("Jean Dupont habite à Paris")

    assert isinstance(result, AnonymizationResult)
    assert result.original_text == "Jean Dupont habite à Paris"
    assert result.anonymized_text == "[PERSON] habite à Paris"
    assert result.entities[0]["source"] == "regex"
    assert result.privacy_score == 12
    assert result.iterations == 1


def test_anonymize_rejects_empty_text(fake_pipeline):
    with pytest.raises(ValueError):
        anonymize("   ")


def test_no_llm_flag_disables_all_llm_nodes(fake_pipeline):
    result = anonymize("Jean Dupont", no_llm=True)

    config = fake_pipeline.last_state["config"]
    assert config["disable_llm"] is True
    assert config["llm_detection"] is False
    assert config["llm_audit"] is False
    assert config["llm_paraphrase"] is False
    assert config["rupta_enabled"] is False
    assert result.config_snapshot["disable_llm"] is True


def test_config_overrides_are_passed_through(fake_pipeline):
    anonymize("Jean Dupont", {"anon_strategy": "mask", "scope_id": "doc-7"})

    config = fake_pipeline.last_state["config"]
    assert config["anon_strategy"] == "mask"
    assert config["scope_id"] == "doc-7"


def test_anonymize_file_roundtrip(fake_pipeline, tmp_path):
    input_path = tmp_path / "in.txt"
    output_path = tmp_path / "out" / "anon.txt"
    input_path.write_text("Jean Dupont habite à Paris", encoding="utf-8")

    result = anonymize_file(str(input_path), str(output_path))

    assert output_path.read_text(encoding="utf-8") == "[PERSON] habite à Paris"
    assert result.entities


def test_cli_anonymize_json_hides_original_by_default(fake_pipeline, capsys):
    rc = cli_main(["anonymize", "--text", "Jean Dupont habite à Paris", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "original_text" not in payload
    assert payload["anonymized_text"] == "[PERSON] habite à Paris"


def test_cli_rejects_invalid_config_overrides(fake_pipeline):
    with pytest.raises(SystemExit):
        cli_main(["anonymize", "--text", "x", "--config-overrides", "not-json"])


def test_anonymize_accepts_config_file_path(fake_pipeline, tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps({"_doc": "comment stripped", "anon_strategy": "redact"}), encoding="utf-8"
    )

    result = anonymize("Jean Dupont", str(config_path))

    config = fake_pipeline.last_state["config"]
    assert config["anon_strategy"] == "redact"
    assert "_doc" not in config
    assert result.config_snapshot["anon_strategy"] == "redact"


def test_baseline_configs_are_valid_and_consistent(fake_pipeline):
    import glob

    # pipeline runtime configs only — configs/evaluation/* are eval-engine configs
    root = os.path.join(os.path.dirname(__file__), "..")
    config_paths = glob.glob(os.path.join(root, "configs", "baselines", "*.json")) + glob.glob(
        os.path.join(root, "configs", "ablations", "*.json")
    )
    assert len(config_paths) >= 8

    for path in config_paths:
        anonymize("Jean Dupont", path)
        config = fake_pipeline.last_state["config"]
        assert config.get("enable_anonymization") is True, path
        # no-LLM configs must disable RUPTA too (paraphrase loop needs audit)
        if config.get("disable_llm"):
            assert config.get("rupta_enabled") is False, path
        if config.get("llm_audit") is False:
            assert config.get("rupta_enabled") is False, path


def test_cli_config_file_flag(fake_pipeline, tmp_path, capsys):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"anon_strategy": "mask"}), encoding="utf-8")

    rc = cli_main(
        ["anonymize", "--text", "Jean Dupont", "--config", str(config_path),
         "--config-overrides", '{"scope_id": "s1"}', "--json"]
    )

    assert rc == 0
    config = fake_pipeline.last_state["config"]
    assert config["anon_strategy"] == "mask"
    assert config["scope_id"] == "s1"  # overrides applied on top of file


def test_cli_missing_config_file_fails_cleanly(fake_pipeline):
    with pytest.raises(SystemExit):
        cli_main(["anonymize", "--text", "x", "--config", "does/not/exist.json"])
