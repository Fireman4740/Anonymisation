from __future__ import annotations

from eval.core.config import build_runtime_config
from eval.core.profiles import (
    canonicalize_label,
    project_label,
    resolve_eval_profile,
)
from eval.core.pipeline import build_report


def test_canonical_label_normalization_common_aliases():
    profile = resolve_eval_profile("production_pii")

    assert canonicalize_label("ADDRESS", profile) == "ADDRESS"
    assert canonicalize_label("ADDR", profile) == "ADDRESS"
    assert canonicalize_label("DATE OF BIRTH", profile) == "DOB"
    assert canonicalize_label("DATE_OF_BIRTH", profile) == "DOB"
    assert canonicalize_label("DOB", profile) == "DOB"
    assert canonicalize_label("PERSON", profile) == "PER"
    assert canonicalize_label("PER", profile) == "PER"


def test_auto_profile_resolution_by_dataset():
    assert resolve_eval_profile("auto", dataset_key="tab").name == "tab_legal"
    assert resolve_eval_profile("auto", dataset_key="ratbench").name == "ratbench_pii"
    assert resolve_eval_profile("auto", dataset_key="cleanconll2003").name == "conll2003_news"
    assert resolve_eval_profile("auto", dataset_key="dbbio").name == "dbbio_person"


def test_benchmark_projection_differs_from_canonical_when_needed():
    rat = resolve_eval_profile("ratbench_pii")
    tab = resolve_eval_profile("tab_legal")
    conll = resolve_eval_profile("conll2003_news")

    assert project_label("ADDR", rat, target="canonical") == "ADDRESS"
    assert project_label("ADDR", rat, target="benchmark") == "ADDRESS"
    assert project_label("DATE OF BIRTH", rat, target="canonical") == "DOB"
    assert project_label("DATE OF BIRTH", rat, target="benchmark") == "DATE"
    assert project_label("PER", rat, target="benchmark") == "PERSON"
    assert project_label("PERSON", tab, target="benchmark") == "SENSITIVE"
    assert project_label("PERSON", conll, target="benchmark") == "PER"


def test_rupta_enabled_defaults_audit_and_paraphrase_on():
    cfg = build_runtime_config(
        enable_detection=True,
        enable_deterministic=True,
        enable_ai=True,
        enable_anonymization=True,
        detection_mode="serial",
        rupta_enabled=True,
        dataset_key="tab",
        profile="auto",
    )

    assert cfg["llm_audit"] is True
    assert cfg["llm_paraphrase"] is True
    assert cfg["rupta_enabled"] is True


def test_rupta_explicit_paraphrase_override_is_preserved():
    cfg = build_runtime_config(
        enable_detection=True,
        enable_deterministic=True,
        enable_ai=True,
        enable_anonymization=True,
        detection_mode="serial",
        rupta_enabled=True,
        llm_paraphrase=False,
        dataset_key="tab",
        profile="auto",
    )

    assert cfg["llm_audit"] is True
    assert cfg["llm_paraphrase"] is False


def test_masking_mode_production_keeps_dataset_detection_profile_but_production_policy():
    cfg = build_runtime_config(
        enable_detection=True,
        enable_deterministic=True,
        enable_ai=True,
        enable_anonymization=True,
        detection_mode="serial",
        dataset_key="tab",
        profile="auto",
        masking_mode="production",
    )

    assert cfg["eval_profile"] == "tab_legal"
    assert cfg["entity_profile"] == "hybrid"
    assert cfg["masking_profile"] == "production_pii"
    assert cfg["anon_policy"]["PER"] == "pseudo"


def test_build_report_stores_canonical_and_benchmark_outputs_for_tab():
    class FakePipeline:
        def invoke(self, state):
            text = state["text"]
            return {
                "text": text,
                "entities": [
                    {
                        "start": 0,
                        "end": 4,
                        "type": "PERSON",
                        "value": text[:4],
                        "source": "test",
                    }
                ],
            }

    def create_initial_state(text, config):
        return {"text": text, "original_text": text, "config": config}

    cfg = build_runtime_config(
        enable_detection=True,
        enable_deterministic=True,
        enable_ai=True,
        enable_anonymization=True,
        detection_mode="serial",
        disable_llm=True,
        dataset_key="tab",
        profile="auto",
        eval_mode="both",
        masking_mode="benchmark",
    )

    report = build_report(
        [("doc-1", "John met Alice.", [(0, 4, "SENSITIVE")])],
        FakePipeline(),
        create_initial_state,
        config=cfg,
    )

    doc = report[0]
    assert doc["masking_profile"] == "tab_legal"
    assert doc["canonical_predictions"] == [(0, 4, "PER")]
    assert doc["benchmark_predictions"] == [(0, 4, "SENSITIVE")]
    assert "canonical_metrics" in doc
    assert "benchmark_metrics" in doc
    assert doc["benchmark_anonymized_text"].startswith("[SENSITIVE]")
    assert doc["anonymized_text"].startswith("[SENSITIVE]")
