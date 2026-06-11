from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List

from atlas_anno.config import load_config
from atlas_anno.io import append_jsonl, project_file, read_json, read_jsonl, write_json, write_json_atomic, write_jsonl, write_text
from atlas_anno.records import (
    anonymization_results_from_rows,
    attack_pairs_from_rows,
    attack_results_from_rows,
    characters_from_rows,
    documents_from_rows,
    scenarios_from_rows,
    worlds_from_rows,
)
from atlas_anno.schemas import AnonymizationResult, AttackPair, AttackResult, CharacterProfile, DocumentRecord, ScenarioSpec, World

_append_lock = threading.Lock()


def _path(name: str) -> Path:
    config = load_config()
    return project_file(str(config.defaults["paths"][name]))


def worlds_path() -> Path:
    return _path("worlds")


def characters_path() -> Path:
    return _path("characters")


def scenarios_path() -> Path:
    return _path("scenarios")


def raw_docs_path() -> Path:
    return _path("raw_docs")


def annotations_path() -> Path:
    return _path("annotations")


def reviewed_annotations_path() -> Path:
    return _path("reviewed_annotations")


def anonymized_path(strategy: str) -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["anonymized_dir"]))
    return directory / f"{strategy}.jsonl"


def attacks_path(strategy: str, attacker_type: str) -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["attacks_dir"]))
    return directory / f"{strategy}_{attacker_type}.jsonl"


def attack_pairs_path() -> Path:
    config = load_config()
    configured = config.defaults["paths"].get("attack_pairs")
    if configured:
        return project_file(str(configured))
    directory = project_file(str(config.defaults["paths"]["attacks_dir"]))
    return directory / "pairs.jsonl"


def report_json_path(strategy: str, name: str) -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["reports_dir"]))
    return directory / f"{strategy}_{name}.json"


def report_markdown_path(strategy: str) -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["reports_dir"]))
    return directory / f"{strategy}_report.md"


def report_html_path(strategy: str) -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["reports_dir"]))
    return directory / f"{strategy}_report.html"


def review_dir(batch: str) -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["review_dir"]))
    return directory / batch


def batches_dir(batch: str) -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["batches_dir"]))
    return directory / batch


def parquet_dir(batch: str) -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["parquet_dir"]))
    return directory / batch


def llm_runs_path() -> Path:
    config = load_config()
    directory = project_file(str(config.defaults["paths"]["logs_dir"]))
    return directory / "llm_runs.jsonl"


def llm_cache_dir() -> Path:
    return _path("cache_dir")


def llm_cache_path(step_name: str, cache_key: str) -> Path:
    return llm_cache_dir() / step_name / f"{cache_key}.json"


def label_studio_tasks_path(batch: str) -> Path:
    return review_dir(batch) / "label_studio_tasks.json"


def label_config_path(batch: str) -> Path:
    return review_dir(batch) / "label_config.xml"


def label_studio_export_path(batch: str) -> Path:
    return review_dir(batch) / "label_studio_export.json"


def label_studio_project_id_path(batch: str) -> Path:
    """Cache file storing the Label Studio project ID for a given batch."""
    return review_dir(batch) / ".label_studio_project_id"


def batch_manifest_path(batch: str) -> Path:
    return batches_dir(batch) / "manifest.json"


def stage_checkpoint_path(batch: str, step_name: str) -> Path:
    return batches_dir(batch) / "checkpoints" / f"{step_name}.jsonl"


def save_worlds(worlds: List[World]) -> Path:
    return write_jsonl(worlds_path(), worlds)


def load_worlds() -> List[World]:
    return worlds_from_rows(read_jsonl(worlds_path()))


def save_characters(characters: List[CharacterProfile]) -> Path:
    return write_jsonl(characters_path(), characters)


def load_characters() -> List[CharacterProfile]:
    return characters_from_rows(read_jsonl(characters_path()))


def save_scenarios(scenarios: List[ScenarioSpec]) -> Path:
    return write_jsonl(scenarios_path(), scenarios)


def load_scenarios() -> List[ScenarioSpec]:
    return scenarios_from_rows(read_jsonl(scenarios_path()))


def save_documents(documents: List[DocumentRecord], *, annotated: bool = False) -> Path:
    target = annotations_path() if annotated else raw_docs_path()
    return write_jsonl(target, documents)


def load_documents(*, annotated: bool = False) -> List[DocumentRecord]:
    target = annotations_path() if annotated else raw_docs_path()
    return documents_from_rows(read_jsonl(target))


def save_reviewed_documents(documents: List[DocumentRecord]) -> Path:
    return write_jsonl(reviewed_annotations_path(), documents)


def load_reviewed_documents() -> List[DocumentRecord]:
    return documents_from_rows(read_jsonl(reviewed_annotations_path()))


def save_anonymization_results(strategy: str, results: List[AnonymizationResult]) -> Path:
    return write_jsonl(anonymized_path(strategy), results)


def load_anonymization_results(strategy: str) -> List[AnonymizationResult]:
    return anonymization_results_from_rows(read_jsonl(anonymized_path(strategy)))


def save_attack_results(strategy: str, attacker_type: str, results: List[AttackResult]) -> Path:
    return write_jsonl(attacks_path(strategy, attacker_type), results)


def load_attack_results(strategy: str, attacker_type: str) -> List[AttackResult]:
    path = attacks_path(strategy, attacker_type)
    if not path.exists():
        return []
    return attack_results_from_rows(read_jsonl(path))


def save_attack_pairs(pairs: List[AttackPair]) -> Path:
    return write_jsonl(attack_pairs_path(), pairs)


def load_attack_pairs() -> List[AttackPair]:
    path = attack_pairs_path()
    if not path.exists():
        return []
    return attack_pairs_from_rows(read_jsonl(path))


def save_report(strategy: str, name: str, payload: Dict[str, Any]) -> Path:
    return write_json(report_json_path(strategy, name), payload)


def load_report(strategy: str, name: str) -> Dict[str, Any]:
    path = report_json_path(strategy, name)
    if not path.exists():
        return {}
    return read_json(path)


def append_llm_run(payload: Dict[str, Any]) -> Path:
    with _append_lock:
        return append_jsonl(llm_runs_path(), payload)


def load_llm_cache_entry(step_name: str, cache_key: str) -> Dict[str, Any] | None:
    path = llm_cache_path(step_name, cache_key)
    if not path.exists():
        return None
    return read_json(path)


def save_llm_cache_entry(step_name: str, cache_key: str, payload: Dict[str, Any]) -> Path:
    return write_json_atomic(llm_cache_path(step_name, cache_key), payload)


def load_stage_checkpoints(batch: str, step_name: str) -> List[Dict[str, Any]]:
    return read_jsonl(stage_checkpoint_path(batch, step_name))


def append_stage_checkpoint(batch: str, step_name: str, payload: Dict[str, Any]) -> Path:
    with _append_lock:
        return append_jsonl(stage_checkpoint_path(batch, step_name), payload)


def save_label_studio_tasks(batch: str, payload: List[Dict[str, Any]]) -> Path:
    return write_json(label_studio_tasks_path(batch), payload)


def save_label_config(batch: str, payload: str) -> Path:
    return write_text(label_config_path(batch), payload)


def save_label_studio_export(batch: str, payload: List[Dict[str, Any]]) -> Path:
    return write_json(label_studio_export_path(batch), payload)


def save_batch_manifest(batch: str, payload: Dict[str, Any]) -> Path:
    return write_json(batch_manifest_path(batch), payload)


def load_batch_manifest(batch: str) -> Dict[str, Any]:
    path = batch_manifest_path(batch)
    if not path.exists():
        return {}
    return read_json(path)
