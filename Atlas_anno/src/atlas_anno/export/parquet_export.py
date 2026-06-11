from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from atlas_anno.io import flatten_record, serialize
from atlas_anno.surface_grounding import document_surface_grounding
from atlas_anno.storage import (
    load_anonymization_results,
    load_attack_pairs,
    load_attack_results,
    load_batch_manifest,
    load_characters,
    load_documents,
    load_scenarios,
    load_worlds,
    parquet_dir,
    save_batch_manifest,
)


def _pyarrow_writer(rows: List[Dict[str, object]], path: Path) -> Path:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for parquet export. Install `atlas-anno[parquet]`.") from exc

    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _flatten_rows(records: Iterable[object]) -> List[Dict[str, object]]:
    return [flatten_record(record) for record in records]


def _grounding_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for document in load_documents(annotated=False):
        for mention in document_surface_grounding(document):
            rows.append(
                {
                    "doc_id": document.doc_id,
                    "label": mention.label,
                    "canonical_value": mention.canonical_value,
                    "snippet": mention.snippet,
                    "occurrence_hint": mention.occurrence_hint,
                    "difficulty_mode": mention.difficulty_mode,
                    "hardness": mention.hardness,
                    "certainty": mention.certainty,
                    "cue_type": mention.cue_type,
                }
            )
    return rows


def export_parquet_batch(batch: str) -> Dict[str, str]:
    base_dir = parquet_dir(batch)
    datasets: List[Tuple[str, List[Dict[str, object]]]] = [
        ("worlds", _flatten_rows(load_worlds())),
        ("characters", _flatten_rows(load_characters())),
        ("scenarios", _flatten_rows(load_scenarios())),
        ("raw_docs", _flatten_rows(load_documents(annotated=False))),
        ("annotations", _flatten_rows(load_documents(annotated=True))),
        ("surface_grounding", _grounding_rows()),
        ("attack_pairs", _flatten_rows(load_attack_pairs())),
        ("anonymized_masking", _flatten_rows(load_anonymization_results("masking"))),
        ("anonymized_generalization", _flatten_rows(load_anonymization_results("generalization"))),
        ("anonymized_rewrite_balanced", _flatten_rows(load_anonymization_results("rewrite_balanced"))),
        ("attacks_structured_masking", _flatten_rows(load_attack_results("masking", "structured"))),
        ("attacks_llm_masking", _flatten_rows(load_attack_results("masking", "llm"))),
    ]
    exported: Dict[str, str] = {}
    for name, rows in datasets:
        if not rows:
            continue
        path = base_dir / f"{name}.parquet"
        _pyarrow_writer(rows, path)
        exported[name] = str(path)

    manifest = load_batch_manifest(batch)
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts.update({f"parquet_{name}": path for name, path in exported.items()})
    manifest["artifacts"] = artifacts
    save_batch_manifest(batch, manifest)
    return exported


def run_export_parquet_command(batch: str) -> None:
    export_parquet_batch(batch)
