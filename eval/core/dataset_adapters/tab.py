from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from eval.core.dataset_adapters.base import DatasetAdapter, DatasetEvaluationContext, DatasetRunRequest
from eval.core.io import read_jsonl_sample


class TabDatasetAdapter(DatasetAdapter):
    def protocol_metadata(
        self,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
    ) -> Dict[str, Any]:
        path = Path(context.root) / "eval" / "datasets" / "TAB" / f"{request.split}.jsonl"
        records = read_jsonl_sample(path)
        has_offsets = False
        has_types = False
        has_masked_entities = False
        for record in records:
            annotations = record.get("annotations") or record.get("entities") or []
            if isinstance(annotations, list):
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue
                    has_offsets = has_offsets or ("start" in annotation and "end" in annotation)
                    has_types = has_types or bool(annotation.get("type") or annotation.get("label"))
            meta = record.get("meta") or {}
            if isinstance(meta, dict) and meta.get("masked_entities"):
                has_masked_entities = True
        if has_offsets and has_types:
            status = "official_offsets"
            warning = None
        elif has_masked_entities:
            status = "converted_no_offsets"
            warning = "TAB local JSONL exposes masked entity strings but not official offsets/types."
        else:
            status = "unknown_schema"
            warning = "TAB schema could not be verified from the local JSONL sample."
        return {
            "name": "TAB",
            "protocol": "legal_text_anonymization",
            "annotation_status": status,
            "source_path": str(path),
            "warning": warning,
        }
