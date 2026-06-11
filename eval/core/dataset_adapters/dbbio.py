from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from eval.core.dataset_adapters.base import DatasetAdapter, DatasetEvaluationContext, DatasetRunRequest
from eval.core.io import read_jsonl_sample


class DbBioDatasetAdapter(DatasetAdapter):
    def protocol_metadata(
        self,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
    ) -> Dict[str, Any]:
        path = Path(context.root) / "eval" / "datasets" / "DB-bio" / "test.jsonl"
        records = read_jsonl_sample(path, limit=200)
        fields = ("label", "l1", "l2", "l3")
        present = sorted({field for record in records for field in fields if record.get(field) is not None})
        n_labeled = sum(1 for record in records if any(record.get(field) is not None for field in fields))
        return {
            "name": "DB-bio/RUPTA",
            "protocol": "identity_leakage_plus_utility_proxy",
            "annotation_status": "person_name_value_search",
            "source_path": str(path),
            "n_labeled_documents": n_labeled,
            "label_fields": present,
            "warning": (
                "Utility preservation is reported as proxy metadata until a classifier/reference scorer "
                "is configured."
            ),
        }
