from __future__ import annotations

from typing import Any, Dict

from eval.core.dataset_adapters.base import DatasetAdapter, DatasetEvaluationContext, DatasetRunRequest


class AnonymizationDatasetAdapter(DatasetAdapter):
    def protocol_metadata(
        self,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
    ) -> Dict[str, Any]:
        return {
            "name": "local_synthetic_anonymization",
            "protocol": "internal_regression",
            "annotation_status": "local_exact_offsets",
            "warning": None,
        }
