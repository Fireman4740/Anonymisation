from __future__ import annotations

from typing import Any, Dict

from eval.core.dataset_adapters.base import DatasetAdapter, DatasetEvaluationContext, DatasetRunRequest


class PersonalRedditDatasetAdapter(DatasetAdapter):
    def protocol_metadata(
        self,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
    ) -> Dict[str, Any]:
        return {
            "name": "PersonalReddit",
            "protocol": "synthetic_personal_attribute_leakage",
            "annotation_status": "attribute_value_search",
            "warning": "PersonalReddit spans are derived by matching synthetic profile values in responses.",
        }
