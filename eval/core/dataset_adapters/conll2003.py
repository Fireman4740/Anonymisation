from __future__ import annotations

from typing import Any, Dict

from eval.core.dataset_adapters.base import DatasetAdapter, DatasetEvaluationContext, DatasetRunRequest


class Conll2003DatasetAdapter(DatasetAdapter):
    def protocol_metadata(
        self,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
    ) -> Dict[str, Any]:
        return {
            "name": "CleanCoNLL/CoNLL-2003",
            "protocol": "ner_sanity",
            "annotation_status": "token_bio_to_char_offsets",
            "warning": "CoNLL-2003 is reported as NER sanity only, not as an anonymization benchmark.",
        }
