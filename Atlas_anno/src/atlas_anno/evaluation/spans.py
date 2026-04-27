from __future__ import annotations

from typing import Dict, List, Set, Tuple

from atlas_anno.schemas import AnnotationSpan, DocumentRecord


SpanKey = Tuple[int, int, str]


def _span_set(spans: List[AnnotationSpan]) -> Set[SpanKey]:
    return {(span.start, span.end, span.label) for span in spans}


def evaluate_span_metrics(documents: List[DocumentRecord]) -> Dict[str, object]:
    details = []
    total_tp = total_fp = total_fn = 0
    for document in documents:
        gold_rows = document.metadata.get("gold_annotations", {}).get("spans", [])
        gold = {(row["start"], row["end"], row["label"]) for row in gold_rows}
        predicted = _span_set(document.annotations.spans)
        tp = len(gold & predicted)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        details.append(
            {
                "doc_id": document.doc_id,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "predicted_spans": len(predicted),
                "gold_spans": len(gold),
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "summary": {
            "documents": len(documents),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "details": details,
    }

