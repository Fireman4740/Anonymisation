from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def compute_dataset_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_docs = len(data)
    if total_docs == 0:
        return {
            "total_docs": 0,
            "avg_prec": 0.0,
            "avg_rec": 0.0,
            "avg_f2": 0.0,
            "leaky_docs": 0,
        }

    return {
        "total_docs": total_docs,
        "avg_prec": sum(d.get("precision", 0.0) for d in data) / total_docs,
        "avg_rec": sum(d.get("recall", 0.0) for d in data) / total_docs,
        "avg_f2": sum(d.get("f2", 0.0) for d in data) / total_docs,
        "leaky_docs": sum(1 for d in data if d.get("leaks_count", 0) > 0),
    }


def compute_label_metrics(report: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute precision, recall, f1 per entity label across all documents.

    Args:
        report: List of document results, each containing 'entities' with
                'tp', 'fp', 'fn', 'matched_labels' info.

    Returns:
        Dict mapping label -> {precision, recall, f1, support}
    """
    # Aggregate TP, FP, FN per label
    label_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    )

    for doc in report:
        entities = doc.get("entities", [])
        for ent in entities:
            label = ent.get("label", ent.get("entity_type", "UNKNOWN"))
            label_stats[label]["support"] += 1

        # Document-level TP/FP/FN for each label
        tp_by_label = doc.get("tp_by_label", {})
        fp_by_label = doc.get("fp_by_label", {})
        fn_by_label = doc.get("fn_by_label", {})

        for label, count in tp_by_label.items():
            label_stats[label]["tp"] += count
        for label, count in fp_by_label.items():
            label_stats[label]["fp"] += count
        for label, count in fn_by_label.items():
            label_stats[label]["fn"] += count

    # Compute precision, recall, f1 per label
    result: Dict[str, Dict[str, Any]] = {}
    for label, stats in label_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        support = stats["support"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return result
