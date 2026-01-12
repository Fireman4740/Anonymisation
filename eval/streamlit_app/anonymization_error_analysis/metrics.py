from __future__ import annotations

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
