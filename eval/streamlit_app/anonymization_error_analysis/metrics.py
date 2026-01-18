from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .spans import calculate_overlap
except ImportError:
    # Fallback if running as script or different context
    def calculate_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        start1, end1 = span1
        start2, end2 = span2
        return max(0, min(end1, end2) - max(start1, start2)) > 0


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


def compute_label_metrics(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute Precision, Recall, F1 per entity label.
    Matching strategy: Overlap > 0 and Label exact match.
    """
    # structure: label -> {tp: int, fp: int, fn: int}
    counts: Dict[str, Dict[str, int]] = {}

    for doc in data:
        # Standardize format: list of [start, end, label]
        gts = doc.get("ground_truth", [])
        preds = doc.get("predictions", [])

        # Normalize to tuples (start, end, label)
        gts = [tuple(x) if isinstance(x, list) else x for x in gts]
        preds = [tuple(x) if isinstance(x, list) else x for x in preds]

        # Organize by label for this doc to simplify logic
        # But we need to match individual spans.

        # Track which preds are used to avoid double counting if needed?
        # Usually for strict metrics one-to-one mapping is preferred,
        # but for simple "Entity exists" metrics:
        # TP: GT span matches ANY Pred span with same label
        # FN: GT span matches NO Pred span with same label
        # FP: Pred span matches NO GT span with same label

        # Note: This ignores "one pred matching two GTs" edge cases,
        # but is standard for loose named entity evaluation.

        matched_preds_indices = set()

        # Check GTs (TPs and FNs)
        for gt in gts:
            if len(gt) < 3:
                continue
            gt_start, gt_end, gt_label = gt[0], gt[1], gt[2]

            if gt_label not in counts:
                counts[gt_label] = {"tp": 0, "fp": 0, "fn": 0}

            found_match = False
            for idx, pred in enumerate(preds):
                if len(pred) < 3:
                    continue
                p_start, p_end, p_label = pred[0], pred[1], pred[2]

                if p_label == gt_label:
                    if calculate_overlap((gt_start, gt_end), (p_start, p_end)):
                        found_match = True
                        matched_preds_indices.add(idx)
                        # We break here: one GT satisfied by at least one Pred is a TP
                        break

            if found_match:
                counts[gt_label]["tp"] += 1
            else:
                counts[gt_label]["fn"] += 1

        # Check Preds (FPs)
        for idx, pred in enumerate(preds):
            if len(pred) < 3:
                continue
            _, _, p_label = pred[0], pred[1], pred[2]

            if p_label not in counts:
                counts[p_label] = {"tp": 0, "fp": 0, "fn": 0}

            if idx not in matched_preds_indices:
                # Need to verify if it was matched by ANY GT (bi-directional check logic above only checked if GT found a Pred)
                # But wait: matched_preds_indices stores indices of preds that successfully matched a GT.
                # So if idx is not in there, it matched NO GT.
                counts[p_label]["fp"] += 1

    # Calculate metrics
    metrics = {}
    for label, c in counts.items():
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    return metrics
