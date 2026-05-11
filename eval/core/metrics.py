from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Span = Tuple[int, int, str]
DocumentReport = List[Dict[str, Any]]


def clamp01(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    precision = clamp01(precision)
    recall = clamp01(recall)
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0
    return clamp01((1.0 + beta_sq) * precision * recall / denom)


def spans_overlap(left: Span, right: Span) -> bool:
    return max(0, min(int(left[1]), int(right[1])) - max(int(left[0]), int(right[0]))) > 0


def _span_tuple(value: Sequence[Any]) -> Optional[Span]:
    if len(value) < 3:
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None
    return (start, end, str(value[2]))


def normalize_spans(values: Iterable[Sequence[Any]]) -> List[Span]:
    spans: List[Span] = []
    seen: set[Span] = set()
    for raw in values:
        span = _span_tuple(raw)
        if span is None or span in seen:
            continue
        seen.add(span)
        spans.append(span)
    spans.sort(key=lambda item: (item[0], item[1], item[2]))
    return spans


def strict_span_metrics(
    ground_truth: Iterable[Sequence[Any]],
    predictions: Iterable[Sequence[Any]],
    *,
    typed: bool = True,
) -> Dict[str, Any]:
    gt = normalize_spans(ground_truth)
    pred = normalize_spans(predictions)
    if typed:
        gt_keys = {(s, e, label) for s, e, label in gt}
        pred_keys = {(s, e, label) for s, e, label in pred}
    else:
        gt_keys = {(s, e) for s, e, _ in gt}
        pred_keys = {(s, e) for s, e, _ in pred}

    tp = len(gt_keys & pred_keys)
    fp = len(pred_keys - gt_keys)
    fn = len(gt_keys - pred_keys)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f2": round(fbeta(precision, recall, beta=2.0), 4),
    }


def relaxed_overlap_metrics(
    ground_truth: Iterable[Sequence[Any]],
    predictions: Iterable[Sequence[Any]],
    *,
    typed: bool = False,
) -> Dict[str, Any]:
    gt = normalize_spans(ground_truth)
    pred = normalize_spans(predictions)
    gt_matched: set[int] = set()
    pred_matched: set[int] = set()

    for gt_index, gt_span in enumerate(gt):
        for pred_index, pred_span in enumerate(pred):
            if pred_index in pred_matched:
                continue
            if typed and gt_span[2] != pred_span[2]:
                continue
            if spans_overlap(gt_span, pred_span):
                gt_matched.add(gt_index)
                pred_matched.add(pred_index)
                break

    tp = len(pred_matched)
    fp = len(pred) - len(pred_matched)
    fn = len(gt) - len(gt_matched)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = len(gt_matched) / len(gt) if gt else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f2": round(fbeta(precision, recall, beta=2.0), 4),
    }


def label_metrics(report: DocumentReport, *, span_key: str = "benchmark") -> Dict[str, Dict[str, Any]]:
    label_tp: Dict[str, int] = {}
    label_fp: Dict[str, int] = {}
    label_fn: Dict[str, int] = {}
    label_exact_tp: Dict[str, int] = {}
    label_exact_fn: Dict[str, int] = {}

    for document in report:
        for label, count in (document.get("tp_by_label") or {}).items():
            label_tp[str(label)] = label_tp.get(str(label), 0) + int(count)
        for label, count in (document.get("fp_by_label") or {}).items():
            label_fp[str(label)] = label_fp.get(str(label), 0) + int(count)
        for label, count in (document.get("fn_by_label") or {}).items():
            label_fn[str(label)] = label_fn.get(str(label), 0) + int(count)
        for label, count in (document.get("exact_tp_by_label") or {}).items():
            label_exact_tp[str(label)] = label_exact_tp.get(str(label), 0) + int(count)
        for label, count in (document.get("exact_fn_by_label") or {}).items():
            label_exact_fn[str(label)] = label_exact_fn.get(str(label), 0) + int(count)

    labels = sorted(set(label_tp) | set(label_fp) | set(label_fn) | set(label_exact_tp) | set(label_exact_fn))
    result: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        tp = label_tp.get(label, 0)
        fp = label_fp.get(label, 0)
        fn = label_fn.get(label, 0)
        exact_tp = label_exact_tp.get(label, 0)
        exact_fn = label_exact_fn.get(label, 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        exact_recall = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) else 0.0
        result[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "exact_recall": round(exact_recall, 4),
            "f1": round(fbeta(precision, recall, beta=1.0), 4),
            "f2": round(fbeta(precision, recall, beta=2.0), 4),
            "support": tp + fn,
        }
    return result


def span_detection_axes(report: DocumentReport, aggregate: Mapping[str, Any]) -> Dict[str, Any]:
    all_gt: List[Span] = []
    all_pred: List[Span] = []
    all_canonical_gt: List[Span] = []
    all_canonical_pred: List[Span] = []

    for document in report:
        all_gt.extend(normalize_spans(document.get("benchmark_ground_truth") or document.get("ground_truth") or []))
        all_pred.extend(normalize_spans(document.get("benchmark_predictions") or document.get("predictions") or []))
        all_canonical_gt.extend(
            normalize_spans(document.get("canonical_ground_truth") or document.get("raw_ground_truth") or [])
        )
        all_canonical_pred.extend(
            normalize_spans(document.get("canonical_predictions") or document.get("raw_predictions") or [])
        )

    return {
        "status": "ok",
        "protocol": "span_detection",
        "relaxed_overlap": {
            "precision": float(aggregate.get("micro_precision", 0.0)),
            "recall": float(aggregate.get("micro_recall", 0.0)),
            "typed_recall": float(aggregate.get("micro_exact_label_recall", 0.0)),
            "f2": float(aggregate.get("micro_f2", 0.0)),
        },
        "strict_exact_typed": strict_span_metrics(all_gt, all_pred, typed=True),
        "strict_exact_untyped": strict_span_metrics(all_gt, all_pred, typed=False),
        "relaxed_overlap_typed": relaxed_overlap_metrics(all_gt, all_pred, typed=True),
        "canonical_strict_exact_typed": strict_span_metrics(all_canonical_gt, all_canonical_pred, typed=True),
        "label_metrics": label_metrics(report),
        "counts": {
            "n_documents": int(aggregate.get("n_documents") or len(report)),
            "total_predictions": int(aggregate.get("total_predictions") or 0),
            "total_ground_truth": int(aggregate.get("total_ground_truth") or 0),
            "total_missed_spans": int(aggregate.get("total_leaks") or 0),
        },
    }


def gold_text_leakage(report: DocumentReport) -> Dict[str, Any]:
    total = 0
    leaked = 0
    leaked_by_label: Dict[str, Dict[str, int]] = {}

    for document in report:
        original = str(document.get("full_text") or "")
        anonymized = str(document.get("anonymized_text") or "")
        anonymized_lower = anonymized.lower()
        for start, end, label in normalize_spans(document.get("benchmark_ground_truth") or document.get("ground_truth") or []):
            if end > len(original):
                continue
            value = original[start:end].strip()
            if not value:
                continue
            total += 1
            stats = leaked_by_label.setdefault(label, {"total": 0, "leaked": 0})
            stats["total"] += 1
            is_leaked = value.lower() in anonymized_lower
            if is_leaked:
                leaked += 1
                stats["leaked"] += 1

    per_label = {
        label: {
            "total": stats["total"],
            "leaked": stats["leaked"],
            "leak_rate": round(stats["leaked"] / stats["total"], 4) if stats["total"] else 0.0,
        }
        for label, stats in sorted(leaked_by_label.items())
    }
    leak_rate = leaked / total if total else 0.0
    return {
        "status": "ok",
        "protocol": "gold_value_string_leakage",
        "total_gold_values": total,
        "total_leaked_values": leaked,
        "gold_text_leak_rate": round(leak_rate, 4),
        "gold_text_protection_rate": round(1.0 - leak_rate, 4),
        "per_label": per_label,
    }


def runtime_metrics(report: DocumentReport, elapsed_s: float) -> Dict[str, Any]:
    n_documents = len(report)
    errors = [doc for doc in report if doc.get("error")]
    timeout_count = sum(1 for doc in errors if "timeout" in str(doc.get("error", "")).lower())
    seconds_per_doc = float(elapsed_s) / n_documents if n_documents else 0.0
    return {
        "status": "ok" if not errors else "partial",
        "elapsed_s": round(float(elapsed_s), 3),
        "n_documents": n_documents,
        "seconds_per_doc": round(seconds_per_doc, 4),
        "error_count": len(errors),
        "error_rate": round(len(errors) / n_documents, 4) if n_documents else 0.0,
        "timeout_count": timeout_count,
    }


def utility_preservation_stub(dataset: str, records_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if dataset == "dbbio":
        meta = dict(records_meta or {})
        n_labeled = int(meta.get("n_labeled_documents") or 0)
        return {
            "status": "proxy" if n_labeled else "not_available",
            "protocol": "dbbio_label_metadata_proxy",
            "score": None,
            "n_labeled_documents": n_labeled,
            "label_fields": list(meta.get("label_fields") or []),
            "note": "No classifier/reference utility scorer is configured; labels are exposed for a future utility model.",
        }
    return {
        "status": "not_applicable",
        "protocol": "not_applicable",
        "score": None,
    }


def score_dataset_axes(
    *,
    dataset: str,
    span_detection: Mapping[str, Any],
    anonymization_leakage: Mapping[str, Any],
    runtime: Mapping[str, Any],
    ratbench_reid_risk: Optional[Mapping[str, Any]] = None,
    utility_preservation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    relaxed = span_detection.get("relaxed_overlap") or {}
    span_score = clamp01(
        0.45 * clamp01(relaxed.get("f2"))
        + 0.35 * clamp01(relaxed.get("recall"))
        + 0.20 * clamp01(relaxed.get("typed_recall"))
    )
    leak_score = clamp01(anonymization_leakage.get("gold_text_protection_rate"))

    risk_status = None
    risk_score: Optional[float] = None
    if dataset == "ratbench":
        risk_status = str((ratbench_reid_risk or {}).get("status") or "risk_degraded")
        if ratbench_reid_risk and "avg_risk" in ratbench_reid_risk:
            risk_score = clamp01(1.0 - clamp01(ratbench_reid_risk.get("avg_risk")))
        else:
            risk_score = 0.0

    utility_score: Optional[float] = None
    if utility_preservation and utility_preservation.get("score") is not None:
        utility_score = clamp01(utility_preservation.get("score"))

    seconds_per_doc = max(0.0, float(runtime.get("seconds_per_doc") or 0.0))
    runtime_score = clamp01(1.0 / (1.0 + (seconds_per_doc / 12.0)))

    weights: Dict[str, float] = {
        "span_detection": 0.35,
        "anonymization_leakage": 0.30,
        "runtime": 0.10,
    }
    components: Dict[str, Optional[float]] = {
        "span_detection": round(span_score, 6),
        "anonymization_leakage": round(leak_score, 6),
        "runtime": round(runtime_score, 6),
    }

    if dataset == "ratbench":
        weights["ratbench_reid_risk"] = 0.25
        components["ratbench_reid_risk"] = round(float(risk_score or 0.0), 6)
    else:
        weights["span_detection"] += 0.15
        weights["anonymization_leakage"] += 0.10

    if utility_score is not None:
        weights["utility_preservation"] = 0.10
        components["utility_preservation"] = round(utility_score, 6)
    else:
        components["utility_preservation"] = None

    active_weights = {
        key: weight
        for key, weight in weights.items()
        if components.get(key) is not None
    }
    total_weight = sum(active_weights.values()) or 1.0
    normalized_weights = {key: weight / total_weight for key, weight in active_weights.items()}
    score = sum(float(components[key] or 0.0) * weight for key, weight in normalized_weights.items())

    score_status = "full"
    if dataset == "ratbench" and risk_status != "risk_full":
        score_status = "degraded"
    if utility_preservation and utility_preservation.get("status") == "proxy":
        score_status = "degraded" if score_status != "degraded" else score_status

    return {
        "score": round(clamp01(score), 6),
        "score_status": score_status,
        "score_components": components,
        "score_weights": {key: round(value, 6) for key, value in normalized_weights.items()},
    }
