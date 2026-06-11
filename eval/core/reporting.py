from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple  # noqa: F401

from eval.core.config import normalize_runtime_config
from eval.core.metrics import fbeta
from eval.run_store import utc_now_iso

DocumentReport = List[Dict[str, Any]]


def aggregate_document_metrics(report: DocumentReport) -> Dict[str, Any]:
    if not report:
        return {
            "n_documents": 0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_exact_label_recall": 0.0,
            "macro_f2": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_exact_label_recall": 0.0,
            "micro_f2": 0.0,
            "total_predictions": 0,
            "total_ground_truth": 0,
            "total_leaks": 0,
        }

    n_documents = len(report)
    macro_precision = sum(float(doc.get("precision", 0.0)) for doc in report) / n_documents
    macro_recall = sum(float(doc.get("recall", 0.0)) for doc in report) / n_documents
    macro_exact_label_recall = (
        sum(float(doc.get("exact_label_recall", 0.0)) for doc in report) / n_documents
    )
    macro_f2 = sum(float(doc.get("f2", 0.0)) for doc in report) / n_documents

    precision_tp = sum(int(doc.get("precision_tp", 0)) for doc in report)
    precision_fp = sum(int(doc.get("precision_fp", 0)) for doc in report)
    recall_tp = sum(int(doc.get("recall_tp", 0)) for doc in report)
    recall_fn = sum(int(doc.get("recall_fn", 0)) for doc in report)
    exact_recall_tp = sum(int(doc.get("exact_recall_tp", 0)) for doc in report)
    exact_recall_fn = sum(int(doc.get("exact_recall_fn", 0)) for doc in report)

    micro_precision = (
        precision_tp / (precision_tp + precision_fp)
        if (precision_tp + precision_fp) > 0
        else 0.0
    )
    micro_recall = (
        recall_tp / (recall_tp + recall_fn)
        if (recall_tp + recall_fn) > 0
        else 0.0
    )
    micro_exact_label_recall = (
        exact_recall_tp / (exact_recall_tp + exact_recall_fn)
        if (exact_recall_tp + exact_recall_fn) > 0
        else 0.0
    )
    micro_f2 = fbeta(micro_precision, micro_recall, beta=2.0)

    bleu_scores = [float(doc["bleu_score"]) for doc in report if "bleu_score" in doc]
    macro_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None

    # Strict-match aggregations (present when evaluate_spans() extended)
    def _macro(key: str) -> Optional[float]:
        vals = [float(doc[key]) for doc in report if key in doc]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _micro_strict(tp_key: str, fp_fn_key: str) -> Optional[float]:
        tp = sum(int(doc.get(tp_key, 0)) for doc in report if tp_key in doc)
        denom = sum(int(doc.get(tp_key, 0)) + int(doc.get(fp_fn_key, 0)) for doc in report if tp_key in doc)
        return round(tp / denom, 4) if denom > 0 else None

    result: Dict[str, Any] = {
        "n_documents": n_documents,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_exact_label_recall": round(macro_exact_label_recall, 4),
        "macro_f2": round(macro_f2, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_exact_label_recall": round(micro_exact_label_recall, 4),
        "micro_f2": round(micro_f2, 4),
        "total_predictions": sum(int(doc.get("pred_count", 0)) for doc in report),
        "total_ground_truth": sum(int(doc.get("truth_count", 0)) for doc in report),
        "total_leaks": sum(int(doc.get("leaks_count", 0)) for doc in report),
        "macro_bleu": round(macro_bleu, 4) if macro_bleu is not None else None,
    }

    # Strict-match metrics (only when available in the report)
    if any("strict_precision" in doc for doc in report):
        result["macro_strict_precision"] = _macro("strict_precision")
        result["macro_strict_recall"] = _macro("strict_recall")
        result["macro_strict_f1"] = _macro("strict_f1")
        result["macro_strict_f2"] = _macro("strict_f2")
        result["micro_strict_precision"] = _micro_strict("strict_precision_tp", "strict_precision_fp")
        result["micro_strict_recall"] = _micro_strict("strict_recall_tp", "strict_recall_fn")
        # Error classification totals
        for err_key in ("missed", "spurious", "boundary_error", "type_error"):
            result[f"total_{err_key}"] = sum(
                int((doc.get("error_classification") or {}).get(err_key, 0)) for doc in report
            )

    return result


def build_report_meta(
    *,
    dataset_name: str,
    dataset_path: Optional[str] = None,
    limit: Optional[int] = None,
    config: Optional[Mapping[str, Any]] = None,
    pipeline: str = "pipegraph",
    run_name: Optional[str] = None,
    extras: Optional[Mapping[str, Any]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    dataset_meta: Dict[str, Any] = {"name": dataset_name}
    if dataset_path is not None:
        dataset_meta["path"] = dataset_path

    meta: Dict[str, Any] = {
        "created_at": created_at or utc_now_iso(),
        "pipeline": pipeline,
        "run_name": run_name,
        "dataset": dataset_meta,
        "limit": limit,
        "config": normalize_runtime_config(config),
    }
    if extras:
        meta.update(dict(extras))
    return meta


def build_report_payload(*, meta: Mapping[str, Any], data: DocumentReport) -> Dict[str, Any]:
    return {"meta": dict(meta), "data": list(data)}


def save_report_payload(path: str, *, meta: Mapping[str, Any], data: DocumentReport) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = build_report_payload(meta=meta, data=data)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
    return path


def load_report_payload(path: str) -> Tuple[Dict[str, Any], DocumentReport]:
    with open(path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if isinstance(payload, dict) and isinstance(payload.get("meta"), dict) and isinstance(payload.get("data"), list):
        return payload["meta"], payload["data"]

    if isinstance(payload, dict) and isinstance(payload.get("details"), list):
        meta = {key: value for key, value in payload.items() if key != "details"}
        return meta, payload["details"]

    if isinstance(payload, list):
        return {}, payload

    raise ValueError("Unsupported evaluation report format")
