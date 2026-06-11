from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from eval.core.metrics import fbeta, spans_overlap
from eval.core.reporting import build_report_meta
from eval.core.pipeline import build_report
from eval.core.loaders.ratbench import DIRECT_ID_LABEL_MAP, evaluate_text_leaks, get_ratbench_metadata

Span = Tuple[int, int, str]
DocumentReport = List[Dict[str, Any]]


# R_succ threshold from RAT-Bench paper (arXiv 2602.12806, §4): R > 0.2 → re-identified
# Equivalent to k-anonymity k=5 (1/k = 0.2)
_RATBENCH_RISK_THRESHOLD = 0.2


def aggregate_ratbench_metrics(report: DocumentReport) -> Dict[str, Any]:
    if not report:
        return {
            "n_documents": 0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f2": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f2": 0.0,
            "total_predictions": 0,
            "total_ground_truth": 0,
            "total_leaks": 0,
            "r_succ_rate": None,
            "avg_risk": None,
            "macro_bleu": None,
        }

    n_documents = len(report)
    precision_tp = sum(int(doc.get("precision_tp", 0)) for doc in report)
    precision_fp = sum(int(doc.get("precision_fp", 0)) for doc in report)
    recall_tp = sum(int(doc.get("recall_tp", 0)) for doc in report)
    recall_fn = sum(int(doc.get("recall_fn", 0)) for doc in report)

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

    # R_succ: fraction of records with re-identification risk R > θ=0.2 (RAT-Bench paper §4)
    risk_docs = [doc for doc in report if "risk" in doc]
    r_succ_rate: Optional[float] = None
    avg_risk: Optional[float] = None
    if risk_docs:
        risks = [float(doc["risk"]) for doc in risk_docs]
        avg_risk = round(sum(risks) / len(risks), 4)
        r_succ_rate = round(sum(1 for r in risks if r > _RATBENCH_RISK_THRESHOLD) / len(risks), 4)

    bleu_docs = [float(doc["bleu_score"]) for doc in report if "bleu_score" in doc]
    macro_bleu = round(sum(bleu_docs) / len(bleu_docs), 4) if bleu_docs else None

    return {
        "n_documents": n_documents,
        "macro_precision": round(sum(float(doc.get("precision", 0.0)) for doc in report) / n_documents, 4),
        "macro_recall": round(sum(float(doc.get("recall", 0.0)) for doc in report) / n_documents, 4),
        "macro_f2": round(sum(float(doc.get("f2", 0.0)) for doc in report) / n_documents, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f2": round(fbeta(micro_precision, micro_recall, beta=2.0), 4),
        "total_predictions": sum(int(doc.get("pred_count", 0)) for doc in report),
        "total_ground_truth": sum(int(doc.get("truth_count", 0)) for doc in report),
        "total_leaks": sum(int(doc.get("leaks_count", 0)) for doc in report),
        # Primary RAT-Bench risk metrics (paper §4)
        "r_succ_rate": r_succ_rate,      # fraction where R > 0.2 (lower = better)
        "avg_risk": avg_risk,             # mean R across documents
        "macro_bleu": macro_bleu,         # utility: text preservation (higher = better)
    }


def metrics_by_difficulty(report: DocumentReport) -> Dict[int, Dict[str, Any]]:
    grouped: Dict[int, DocumentReport] = {}
    for document in report:
        difficulty = int(document.get("ratbench_metadata", {}).get("difficulty", 0))
        grouped.setdefault(difficulty, []).append(document)
    return {difficulty: aggregate_ratbench_metrics(documents) for difficulty, documents in sorted(grouped.items())}


def metrics_by_scenario(report: DocumentReport) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, DocumentReport] = {}
    for document in report:
        scenario = str(document.get("ratbench_metadata", {}).get("scenario", "unknown"))
        grouped.setdefault(scenario, []).append(document)
    return {scenario: aggregate_ratbench_metrics(documents) for scenario, documents in sorted(grouped.items())}


def direct_id_detection_rate(report: DocumentReport) -> Dict[str, Dict[str, float]]:
    type_stats: Dict[str, Dict[str, int]] = {}

    for document in report:
        metadata = document.get("ratbench_metadata", {})
        direct_types = metadata.get("direct_id_types", [])
        predictions = document.get("predictions", [])

        for identifier_type in direct_types:
            stats = type_stats.setdefault(identifier_type, {"total": 0, "detected": 0})
            stats["total"] += 1

            label = DIRECT_ID_LABEL_MAP.get(identifier_type, "SENSITIVE")
            ground_truth = [span for span in document.get("ground_truth", []) if span[2] == label]
            detected = any(
                spans_overlap(truth, prediction)
                for truth in ground_truth
                for prediction in predictions
            )
            if detected:
                stats["detected"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for identifier_type, stats in type_stats.items():
        total = stats["total"]
        detected = stats["detected"]
        result[identifier_type] = {
            "total": total,
            "detected": detected,
            "detection_rate": round(detected / total, 4) if total > 0 else 0.0,
        }
    return result


def compute_leak_summary(report: DocumentReport) -> Dict[str, float]:
    leak_entries = [entry.get("text_leak_analysis", {}) for entry in report if entry.get("text_leak_analysis")]
    if not leak_entries:
        return {
            "avg_leak_rate": 0.0,
            "avg_direct_leak_rate": 0.0,
            "avg_indirect_leak_rate": 0.0,
        }

    count = len(leak_entries)
    return {
        "avg_leak_rate": round(sum(float(leak.get("leak_rate", 0.0)) for leak in leak_entries) / count, 4),
        "avg_direct_leak_rate": round(sum(float(leak.get("direct_leak_rate", 0.0)) for leak in leak_entries) / count, 4),
        "avg_indirect_leak_rate": round(sum(float(leak.get("indirect_leak_rate", 0.0)) for leak in leak_entries) / count, 4),
    }


def build_ratbench_report(
    docs: List[Tuple[str, str, List[Span]]],
    profiles: List[Dict[str, Any]],
    pipeline: Any,
    create_initial_state: Any,
    config: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Any] = None,
    max_workers: Optional[int] = None,
) -> DocumentReport:
    report = build_report(
        docs,
        pipeline,
        create_initial_state,
        config=config,
        progress_cb=progress_cb,
        max_workers=max_workers,
    )

    profile_by_doc_id = {
        f"ratbench_{profile.get('id', '')}_L{profile.get('difficulty', '?')}": profile
        for profile in profiles
    }

    for document in report:
        doc_id = str(document.get("doc_id", ""))
        profile = profile_by_doc_id.get(doc_id)
        if not profile:
            continue

        document["ratbench_metadata"] = get_ratbench_metadata(profile)
        original_text = str(document.get("full_text", ""))
        anonymized_text = str(document.get("anonymized_text", original_text))
        document["text_leak_analysis"] = evaluate_text_leaks(
            original_text=original_text,
            anonymized_text=anonymized_text,
            profile=profile,
        )

    return report


def build_ratbench_result(
    *,
    report: DocumentReport,
    language: str,
    level: Optional[int],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    summary = aggregate_ratbench_metrics(report)
    by_difficulty = metrics_by_difficulty(report)
    by_scenario = metrics_by_scenario(report)
    direct_rates = direct_id_detection_rate(report)
    leak_summary = compute_leak_summary(report)

    return {
        "summary": {
            "language": language,
            "level": level,
            "config": config,
            **summary,
            **leak_summary,
        },
        "by_difficulty": {str(key): value for key, value in by_difficulty.items()},
        "by_scenario": by_scenario,
        "direct_id_detection_rates": direct_rates,
        "details": report,
    }


def build_ratbench_meta(
    *,
    report: DocumentReport,
    language: str,
    level: Optional[int],
    config: Dict[str, Any],
    limit: Optional[int],
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    result = build_ratbench_result(report=report, language=language, level=level, config=config)
    level_str = f"L{level}" if level else "all"
    meta = build_report_meta(
        dataset_name=f"RAT-Bench/{language}/{level_str}",
        limit=limit,
        config=config,
        run_name=run_name,
        extras={
            "aggregate_metrics": result["summary"],
            "by_difficulty": result["by_difficulty"],
            "by_scenario": result["by_scenario"],
            "direct_id_detection_rates": result["direct_id_detection_rates"],
        },
    )
    meta["dataset"] = {
        **meta["dataset"],
        "benchmark": "RAT-Bench",
        "source": "imperial-cpg/rat-bench",
        "language": language,
        "level": level,
    }
    return meta
