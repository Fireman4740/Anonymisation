from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from arc_pipegraph.objective import compute_primary_metric
from arc_pipegraph.pipeline_adapter import (
    CandidateSpec,
    build_pipegraph_runtime_config,
    load_candidate,
)
from eval.core.bootstrap import load_pipegraph, project_root
from eval.core.datasets import get_allowed_labels, load_benchmark_docs, normalize_dataset_key
from eval.core.metrics import (
    gold_text_leakage,
    runtime_metrics,
    score_dataset_axes,
    span_detection_axes,
    utility_preservation_stub,
)
from eval.core.ratbench import (
    compute_leak_summary,
    direct_id_detection_rate,
    metrics_by_difficulty,
    metrics_by_scenario,
)
from eval.core.reporting import aggregate_document_metrics
from eval.pipegraph_eval_local import build_report
from eval.ratbench_loader import evaluate_text_leaks, get_ratbench_metadata, load_ratbench_profiles
from eval.run_store import utc_now_iso

AVAILABLE_DATASETS = ("tab", "dbbio", "anonymization", "ratbench", "conll2003")
DEFAULT_DATASETS = ("tab", "dbbio", "ratbench", "conll2003", "anonymization")


def _json_default(value: Any) -> str:
    return str(value)


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "dataset"


def _utc_run_id(prefix: str = "eval") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _default_output_dir(root: str) -> str:
    return os.path.join(root, "artifacts", "eval-runs", _utc_run_id())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default))
            handle.write("\n")


def _read_jsonl_sample(path: Path, limit: int = 25) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                records.append(raw)
    return records


def _inspect_tab_protocol(path: Path) -> Dict[str, Any]:
    records = _read_jsonl_sample(path)
    has_offsets = False
    has_types = False
    has_masked_entities = False

    for record in records:
        annotations = record.get("annotations") or record.get("entities") or []
        if isinstance(annotations, list):
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                has_offsets = has_offsets or ("start" in annotation and "end" in annotation)
                has_types = has_types or bool(annotation.get("type") or annotation.get("label"))
        meta = record.get("meta") or {}
        if isinstance(meta, dict) and meta.get("masked_entities"):
            has_masked_entities = True

    if has_offsets and has_types:
        status = "official_offsets"
        warning = None
    elif has_masked_entities:
        status = "converted_no_offsets"
        warning = "TAB local JSONL exposes masked entity strings but not official offsets/types."
    else:
        status = "unknown_schema"
        warning = "TAB schema could not be verified from the local JSONL sample."

    return {
        "name": "TAB",
        "protocol": "legal_text_anonymization",
        "annotation_status": status,
        "source_path": str(path),
        "warning": warning,
    }


def _inspect_dbbio_utility(path: Path) -> Dict[str, Any]:
    records = _read_jsonl_sample(path, limit=200)
    fields = ("label", "l1", "l2", "l3")
    present = sorted({field for record in records for field in fields if record.get(field) is not None})
    n_labeled = sum(1 for record in records if any(record.get(field) is not None for field in fields))
    return {
        "name": "DB-bio/RUPTA",
        "protocol": "identity_leakage_plus_utility_proxy",
        "annotation_status": "person_name_value_search",
        "source_path": str(path),
        "n_labeled_documents": n_labeled,
        "label_fields": present,
        "warning": (
            "Utility preservation is reported as proxy metadata until a classifier/reference scorer "
            "is configured."
        ),
    }


def _dataset_protocol_metadata(
    *,
    dataset: str,
    root: str,
    split: str,
    language: str,
    level: Optional[int],
) -> Dict[str, Any]:
    if dataset == "tab":
        return _inspect_tab_protocol(Path(root) / "eval" / "datasets" / "TAB" / f"{split}.jsonl")
    if dataset == "dbbio":
        return _inspect_dbbio_utility(Path(root) / "eval" / "datasets" / "DB-bio" / "test.jsonl")
    if dataset == "ratbench":
        return {
            "name": "RAT-Bench",
            "protocol": "profile_value_search_plus_reidentification_risk",
            "annotation_status": "value_search_no_offsets",
            "language": language,
            "level": level,
            "warning": "RAT-Bench profiles provide attribute values; char offsets are derived by value matching.",
        }
    if dataset == "conll2003":
        return {
            "name": "CleanCoNLL/CoNLL-2003",
            "protocol": "ner_sanity",
            "annotation_status": "token_bio_to_char_offsets",
            "warning": "CoNLL-2003 is reported as NER sanity only, not as an anonymization benchmark.",
        }
    if dataset == "anonymization":
        return {
            "name": "local_synthetic_anonymization",
            "protocol": "internal_regression",
            "annotation_status": "local_exact_offsets",
            "warning": None,
        }
    return {"name": dataset, "protocol": "unknown", "annotation_status": "unknown", "warning": None}


def _dotenv_has_openrouter_key(root: str) -> bool:
    dotenv_path = Path(root) / ".env"
    if not dotenv_path.exists():
        return False
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "OPENROUTER_API_KEY" and value.strip().strip("\"'"):
                return True
    except OSError:
        return False
    return False


def _openrouter_key_available(root: str) -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip()) or _dotenv_has_openrouter_key(root)


def _candidate_from_args(args: argparse.Namespace) -> CandidateSpec:
    candidate_path = getattr(args, "candidate", None)
    if candidate_path:
        candidate = load_candidate(str(candidate_path))
    else:
        candidate = CandidateSpec(candidate_id="default", config={})

    config = dict(candidate.config)
    if getattr(args, "llm_provider", None):
        config["llm_provider"] = str(args.llm_provider)
    if getattr(args, "llm_model", None):
        config["llm_model"] = str(args.llm_model)

    return CandidateSpec(
        candidate_id=candidate.candidate_id,
        config=config,
        ignored_keys=candidate.ignored_keys,
        warnings=candidate.warnings,
    )


def _load_pipeline() -> Tuple[Any, Any]:
    create_pipeline_graph, create_initial_state = load_pipegraph()
    return create_pipeline_graph(), create_initial_state


def _attach_ratbench_metadata(report: List[Dict[str, Any]], profiles: List[Dict[str, Any]]) -> None:
    profile_by_doc_id = {
        f"ratbench_{profile.get('id', '')}_L{profile.get('difficulty', '?')}": profile
        for profile in profiles
    }
    for document in report:
        profile = profile_by_doc_id.get(str(document.get("doc_id", "")))
        if not profile:
            continue
        original_text = str(document.get("full_text") or "")
        anonymized_text = str(document.get("anonymized_text") or original_text)
        document["ratbench_metadata"] = get_ratbench_metadata(profile)
        document["text_leak_analysis"] = evaluate_text_leaks(
            original_text=original_text,
            anonymized_text=anonymized_text,
            profile=profile,
        )


def _serialize_risk_details(details: Any) -> List[Dict[str, Any]]:
    if details is None:
        return []
    if hasattr(details, "empty") and hasattr(details, "to_dict"):
        if bool(getattr(details, "empty")):
            return []
        return list(details.to_dict(orient="records"))
    if isinstance(details, list):
        return [dict(item) for item in details if isinstance(item, Mapping)]
    return []


def _evaluate_ratbench_risk_axis(
    *,
    args: argparse.Namespace,
    root: str,
    pipeline: Any,
    create_initial_state: Any,
    profiles: List[Dict[str, Any]],
    runtime_config: Mapping[str, Any],
    report: List[Dict[str, Any]],
    output_dir: Path,
    dataset_key: str,
) -> Dict[str, Any]:
    if getattr(args, "skip_risk", False):
        return {
            "status": "risk_skipped",
            "protocol": "ratbench_reidentification_risk",
            "reason": "disabled_by_cli",
        }

    if not _openrouter_key_available(root):
        message = "OPENROUTER_API_KEY not set; RAT-Bench LLM attacker skipped."
        if getattr(args, "require_risk", False):
            raise RuntimeError(message)
        return {
            "status": "risk_degraded",
            "protocol": "ratbench_reidentification_risk",
            "provider": "openrouter",
            "error": message,
        }

    from eval.cli.evaluate_ratbench_risk import evaluate_ratbench_risk_from_pipeline

    payload = evaluate_ratbench_risk_from_pipeline(
        pipeline,
        create_initial_state,
        profiles,
        config=dict(runtime_config),
        limit=getattr(args, "risk_limit", None) or getattr(args, "limit", None),
        report=report,
    )
    if payload.get("error"):
        if getattr(args, "require_risk", False):
            raise RuntimeError(str(payload["error"]))
        return {
            "status": "risk_degraded",
            "protocol": "ratbench_reidentification_risk",
            "provider": "openrouter",
            "error": payload["error"],
        }

    metrics = dict(payload.get("metrics") or {})
    details = _serialize_risk_details(payload.get("detailed_results"))
    details_path = output_dir / "ratbench" / f"{_safe_key(dataset_key)}_risk_details.jsonl"
    if details:
        _write_jsonl(details_path, details)

    return {
        "status": "risk_full",
        "protocol": "ratbench_reidentification_risk",
        "provider": "openrouter",
        "attacker_model_policy": "eval.cli.evaluate_ratbench_risk._OPENROUTER_MODELS",
        "details_path": str(details_path) if details else None,
        **metrics,
    }


def _evaluate_one_dataset(
    *,
    args: argparse.Namespace,
    root: str,
    output_dir: Path,
    candidate: CandidateSpec,
    pipeline: Any,
    create_initial_state: Any,
    dataset: str,
    split: str,
    language: str,
    level: Optional[int],
) -> Dict[str, Any]:
    dataset_key = f"ratbench/{language}/L{level}" if dataset == "ratbench" and level else dataset
    runtime_config = build_pipegraph_runtime_config(
        candidate,
        dataset_key=dataset,
        profile=str(getattr(args, "profile", "auto")),
        eval_mode=str(getattr(args, "eval_mode", "both")),
        masking_mode=str(getattr(args, "masking_mode", "benchmark")),
    )
    if getattr(args, "no_llm", False):
        runtime_config.update(
            {
                "disable_llm": True,
                "llm_detection": False,
                "llm_verification": False,
                "llm_audit": False,
                "llm_paraphrase": False,
                "rupta_enabled": False,
            }
        )
    docs, dataset_name = load_benchmark_docs(
        dataset=dataset,
        project_root=root,
        limit=getattr(args, "limit", None),
        level=level,
        language=language,
        split=split,
    )
    if not docs:
        raise ValueError(f"No documents loaded for {dataset_key}")

    allowed_labels = get_allowed_labels(dataset, profile=str(runtime_config.get("eval_profile") or "auto"))
    started = time.time()
    report = build_report(
        docs,
        pipeline,
        create_initial_state,
        config=runtime_config,
        max_workers=getattr(args, "doc_workers", None),
        allowed_labels=allowed_labels,
    )
    elapsed_s = time.time() - started

    profiles: List[Dict[str, Any]] = []
    if dataset == "ratbench":
        profiles = load_ratbench_profiles(language=language, level=level, limit=getattr(args, "limit", None))
        _attach_ratbench_metadata(report, profiles)

    aggregate = aggregate_document_metrics(report)
    protocol = _dataset_protocol_metadata(
        dataset=dataset,
        root=root,
        split=split,
        language=language,
        level=level,
    )

    span_axis = span_detection_axes(report, aggregate)
    leakage_axis = gold_text_leakage(report)
    utility_axis = utility_preservation_stub(dataset, records_meta=protocol)
    runtime_axis = runtime_metrics(report, elapsed_s)
    risk_axis: Optional[Dict[str, Any]] = None

    if dataset == "ratbench":
        leak_summary = compute_leak_summary(report)
        leakage_axis["ratbench_profile_leakage"] = {
            **leak_summary,
            "avg_protection_rate": round(1.0 - float(leak_summary.get("avg_leak_rate", 0.0)), 4),
            "avg_direct_protection_rate": round(
                1.0 - float(leak_summary.get("avg_direct_leak_rate", 0.0)), 4
            ),
            "avg_indirect_protection_rate": round(
                1.0 - float(leak_summary.get("avg_indirect_leak_rate", 0.0)), 4
            ),
        }
        aggregate.update(leak_summary)
        aggregate["by_difficulty"] = metrics_by_difficulty(report)
        aggregate["by_scenario"] = metrics_by_scenario(report)
        aggregate["direct_id_detection_rates"] = direct_id_detection_rate(report)
        risk_axis = _evaluate_ratbench_risk_axis(
            args=args,
            root=root,
            pipeline=pipeline,
            create_initial_state=create_initial_state,
            profiles=profiles,
            runtime_config=runtime_config,
            report=report,
            output_dir=output_dir,
            dataset_key=dataset_key,
        )

    score_details = score_dataset_axes(
        dataset=dataset,
        span_detection=span_axis,
        anonymization_leakage=leakage_axis,
        runtime=runtime_axis,
        ratbench_reid_risk=risk_axis,
        utility_preservation=utility_axis,
    )

    dataset_dir = output_dir / "datasets" / _safe_key(dataset_key)
    docs_path = dataset_dir / "documents.jsonl"
    metrics_path = dataset_dir / "metrics.json"
    _write_jsonl(docs_path, report)

    axes: Dict[str, Any] = {
        "span_detection": span_axis,
        "anonymization_leakage": leakage_axis,
        "utility_preservation": utility_axis,
        "runtime": runtime_axis,
    }
    if risk_axis is not None:
        axes["ratbench_reid_risk"] = risk_axis

    result = {
        "status": "ok",
        "dataset_key": dataset_key,
        "dataset": dataset,
        "dataset_name": dataset_name,
        "protocol": protocol,
        "n_documents": len(report),
        "elapsed_s": round(elapsed_s, 3),
        "metrics": aggregate,
        "axes": axes,
        "documents_path": str(docs_path),
        **score_details,
    }
    _write_json(metrics_path, result)
    result["metrics_path"] = str(metrics_path)
    return result


def _make_summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# PipeGraph Pipeline Evaluation",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Candidate: `{payload.get('candidate_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Primary metric: `{payload.get('primary_metric')}` ({payload.get('primary_metric_status')})",
        "",
        "| Dataset | Protocol | Docs | Score | Score status | Span F2 | Gold leak rate | Risk |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for key, result in sorted((payload.get("datasets") or {}).items()):
        axes = result.get("axes") or {}
        span = ((axes.get("span_detection") or {}).get("relaxed_overlap") or {})
        leakage = axes.get("anonymization_leakage") or {}
        risk = axes.get("ratbench_reid_risk") or {}
        risk_value = risk.get("avg_risk", "")
        protocol = (result.get("protocol") or {}).get("protocol", "")
        lines.append(
            "| {key} | {protocol} | {docs} | {score:.4f} | {status} | {f2:.4f} | {leak:.4f} | {risk} |".format(
                key=key,
                protocol=protocol,
                docs=int(result.get("n_documents") or 0),
                score=float(result.get("score") or 0.0),
                status=result.get("score_status", ""),
                f2=float(span.get("f2") or 0.0),
                leak=float(leakage.get("gold_text_leak_rate") or 0.0),
                risk=risk_value,
            )
        )

    warnings = [
        (key, (result.get("protocol") or {}).get("warning"))
        for key, result in sorted((payload.get("datasets") or {}).items())
        if (result.get("protocol") or {}).get("warning")
    ]
    if warnings:
        lines.extend(["", "## Protocol Notes", ""])
        for key, warning in warnings:
            lines.append(f"- `{key}`: {warning}")
    lines.append("")
    return "\n".join(lines)


def _arg_list(args: argparse.Namespace, name: str, default: Sequence[Any]) -> List[Any]:
    value = getattr(args, name, None)
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    root = project_root()
    output_dir = Path(getattr(args, "output", None) or _default_output_dir(root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = _candidate_from_args(args)
    pipeline, create_initial_state = _load_pipeline()

    split = str(getattr(args, "split", "test"))
    datasets = [normalize_dataset_key(str(dataset)) for dataset in _arg_list(args, "datasets", DEFAULT_DATASETS)]
    languages = [str(item) for item in _arg_list(args, "ratbench_languages", [getattr(args, "language", "english")])]
    ratbench_levels = [int(item) for item in _arg_list(args, "ratbench_levels", [1])]

    run_config = {
        "created_at": utc_now_iso(),
        "candidate": getattr(args, "candidate", None),
        "candidate_id": candidate.candidate_id,
        "candidate_config": candidate.config,
        "candidate_ignored_keys": list(candidate.ignored_keys),
        "candidate_warnings": list(candidate.warnings),
        "datasets": datasets,
        "split": split,
        "limit": getattr(args, "limit", None),
        "ratbench_languages": languages,
        "ratbench_levels": ratbench_levels,
        "llm_provider": getattr(args, "llm_provider", None),
        "llm_attacker_provider": "openrouter",
        "require_risk": bool(getattr(args, "require_risk", False)),
        "skip_risk": bool(getattr(args, "skip_risk", False)),
        "profile": getattr(args, "profile", "auto"),
        "eval_mode": getattr(args, "eval_mode", "both"),
        "masking_mode": getattr(args, "masking_mode", "benchmark"),
        "doc_workers": getattr(args, "doc_workers", None),
        "no_llm": bool(getattr(args, "no_llm", False)),
    }
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(
        output_dir / "candidate_effective_config.json",
        {
            "candidate_id": candidate.candidate_id,
            "config": candidate.config,
            "ignored_keys": list(candidate.ignored_keys),
            "warnings": list(candidate.warnings),
        },
    )

    dataset_results: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []

    for dataset in datasets:
        if dataset not in AVAILABLE_DATASETS:
            errors.append({"dataset": dataset, "error": f"Unknown dataset: {dataset}"})
            dataset_results[dataset] = {"status": "error", "dataset_key": dataset, "score": 0.0}
            continue

        if dataset == "ratbench":
            for language in languages:
                for level in ratbench_levels:
                    key = f"ratbench/{language}/L{level}"
                    try:
                        dataset_results[key] = _evaluate_one_dataset(
                            args=args,
                            root=root,
                            output_dir=output_dir,
                            candidate=candidate,
                            pipeline=pipeline,
                            create_initial_state=create_initial_state,
                            dataset=dataset,
                            split=split,
                            language=language,
                            level=level,
                        )
                    except Exception as exc:
                        errors.append({"dataset": key, "error": str(exc)})
                        dataset_results[key] = {
                            "status": "error",
                            "dataset_key": key,
                            "score": 0.0,
                            "score_status": "error",
                            "error": str(exc),
                        }
            continue

        try:
            dataset_results[dataset] = _evaluate_one_dataset(
                args=args,
                root=root,
                output_dir=output_dir,
                candidate=candidate,
                pipeline=pipeline,
                create_initial_state=create_initial_state,
                dataset=dataset,
                split=split,
                language="english",
                level=None,
            )
        except Exception as exc:
            errors.append({"dataset": dataset, "error": str(exc)})
            dataset_results[dataset] = {
                "status": "error",
                "dataset_key": dataset,
                "score": 0.0,
                "score_status": "error",
                "error": str(exc),
            }

    primary_metric, aggregate = compute_primary_metric(dataset_results)
    ok_count = sum(1 for result in dataset_results.values() if result.get("status") == "ok")
    degraded_count = sum(1 for result in dataset_results.values() if result.get("score_status") == "degraded")
    status = "ok" if not errors else "partial"
    if ok_count == 0:
        status = "error"
    primary_metric_status = "degraded" if degraded_count else "full"
    if status == "error":
        primary_metric_status = "error"

    payload = {
        "run_id": output_dir.name,
        "created_at": run_config["created_at"],
        "status": status,
        "primary_metric": primary_metric,
        "primary_metric_status": primary_metric_status,
        "metric_direction": "maximize",
        "candidate_id": candidate.candidate_id,
        "output_dir": str(output_dir),
        "datasets": dataset_results,
        "aggregate": {
            **aggregate,
            "n_ok": ok_count,
            "n_error": len(errors),
            "n_degraded": degraded_count,
            "candidate_config": candidate.config,
            "ignored_candidate_keys": list(candidate.ignored_keys),
            "candidate_warnings": list(candidate.warnings),
        },
        "errors": errors,
    }

    manifest = {
        "run_id": payload["run_id"],
        "created_at": payload["created_at"],
        "output_dir": str(output_dir),
        "files": {
            "run_config": str(output_dir / "run_config.json"),
            "summary_json": str(output_dir / "summary.json"),
            "summary_md": str(output_dir / "summary.md"),
            "candidate_effective_config": str(output_dir / "candidate_effective_config.json"),
        },
        "datasets": {
            key: {
                "documents_path": value.get("documents_path"),
                "metrics_path": value.get("metrics_path"),
            }
            for key, value in dataset_results.items()
        },
    }
    _write_json(output_dir / "summary.json", payload)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "summary.md").write_text(_make_summary_markdown(payload), encoding="utf-8")
    return payload


def run_arc_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    return run_evaluation(args)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official PipeGraph pipeline evaluation runner.")
    parser.add_argument("--candidate", default=None, help="Optional candidate JSON path")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=AVAILABLE_DATASETS)
    parser.add_argument("--split", choices=["test", "dev", "train"], default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default=None, help="Output directory. Defaults to artifacts/eval-runs/<timestamp>")
    parser.add_argument("--save-runs", action="store_true", help="Accepted for compatibility; official outputs are always saved.")
    parser.add_argument("--doc-workers", type=int, default=None)
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--eval-mode", choices=["canonical", "benchmark", "both"], default="both")
    parser.add_argument("--masking-mode", choices=["production", "benchmark"], default="benchmark")
    parser.add_argument("--language", choices=["english", "mandarin", "spanish"], default="english")
    parser.add_argument("--ratbench-languages", nargs="+", choices=["english", "mandarin", "spanish"], default=None)
    parser.add_argument("--ratbench-levels", nargs="+", type=int, choices=[1, 2, 3], default=[1])
    parser.add_argument("--llm-provider", default=None, help="Pipeline LLM provider override, e.g. openrouter or ollama")
    parser.add_argument("--llm-model", default=None, help="Pipeline LLM model override")
    parser.add_argument("--llm-attacker-model", default=None, help="Reserved for RAT-Bench attacker model logging")
    parser.add_argument("--no-llm", action="store_true", help="Disable pipeline LLM modules for local smoke tests")
    parser.add_argument("--skip-risk", "--no-risk", action="store_true", help="Skip RAT-Bench LLM re-identification risk")
    parser.add_argument("--require-risk", action="store_true", help="Fail RAT-Bench when native risk cannot run")
    parser.add_argument("--risk-limit", type=int, default=None, help="Optional RAT-Bench risk profile limit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    payload = run_evaluation(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default))
    return 0 if payload.get("status") in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
