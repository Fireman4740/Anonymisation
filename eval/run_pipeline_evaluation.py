from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.core.arc_compat import (
    CandidateSpec,
    build_pipegraph_runtime_config,
    compute_primary_metric,
    load_candidate,
)
from eval.core.bootstrap import load_pipegraph, project_root
from eval.core.datasets import get_allowed_labels, load_benchmark_docs, normalize_dataset_key
from eval.core.dataset_adapters import DatasetAdapter, DatasetEvaluationContext, DatasetRunRequest
from eval.core.io import json_default, safe_key, utc_run_id, write_json, write_jsonl
from eval.core.reporting import aggregate_document_metrics
from eval.core.pipeline import build_report
from eval.core.loaders.ratbench import load_ratbench_profiles
from eval.registry import get_registry
from eval.run_store import utc_now_iso


def _default_datasets() -> Tuple[str, ...]:
    return tuple(get_registry().list())


DEFAULT_DATASETS = _default_datasets()
AVAILABLE_DATASETS = DEFAULT_DATASETS


def _default_output_dir(root: str) -> str:
    return os.path.join(root, "artifacts", "eval-runs", utc_run_id())


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


def _arg_list(args: argparse.Namespace, name: str, default: Sequence[Any]) -> List[Any]:
    value = getattr(args, name, None)
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _runtime_config(args: argparse.Namespace, candidate: CandidateSpec, request: DatasetRunRequest) -> Dict[str, Any]:
    runtime_config = build_pipegraph_runtime_config(
        candidate,
        dataset_key=request.dataset,
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
    return runtime_config


def _evaluate_one_dataset(
    *,
    args: argparse.Namespace,
    context: DatasetEvaluationContext,
    adapter: DatasetAdapter,
    request: DatasetRunRequest,
    candidate: CandidateSpec,
) -> Dict[str, Any]:
    runtime_config = _runtime_config(args, candidate, request)
    docs, dataset_name = adapter.load(request, context)
    if not docs:
        raise ValueError(f"No documents loaded for {request.dataset_key}")

    allowed_labels = get_allowed_labels(
        request.dataset,
        profile=str(runtime_config.get("eval_profile") or "auto"),
    )
    started = time.time()
    report = build_report(
        docs,
        context.pipeline,
        context.create_initial_state,
        config=runtime_config,
        max_workers=getattr(args, "doc_workers", None),
        allowed_labels=allowed_labels,
    )
    elapsed_s = time.time() - started

    enrichment = adapter.enrich_report(report, request, context, runtime_config)
    aggregate = aggregate_document_metrics(report)
    protocol = adapter.protocol_metadata(request, context)
    axes, score_details = adapter.build_axes(
        report,
        aggregate,
        request,
        context,
        runtime_config,
        elapsed_s,
        enrichment,
    )

    dataset_dir = context.output_dir / "datasets" / safe_key(request.dataset_key)
    docs_path = dataset_dir / "documents.jsonl"
    metrics_path = dataset_dir / "metrics.json"
    write_jsonl(docs_path, report)

    result = {
        "status": "ok",
        "dataset_key": request.dataset_key,
        "dataset": request.dataset,
        "dataset_name": dataset_name,
        "protocol": protocol,
        "n_documents": len(report),
        "elapsed_s": round(elapsed_s, 3),
        "metrics": aggregate,
        "axes": axes,
        "documents_path": str(docs_path),
        **score_details,
    }
    write_json(metrics_path, result)
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


def _expand_requests(args: argparse.Namespace, datasets: Sequence[str]) -> tuple[list[tuple[DatasetAdapter, DatasetRunRequest]], list[dict[str, Any]]]:
    registry = get_registry()
    requests: list[tuple[DatasetAdapter, DatasetRunRequest]] = []
    errors: list[dict[str, Any]] = []
    for raw_dataset in datasets:
        dataset = normalize_dataset_key(str(raw_dataset))
        try:
            adapter = registry.get(dataset)
        except KeyError as exc:
            errors.append({"dataset": dataset, "error": str(exc)})
            continue
        for request in adapter.expand_run_requests(args):
            requests.append((adapter, request))
    return requests, errors


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
    write_json(output_dir / "run_config.json", run_config)
    write_json(
        output_dir / "candidate_effective_config.json",
        {
            "candidate_id": candidate.candidate_id,
            "config": candidate.config,
            "ignored_keys": list(candidate.ignored_keys),
            "warnings": list(candidate.warnings),
        },
    )

    dataset_results: Dict[str, Dict[str, Any]] = {}
    requests, errors = _expand_requests(args, datasets)
    context = DatasetEvaluationContext(
        args=args,
        root=root,
        output_dir=output_dir,
        pipeline=pipeline,
        create_initial_state=create_initial_state,
        load_benchmark_docs=load_benchmark_docs,
        load_ratbench_profiles=load_ratbench_profiles,
        openrouter_key_available=_openrouter_key_available,
    )

    for error in errors:
        dataset = str(error.get("dataset", "unknown"))
        dataset_results[dataset] = {
            "status": "error",
            "dataset_key": dataset,
            "score": 0.0,
            "score_status": "error",
            "error": error.get("error"),
        }

    for adapter, request in requests:
        try:
            dataset_results[request.dataset_key] = _evaluate_one_dataset(
                args=args,
                context=context,
                adapter=adapter,
                request=request,
                candidate=candidate,
            )
        except Exception as exc:
            errors.append({"dataset": request.dataset_key, "error": str(exc)})
            dataset_results[request.dataset_key] = {
                "status": "error",
                "dataset_key": request.dataset_key,
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
    write_json(output_dir / "summary.json", payload)
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "summary.md").write_text(_make_summary_markdown(payload), encoding="utf-8")
    return payload


def run_arc_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    return run_evaluation(args)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    dataset_choices = tuple(get_registry().list())
    parser = argparse.ArgumentParser(description="Official PipeGraph pipeline evaluation runner.")
    parser.add_argument("--candidate", default=None, help="Optional candidate JSON path")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=dataset_choices)
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
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default))
    return 0 if payload.get("status") in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
