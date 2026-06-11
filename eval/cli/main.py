"""Unified evaluation CLI.

Single entry point for every evaluation workflow::

    python -m eval list-datasets
    python -m eval run --dataset tab --config configs/evaluation/no_llm.json
    python -m eval run --dataset all --config configs/evaluation/full_llm.json
    python -m eval ablation --dataset tab --config configs/evaluation/no_llm.json \
        --ablation-config configs/evaluation/ablations/default.json
    python -m eval compare --runs runs/evaluation/A runs/evaluation/B --output runs/comparison/
    python -m eval report --run runs/evaluation/A

All logic lives in ``eval.api`` (usable from the UI without subprocess);
this module only parses arguments and prints summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from eval.api import (
    EvaluationRunner,
    compare_runs,
    list_available_datasets,
    load_evaluation_report,
    load_metrics,
)


def _print_run_summary(payload: Dict[str, Any]) -> None:
    print("=" * 64)
    print(f"Run        : {payload.get('run_id')}  [{payload.get('status')}]")
    print(f"Config     : {payload.get('config_name')}")
    metric = payload.get("primary_metric")
    if isinstance(metric, (int, float)):
        print(f"Primary    : {metric:.4f} ({payload.get('primary_metric_status')})")
    print(f"Wall time  : {payload.get('wall_time_s')} s")
    print(f"Output dir : {payload.get('output_dir')}")
    datasets = payload.get("datasets") or {}
    if datasets:
        print("-" * 64)
        for key, result in sorted(datasets.items()):
            score = result.get("score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            n_docs = result.get("n_documents", "?")
            print(f"  {key:<28} score={score_str:<8} docs={n_docs} [{result.get('status')}]")
    errors = payload.get("errors") or []
    if errors:
        print("-" * 64)
        print(f"Errors ({len(errors)}):")
        for error in errors[:5]:
            print(f"  {error.get('dataset')}: {error.get('error')}")
    print("=" * 64)


def cmd_list_datasets(_args: argparse.Namespace) -> int:
    for info in list_available_datasets():
        supports = ", ".join(key for key, value in info["supports"].items() if value)
        print(f"{info['name']:<16} {info['description']}")
        print(f"{'':<16} supports: {supports}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    runner = EvaluationRunner.from_config(args.config) if args.config else EvaluationRunner()
    overrides: Dict[str, Any] = {}
    if args.split:
        overrides["split"] = args.split
    if args.limit is not None:
        overrides["limit"] = args.limit
    if args.no_llm:
        overrides["no_llm"] = True
    if args.skip_risk:
        overrides["skip_risk"] = True
    payload = runner.run(dataset=args.dataset, output=args.output, **overrides)
    _print_run_summary(payload)
    return 0 if payload.get("status") != "error" else 1


def cmd_ablation(args: argparse.Namespace) -> int:
    runner = EvaluationRunner.from_config(args.config) if args.config else EvaluationRunner()
    if args.limit is not None:
        runner.config["limit"] = args.limit
    summary = runner.run_ablation(
        dataset=args.dataset, ablation_config=args.ablation_config, output=args.output
    )
    print("=" * 64)
    print(f"Ablation on {', '.join(summary['dataset'])} — {len(summary['variants'])} variants")
    for row in sorted(summary["variants"], key=lambda r: (r.get("primary_metric") or 0), reverse=True):
        metric = row.get("primary_metric")
        metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "n/a"
        print(f"  {row['name']:<24} primary={metric_str} [{row.get('status')}]")
    print(f"Output dir : {summary.get('output_dir')}")
    print("=" * 64)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    comparison = compare_runs(args.runs, output_dir=args.output)
    for row in comparison["runs"]:
        metric = row.get("primary_metric")
        metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "n/a"
        delta = row.get("delta_vs_first")
        delta_str = f" ({delta:+.4f})" if isinstance(delta, (int, float)) else ""
        print(f"  {row['run_dir']:<48} primary={metric_str}{delta_str}")
    if comparison.get("output_dir"):
        print(f"Comparison artifacts: {comparison['output_dir']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if args.format == "json":
        print(json.dumps(load_metrics(args.run), indent=2, ensure_ascii=False))
    else:
        print(load_evaluation_report(args.run))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anonymation-eval",
        description="Unified evaluation engine for the anonymization pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-datasets", help="List registered datasets").set_defaults(func=cmd_list_datasets)

    p_run = sub.add_parser("run", help="Evaluate one dataset or all")
    p_run.add_argument("--dataset", default="all", help="Dataset name or 'all'")
    p_run.add_argument("--config", default=None, help="Evaluation config (JSON/YAML)")
    p_run.add_argument("--split", default=None, choices=["test", "dev", "train"])
    p_run.add_argument("--limit", type=int, default=None, help="Max documents (debug)")
    p_run.add_argument("--output", default=None, help="Run output directory")
    p_run.add_argument("--no-llm", action="store_true", help="Disable pipeline LLM modules")
    p_run.add_argument("--skip-risk", action="store_true", help="Skip RAT-Bench LLM risk axis")
    p_run.set_defaults(func=cmd_run)

    p_abl = sub.add_parser("ablation", help="Run an ablation study")
    p_abl.add_argument("--dataset", default="all")
    p_abl.add_argument("--config", default=None, help="Base evaluation config")
    p_abl.add_argument("--ablation-config", required=True, help="Ablation variants (JSON/YAML)")
    p_abl.add_argument("--limit", type=int, default=None, help="Max documents per variant")
    p_abl.add_argument("--output", default=None)
    p_abl.set_defaults(func=cmd_ablation)

    p_cmp = sub.add_parser("compare", help="Compare several run directories")
    p_cmp.add_argument("--runs", nargs="+", required=True)
    p_cmp.add_argument("--output", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    p_rep = sub.add_parser("report", help="Print the report of a run")
    p_rep.add_argument("--run", required=True)
    p_rep.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
