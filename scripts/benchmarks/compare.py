"""Comparison helpers between baseline and RUPTA results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "summary" in data:
        return data["summary"]
    datasets = data.get("datasets", {})
    for dataset in datasets.values():
        if "summary" in dataset:
            return dataset["summary"]
    raise ValueError(f"Unable to locate summaries in {path}")


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def build_report(baseline_path: Path, rupta_path: Path) -> str:
    baseline = load_summary(baseline_path)
    rupta = load_summary(rupta_path)

    def describe(metric: str, formatter) -> str:
        base_val = baseline.get(metric)
        rup_val = rupta.get(metric)
        delta = diff(base_val, rup_val)
        delta_txt = formatter(delta) if delta is not None else "N/A"
        return (
            f"- {metric.replace('_', ' ')}\n"
            f"  * baseline : {formatter(base_val)}\n"
            f"  * rupta    : {formatter(rup_val)}\n"
            f"  * delta    : {delta_txt}\n"
        )

    lines = [
        "# Rapport comparaison Baseline vs RUPTA",
        describe("avg_privacy_rank", format_number),
        describe("privacy_non_identified_rate", format_rate),
        describe("avg_utility_confidence", format_number),
        describe("utility_preserved_rate", format_rate),
        describe("avg_runtime_seconds", format_number),
    ]
    return "\n".join(lines)


def handle_cli(args) -> str:
    report = build_report(args.baseline, args.rupta)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"\nReport written to {args.output}")
    return report
