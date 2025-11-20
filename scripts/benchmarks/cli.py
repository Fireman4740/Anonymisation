"""Command line interface for benchmark workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional


def parse_samples(value: str) -> int:
    """Parse --samples argument, allowing the keyword 'all'."""
    text = str(value).strip()
    if text.lower() in {"all", "full", "max", "everything", "*"}:
        return -1
    number = int(text)
    if number <= 0:
        return -1
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark runner")
    sub = parser.add_subparsers(dest="command")

    ner_parser = sub.add_parser("ner", help="Comparer NER standard et GPU")
    ner_parser.add_argument("--mode", choices=["standard", "gpu", "both"], default="both")
    ner_parser.add_argument("--text-size", choices=["short", "medium", "long"], default="medium")
    ner_parser.add_argument("--runs", type=int, default=3)
    ner_parser.add_argument("--preset", default="balanced")
    ner_parser.add_argument("--threshold", type=float, default=0.35)
    def ner_handler(args):
        from . import ner

        return ner.handle_cli(args)

    ner_parser.set_defaults(handler=ner_handler)

    pipeline_parser = sub.add_parser("pipeline", help="Lancer les benchmarks anonymisation")
    pipeline_parser.add_argument("--dataset", choices=["dbbio", "reddit", "tab", "all"], default="dbbio")
    pipeline_parser.add_argument("--split", default="test")
    pipeline_parser.add_argument(
        "--samples",
        type=parse_samples,
        default=10,
        help="Nombre d'échantillons à traiter (utiliser 'all' ou une valeur <= 0 pour traiter tout le dataset)",
    )
    pipeline_parser.add_argument("--policy", choices=["L0", "L1"], default="L1")
    pipeline_parser.add_argument("--baseline-only", action="store_true")
    pipeline_parser.add_argument("--rupta-only", action="store_true")
    pipeline_parser.add_argument("--output", type=Path)
    pipeline_parser.add_argument("--rate-limit", type=float, default=0.0, help="Pause en secondes entre les samples")
    def pipeline_handler(args):
        from . import pipeline

        return pipeline.handle_cli(args)

    pipeline_parser.set_defaults(handler=pipeline_handler)

    compare_parser = sub.add_parser("compare", help="Comparer deux resultats")
    compare_parser.add_argument("--baseline", required=True, type=Path)
    compare_parser.add_argument("--rupta", required=True, type=Path)
    compare_parser.add_argument("--output", type=Path)
    def compare_handler(args):
        from . import compare

        return compare.handle_cli(args)

    compare_parser.set_defaults(handler=compare_handler)

    quick_parser = sub.add_parser("quick", help="Benchmarks rapides (NER + DB-Bio)")
    quick_parser.add_argument("--samples", type=int, default=3)
    quick_parser.add_argument("--policy", choices=["L0", "L1"], default="L1")
    def quick_handler(args):
        from . import workflows

        return workflows.handle_quick_cli(args)

    quick_parser.set_defaults(handler=quick_handler)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    result = args.handler(args)
    if isinstance(result, int):
        return result
    return 0
