from __future__ import annotations

import argparse
from pathlib import Path

from atlas_anno.anonymization.baselines import run_anonymization_command
from atlas_anno.annotation.preannotator import run_preannotation_command
from atlas_anno.attacks.llm_attacker import run_llm_attack_command
from atlas_anno.attacks.pairs import run_build_attack_pairs_command
from atlas_anno.attacks.structured import run_structured_attack_command
from atlas_anno.dashboard.app import run_dashboard_command
from atlas_anno.diagnostics import run_inspect_llm_runs_command
from atlas_anno.evaluation.aggregate import (
    run_eval_privacy_command,
    run_eval_reid_command,
    run_eval_spans_command,
    run_eval_utility_command,
)
from atlas_anno.evaluation.calibration import run_calibrate_difficulty_command
from atlas_anno.evaluation.diversity import run_eval_diversity_command
from atlas_anno.evaluation.realism_judge import run_judge_realism_command
from atlas_anno.export.parquet_export import run_export_parquet_command
from atlas_anno.generation.pipeline import (
    run_generate_dataset_command,
    run_generate_characters_command,
    run_generate_scenarios_command,
    run_generate_texts_command,
    run_generate_worlds_command,
    run_validate_dataset_command,
)
from atlas_anno.review.label_studio import export_label_studio_review_pack, import_label_studio_review_pack
from atlas_anno.reporting.builder import run_build_report_command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atlas_anno autonomous benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_worlds = subparsers.add_parser("generate-worlds")
    generate_worlds.add_argument("--count", type=int, default=3)

    generate_characters = subparsers.add_parser("generate-characters")
    generate_characters.add_argument("--per-world", type=int, default=34)

    generate_scenarios = subparsers.add_parser("generate-scenarios")
    generate_scenarios.add_argument("--documents", type=int, default=3000)

    generate_dataset = subparsers.add_parser("generate-dataset")
    generate_dataset.add_argument("--documents", type=int, default=100)
    generate_dataset.add_argument("--llm-mode", choices=["primary-fallback", "disabled"], default="primary-fallback")
    generate_dataset.add_argument("--reasoning-workers", type=int, default=None)
    generate_dataset.add_argument("--creative-workers", type=int, default=None)
    generate_dataset.add_argument("--resume", dest="resume", action="store_true")
    generate_dataset.add_argument("--no-resume", dest="resume", action="store_false")
    generate_dataset.add_argument("--cache", dest="cache", action="store_true")
    generate_dataset.add_argument("--no-cache", dest="cache", action="store_false")
    generate_dataset.set_defaults(resume=None, cache=None)

    subparsers.add_parser("generate-texts")
    preannotate = subparsers.add_parser("preannotate")
    preannotate.add_argument("--mode", choices=["hybrid-llm", "disabled"], default="hybrid-llm")
    preannotate.add_argument("--batch", default="pilot_100")
    preannotate.add_argument("--reasoning-workers", type=int, default=None)
    preannotate.add_argument("--resume", dest="resume", action="store_true")
    preannotate.add_argument("--no-resume", dest="resume", action="store_false")
    preannotate.add_argument("--cache", dest="cache", action="store_true")
    preannotate.add_argument("--no-cache", dest="cache", action="store_false")
    preannotate.set_defaults(resume=None, cache=None)
    subparsers.add_parser("validate-dataset")

    anonymizer = subparsers.add_parser("run-anonymizer")
    anonymizer.add_argument("--strategy", choices=["masking", "generalization", "rewrite_balanced"], default="masking")
    anonymizer.add_argument("--mode", choices=["auto", "llm"], default="auto")

    structured = subparsers.add_parser("attack-structured")
    structured.add_argument("--strategy", default="masking")

    llm = subparsers.add_parser("attack-llm")
    llm.add_argument("--strategy", default="masking")

    subparsers.add_parser("build-attack-pairs")
    subparsers.add_parser("calibrate-difficulty")
    subparsers.add_parser("eval-diversity")

    judge_realism = subparsers.add_parser("judge-realism")
    judge_realism.add_argument("--mode", default="primary-fallback", choices=["primary-fallback", "disabled"])
    judge_realism.add_argument("--sample-rate", type=float, default=None)

    spans = subparsers.add_parser("eval-spans")
    spans.add_argument("--strategy", default="masking")

    privacy = subparsers.add_parser("eval-privacy")
    privacy.add_argument("--strategy", default="masking")

    reid = subparsers.add_parser("eval-reid")
    reid.add_argument("--strategy", default="masking")

    utility = subparsers.add_parser("eval-utility")
    utility.add_argument("--strategy", default="masking")

    report = subparsers.add_parser("build-report")
    report.add_argument("--strategy", default="masking")

    export_review = subparsers.add_parser("export-review-pack")
    export_review.add_argument("--target", choices=["label-studio"], default="label-studio")
    export_review.add_argument("--batch", default="pilot_100")
    export_review.add_argument("--selection", choices=["all", "review-required"], default="all")

    import_review = subparsers.add_parser("import-review-pack")
    import_review.add_argument("--target", choices=["label-studio"], default="label-studio")
    import_review.add_argument("--batch", default="pilot_100")
    import_review.add_argument("--input", required=True)

    export_parquet = subparsers.add_parser("export-parquet")
    export_parquet.add_argument("--batch", default="pilot_100")

    inspect_runs = subparsers.add_parser("inspect-llm-runs")
    inspect_runs.add_argument("--limit", type=int, default=20)

    dashboard = subparsers.add_parser("dashboard")
    dashboard.add_argument("--batch", default="pilot_100")
    dashboard.add_argument("--strategy", default="masking")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "generate-worlds": lambda: run_generate_worlds_command(args.count),
        "generate-characters": lambda: run_generate_characters_command(args.per_world),
        "generate-scenarios": lambda: run_generate_scenarios_command(args.documents),
        "generate-dataset": lambda: run_generate_dataset_command(
            args.documents,
            args.llm_mode,
            reasoning_workers=args.reasoning_workers,
            creative_workers=args.creative_workers,
            resume_enabled=args.resume,
            cache_enabled=args.cache,
        ),
        "generate-texts": run_generate_texts_command,
        "preannotate": lambda: run_preannotation_command(
            args.mode,
            batch=args.batch,
            reasoning_workers=args.reasoning_workers,
            resume_enabled=args.resume,
            cache_enabled=args.cache,
        ),
        "validate-dataset": run_validate_dataset_command,
        "run-anonymizer": lambda: run_anonymization_command(args.strategy, args.mode),
        "attack-structured": lambda: run_structured_attack_command(args.strategy),
        "attack-llm": lambda: run_llm_attack_command(args.strategy),
        "build-attack-pairs": run_build_attack_pairs_command,
        "calibrate-difficulty": run_calibrate_difficulty_command,
        "eval-diversity": run_eval_diversity_command,
        "judge-realism": lambda: run_judge_realism_command(args.mode, args.sample_rate),
        "eval-spans": lambda: run_eval_spans_command(args.strategy),
        "eval-privacy": lambda: run_eval_privacy_command(args.strategy),
        "eval-reid": lambda: run_eval_reid_command(args.strategy),
        "eval-utility": lambda: run_eval_utility_command(args.strategy),
        "build-report": lambda: run_build_report_command(args.strategy),
        "export-review-pack": lambda: export_label_studio_review_pack(args.batch, selection=args.selection),
        "import-review-pack": lambda: import_label_studio_review_pack(args.batch, args.input),
        "export-parquet": lambda: run_export_parquet_command(args.batch),
        "inspect-llm-runs": lambda: run_inspect_llm_runs_command(args.limit),
        "dashboard": lambda: run_dashboard_command(batch=args.batch, strategy=args.strategy),
    }
    handlers[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
