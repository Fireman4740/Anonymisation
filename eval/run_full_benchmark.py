#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.core.profiles import EVAL_MODE_CHOICES, MASKING_MODE_CHOICES, PROFILE_CHOICES
from eval.run_pipeline_evaluation import AVAILABLE_DATASETS, run_evaluation

DEFAULT_DATASETS = ["tab", "dbbio", "anonymization", "ratbench", "conll2003"]
DEFAULT_LEVELS = [1, 2, 3]


def _legacy_out_to_output_dir(out_path: Optional[str]) -> Optional[str]:
    if not out_path:
        return None
    root, ext = os.path.splitext(out_path)
    return f"{root}_official" if ext else out_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for the official PipeGraph evaluation runner. "
            "Prefer: python -m eval.run_pipeline_evaluation"
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=AVAILABLE_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--risk-limit", type=int, default=50)
    parser.add_argument("--levels", nargs="+", type=int, choices=[1, 2, 3], default=DEFAULT_LEVELS)
    parser.add_argument("--language", choices=["english", "mandarin", "spanish"], default="english")
    parser.add_argument("--split", choices=["test", "dev", "train"], default="test")

    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--with-llm", action="store_true", help="Accepted for legacy compatibility.")
    parser.add_argument("--with-rupta", action="store_true", help="Accepted for legacy compatibility.")
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="auto")
    parser.add_argument("--eval-mode", choices=EVAL_MODE_CHOICES, default="both")
    parser.add_argument("--masking-mode", choices=MASKING_MODE_CHOICES, default="benchmark")
    parser.add_argument(
        "--detection-mode",
        choices=["serial", "parallel"],
        default="parallel",
        help="Accepted for legacy compatibility; official candidate configs control detection mode.",
    )

    parser.add_argument("--skip-risk", action="store_true")
    parser.add_argument("--risk-only", action="store_true")
    parser.add_argument("--save-runs", action="store_true")
    parser.add_argument("--out", default=None, help="Legacy report path; converted to an output directory.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    payload = run_evaluation(
        argparse.Namespace(
            candidate=None,
            datasets=list(args.datasets),
            split=str(args.split),
            limit=args.risk_limit if args.risk_only else args.limit,
            output=_legacy_out_to_output_dir(args.out),
            save_runs=bool(args.save_runs),
            doc_workers=None,
            profile=str(args.profile),
            eval_mode=str(args.eval_mode),
            masking_mode=str(args.masking_mode),
            language=str(args.language),
            ratbench_languages=[str(args.language)],
            ratbench_levels=[int(level) for level in args.levels],
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_attacker_model=None,
            no_llm=bool(args.no_llm),
            skip_risk=bool(args.skip_risk),
            require_risk=False,
            risk_limit=args.risk_limit,
        )
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
    print(f"Official evaluation written to: {payload.get('output_dir')}")
    return 0 if payload.get("status") in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
