from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.core.profiles import EVAL_MODE_CHOICES, MASKING_MODE_CHOICES, PROFILE_CHOICES
from eval.ratbench_loader import download_ratbench
from eval.run_pipeline_evaluation import run_evaluation


def _legacy_out_to_output_dir(out_path: Optional[str]) -> Optional[str]:
    if not out_path:
        return None
    root, ext = os.path.splitext(out_path)
    return f"{root}_official" if ext else out_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for RAT-Bench evaluation. "
            "Prefer: python -m eval.run_pipeline_evaluation --datasets ratbench"
        )
    )
    parser.add_argument("--language", choices=["english", "mandarin", "spanish"], default="english")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", dest="out_path", default=None)
    parser.add_argument("--enable-detection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ai", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-anonymization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detection-mode", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--with-llm", action="store_true", help="Accepted for legacy compatibility.")
    parser.add_argument("--with-rupta", action="store_true", help="Accepted for legacy compatibility.")
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="auto")
    parser.add_argument("--eval-mode", choices=EVAL_MODE_CHOICES, default="both")
    parser.add_argument("--masking-mode", choices=MASKING_MODE_CHOICES, default="benchmark")
    parser.add_argument("--run-name", default=None, help="Accepted for legacy compatibility.")
    parser.add_argument("--save-run", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-risk", action="store_true")
    parser.add_argument("--require-risk", action="store_true")
    args = parser.parse_args(argv)

    if args.force_download:
        download_ratbench(language=args.language, force_download=True)

    payload = run_evaluation(
        argparse.Namespace(
            candidate=None,
            datasets=["ratbench"],
            split="test",
            limit=int(args.limit) if args.limit else None,
            output=_legacy_out_to_output_dir(args.out_path),
            save_runs=bool(args.save_run),
            doc_workers=1,
            profile=str(args.profile),
            eval_mode=str(args.eval_mode),
            masking_mode=str(args.masking_mode),
            language=str(args.language),
            ratbench_languages=[str(args.language)],
            ratbench_levels=[int(args.level)] if args.level is not None else [1, 2, 3],
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_attacker_model=None,
            no_llm=bool(args.no_llm),
            skip_risk=bool(args.skip_risk),
            require_risk=bool(args.require_risk),
            risk_limit=None,
        )
    )
    print(f"Official evaluation written to: {payload.get('output_dir')}")
    return 0 if payload.get("status") in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
