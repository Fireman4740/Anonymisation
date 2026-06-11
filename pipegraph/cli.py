"""Command-line interface for the PipeGraph anonymization pipeline.

Usage (from the repo root)::

    python -m pipegraph anonymize --text "Jean Dupont habite à Paris" --no-llm
    python -m pipegraph anonymize-file --input doc.txt --output doc.anon.txt
    python -m pipegraph anonymize --text "..." --json > result.json

Evaluation has its own entry points (``python eval/evaluate.py`` and
``python -m eval.run_pipeline_evaluation``).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from pipegraph.api import AnonymizationResult, anonymize, anonymize_file, load_config


def _parse_config_overrides(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--config-overrides must be valid JSON: {exc}")
    if not isinstance(config, dict):
        raise SystemExit("--config-overrides must be a JSON object")
    return config


def _print_result(result: AnonymizationResult, *, as_json: bool, show_original: bool) -> None:
    if as_json:
        payload = result.to_dict()
        if not show_original:
            payload.pop("original_text", None)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    print(result.anonymized_text)
    print("-" * 60, file=sys.stderr)
    print(f"Entities detected: {len(result.entities)}", file=sys.stderr)
    for ent in result.entities:
        etype = ent.get("type") or ent.get("entity_type")
        value = ent.get("value") or ent.get("text")
        source = ent.get("source", "?")
        print(f"  [{etype}] {value!r} ({ent.get('start')}:{ent.get('end')}, {source})", file=sys.stderr)
    if result.privacy_score is not None:
        print(f"Privacy score (0=anonymous): {result.privacy_score}", file=sys.stderr)
    if result.errors:
        print(f"Errors: {result.errors}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipegraph",
        description="Hybrid text anonymization pipeline (regex + NER + LLM).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable every LLM node and the RUPTA loop (fully offline run)",
    )
    common.add_argument(
        "--config",
        metavar="PATH",
        help="Runtime config file (JSON/YAML), e.g. configs/baselines/no_llm.json",
    )
    common.add_argument(
        "--config-overrides",
        metavar="JSON",
        help='Runtime config overrides as JSON, e.g. \'{"anon_strategy": "mask"}\' (applied on top of --config)',
    )
    common.add_argument("--json", action="store_true", help="Print the full result as JSON")
    common.add_argument(
        "--show-original",
        action="store_true",
        help="Include the original text in JSON output (off by default for confidentiality)",
    )

    p_text = subparsers.add_parser("anonymize", parents=[common], help="Anonymize a text string")
    p_text.add_argument("--text", required=True, help="Text to anonymize")

    p_file = subparsers.add_parser("anonymize-file", parents=[common], help="Anonymize a text file")
    p_file.add_argument("--input", required=True, help="Input text file")
    p_file.add_argument("--output", required=True, help="Output file for the anonymized text")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    config.update(_parse_config_overrides(args.config_overrides))

    if args.command == "anonymize":
        result = anonymize(args.text, config, no_llm=args.no_llm)
        _print_result(result, as_json=args.json, show_original=args.show_original)
    elif args.command == "anonymize-file":
        result = anonymize_file(args.input, args.output, config, no_llm=args.no_llm)
        print(f"Anonymized text written to: {args.output}", file=sys.stderr)
        _print_result(result, as_json=args.json, show_original=args.show_original)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
