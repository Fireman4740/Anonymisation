from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.pipegraph_eval_local import build_docs_from_tab, load_pipegraph
from eval.run_store import save_run, utc_now_iso


def _default_tab_path(project_root: str, split: str) -> str:
    return os.path.join(project_root, "eval", "datasets", "TAB", f"{split}.jsonl")


def _default_out_path(project_root: str, split: str) -> str:
    return os.path.join(project_root, "eval", "evaluation", "reports", f"report_TAB_pipegraph_{split}_details.json")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Évalue PipeGraph localement sur TAB et génère un report Streamlit.")
    parser.add_argument("--split", choices=["test", "dev", "train"], default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--in", dest="in_path", default=None, help="Chemin vers TAB <split>.jsonl")
    parser.add_argument("--out", dest="out_path", default=None, help="Chemin du report JSON à écrire")

    parser.add_argument("--enable-detection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ai", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-anonymization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detection-mode", choices=["serial", "parallel"], default="serial")

    parser.add_argument("--run-name", default=None, help="Nom optionnel du run (utilisé si --save-run)")
    parser.add_argument("--save-run", action="store_true", help="Sauvegarde meta+data dans eval/evaluation/runs/")
    args = parser.parse_args(argv)

    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    in_path = args.in_path or _default_tab_path(project_root, args.split)
    out_path = args.out_path or _default_out_path(project_root, args.split)

    create_pipeline_graph, create_initial_state = load_pipegraph()
    pipeline = create_pipeline_graph()

    config: Dict[str, Any] = {
        "enable_detection": bool(args.enable_detection),
        "enable_deterministic": bool(args.enable_deterministic),
        "enable_ai": bool(args.enable_ai),
        "enable_anonymization": bool(args.enable_anonymization),
        "detection_mode": str(args.detection_mode),
    }

    docs = build_docs_from_tab(in_path, limit=int(args.limit) if args.limit else None)

    # Import here to avoid circular import in type-checking contexts
    from eval.pipegraph_eval_local import build_report

    report = build_report(docs, pipeline, create_initial_state, config=config)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(report)} docs to: {out_path}")

    if args.save_run:
        runs_dir = os.path.join(project_root, "eval", "evaluation", "runs")
        meta = {
            "created_at": utc_now_iso(),
            "pipeline": "pipegraph",
            "run_name": args.run_name,
            "dataset": {"name": f"TAB/{args.split}", "path": in_path},
            "limit": int(args.limit),
            "config": config,
        }
        saved = save_run(runs_dir, meta=meta, data=report, run_name=args.run_name)
        print(f"Saved run: {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
