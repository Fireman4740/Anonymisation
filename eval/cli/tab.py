from __future__ import annotations

import argparse
import os
from typing import Optional

from eval.cli.common import build_standard_runtime_config, save_detailed_report, save_optional_run
from eval.core.bootstrap import load_pipegraph, project_root
from eval.pipegraph_eval_local import build_docs_from_tab, build_report


def _default_tab_path(repo_root: str, split: str) -> str:
    return os.path.join(repo_root, "eval", "datasets", "TAB", f"{split}.jsonl")


def _default_out_path(repo_root: str, split: str) -> str:
    return os.path.join(repo_root, "eval", "evaluation", "reports", f"report_TAB_pipegraph_{split}_details.json")


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

    repo_root = project_root()
    in_path = args.in_path or _default_tab_path(repo_root, args.split)
    out_path = args.out_path or _default_out_path(repo_root, args.split)

    create_pipeline_graph, create_initial_state = load_pipegraph()
    pipeline = create_pipeline_graph()

    config = build_standard_runtime_config(
        enable_detection=bool(args.enable_detection),
        enable_deterministic=bool(args.enable_deterministic),
        enable_ai=bool(args.enable_ai),
        enable_anonymization=bool(args.enable_anonymization),
        detection_mode=str(args.detection_mode),
    )
    limit = int(args.limit) if args.limit else None
    docs = build_docs_from_tab(in_path, limit=limit)
    report = build_report(docs, pipeline, create_initial_state, config=config)

    report_meta = save_detailed_report(
        out_path=out_path,
        dataset_name=f"TAB/{args.split}",
        dataset_path=in_path,
        limit=limit,
        config=config,
        report=report,
        run_name=args.run_name,
    )
    print(f"Wrote {len(report)} docs to: {out_path}")

    saved = save_optional_run(
        enabled=bool(args.save_run),
        runs_dir=os.path.join(repo_root, "eval", "evaluation", "runs"),
        report=report,
        meta=report_meta,
        run_name=args.run_name,
    )
    if saved:
        print(f"Saved run: {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
