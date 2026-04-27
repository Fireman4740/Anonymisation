from __future__ import annotations

import argparse
import os
from typing import Optional

from eval.cli.common import build_standard_runtime_config, save_optional_run
from eval.core.bootstrap import load_pipegraph, project_root
from eval.core.ratbench import (
    aggregate_ratbench_metrics,
    build_ratbench_meta,
    build_ratbench_report,
    build_ratbench_result,
)
from eval.core.reporting import save_report_payload
from eval.ratbench_loader import build_docs_from_ratbench, download_ratbench, load_ratbench_profiles


def _default_out_path(repo_root: str, language: str, level: Optional[int]) -> str:
    level_str = f"_L{level}" if level is not None else "_all"
    return os.path.join(
        repo_root,
        "eval",
        "evaluation",
        "reports",
        f"report_RAT-Bench_{language}{level_str}_pipegraph_details.json",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Évalue PipeGraph sur le benchmark RAT-Bench.")
    parser.add_argument("--language", choices=["english", "mandarin", "spanish"], default="english", help="Langue du dataset RAT-Bench")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None, help="Niveau de difficulté (1-3). Omit = tous les niveaux.")
    parser.add_argument("--limit", type=int, default=100, help="Max profils à évaluer")
    parser.add_argument("--out", dest="out_path", default=None, help="Chemin du report JSON")
    parser.add_argument("--enable-detection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ai", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-anonymization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detection-mode", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--run-name", default=None, help="Nom optionnel du run")
    parser.add_argument("--save-run", action="store_true", help="Sauvegarde dans eval/evaluation/runs/")
    parser.add_argument("--force-download", action="store_true", help="Force re-téléchargement du dataset")
    args = parser.parse_args(argv)

    repo_root = project_root()
    out_path = args.out_path or _default_out_path(repo_root, args.language, args.level)

    print("🔧 Chargement du pipeline PipeGraph...")
    create_pipeline_graph, create_initial_state = load_pipegraph()
    pipeline = create_pipeline_graph()

    config = build_standard_runtime_config(
        enable_detection=bool(args.enable_detection),
        enable_deterministic=bool(args.enable_deterministic),
        enable_ai=bool(args.enable_ai),
        enable_anonymization=bool(args.enable_anonymization),
        detection_mode=str(args.detection_mode),
    )

    print(f"📥 Chargement RAT-Bench (language={args.language}, level={args.level}, limit={args.limit})...")
    if args.force_download:
        download_ratbench(language=args.language, force_download=True)
    profiles = load_ratbench_profiles(language=args.language, level=args.level, limit=args.limit)
    docs = build_docs_from_ratbench(language=args.language, level=args.level, limit=args.limit)
    print(f"   → {len(docs)} documents chargés")

    print("⚙️  Évaluation en cours...")
    report = build_ratbench_report(docs, profiles, pipeline, create_initial_state, config=config)
    result = build_ratbench_result(report=report, language=args.language, level=args.level, config=config)
    agg = aggregate_ratbench_metrics(report)

    print("\n" + "=" * 60)
    print("📊 RAT-Bench Evaluation Results")
    print("=" * 60)
    print(f"  Documents:       {agg.get('n_documents', 0)}")
    print(f"  Macro Precision: {agg.get('macro_precision', 0):.4f}")
    print(f"  Macro Recall:    {agg.get('macro_recall', 0):.4f}")
    print(f"  Macro F2:        {agg.get('macro_f2', 0):.4f}")
    print(f"  Total Leaks:     {agg.get('total_leaks', 0)}")

    print("\n📈 Par niveau de difficulté:")
    for level_key, metrics in result["by_difficulty"].items():
        print(f"  Level {level_key}: P={metrics['macro_precision']:.4f}  R={metrics['macro_recall']:.4f}  F2={metrics['macro_f2']:.4f}  (n={metrics['n_documents']})")

    print("\n📋 Par scénario:")
    for scenario, metrics in result["by_scenario"].items():
        print(f"  {scenario}: P={metrics['macro_precision']:.4f}  R={metrics['macro_recall']:.4f}  F2={metrics['macro_f2']:.4f}  (n={metrics['n_documents']})")

    print("\n🔍 Taux de détection des identifiants directs:")
    for identifier_type, stats in result["direct_id_detection_rates"].items():
        print(f"  {identifier_type}: {stats['detected']}/{stats['total']} ({stats['detection_rate']:.1%})")

    print("\n🛡️  Analyse de fuites (text-leak):")
    print(f"  Taux de fuite moyen:           {result['summary']['avg_leak_rate']:.1%}")
    print(f"  Taux de fuite (directs):       {result['summary']['avg_direct_leak_rate']:.1%}")
    print(f"  Taux de fuite (indirects):     {result['summary']['avg_indirect_leak_rate']:.1%}")

    report_meta = build_ratbench_meta(
        report=report,
        language=args.language,
        level=args.level,
        config=config,
        limit=args.limit,
        run_name=args.run_name,
    )
    save_report_payload(out_path, meta=report_meta, data=report)
    print(f"\n💾 Report sauvegardé: {out_path}")

    saved = save_optional_run(
        enabled=bool(args.save_run),
        runs_dir=os.path.join(repo_root, "eval", "evaluation", "runs"),
        report=report,
        meta=report_meta,
        run_name=args.run_name,
    )
    if saved:
        print(f"📁 Run sauvegardé: {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
