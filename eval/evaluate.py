#!/usr/bin/env python3
"""
eval/evaluate.py — Point d'entrée unique pour l'évaluation du pipeline PipeGraph.

Usage
-----
# Benchmark complet (détection + leaks + risque)
  python eval/evaluate.py benchmark --datasets tab ratbench --limit 50

# Un seul dataset
  python eval/evaluate.py benchmark --datasets tab --split test --limit 100

# RAT-Bench L1 uniquement, sans risque (rapide)
  python eval/evaluate.py benchmark --datasets ratbench --levels 1 --skip-risk --limit 50

# Ablation des nœuds sur TAB
  python eval/evaluate.py ablation --dataset tab --suite nodes --limit 20

# Évaluation standalone d'un dataset
  python eval/evaluate.py dataset --dataset tab --split test --limit 50

# Sans LLM (regex+NER uniquement)
  python eval/evaluate.py benchmark --datasets tab --no-llm --limit 50

Sous-commandes
--------------
  benchmark   Évaluation multi-dataset multi-axe (remplace run_full_benchmark.py)
  ablation    Grille d'ablation sur un seul dataset (remplace run_ablation.py)
  dataset     Évaluation standalone par dataset (remplace cli/tab.py, cli/ratbench.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_EVAL_DIR, ".."))
for _p in (_EVAL_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.core.bootstrap import ensure_pipegraph_importable, load_pipegraph, project_root
from eval.core.config import build_runtime_config
from eval.core.datasets import DATASET_ALIASES, get_allowed_labels, load_benchmark_docs
from eval.core.profiles import (
    EVAL_MODE_CHOICES,
    MASKING_MODE_CHOICES,
    PROFILE_CHOICES,
    apply_profile_to_config,
)
from eval.core.ratbench import (
    aggregate_ratbench_metrics,
    compute_leak_summary,
    direct_id_detection_rate,
    metrics_by_difficulty,
    metrics_by_scenario,
)
from eval.core.reporting import aggregate_document_metrics, build_report_meta, save_report_payload
from eval.pipegraph_eval_local import build_report
from eval.run_store import save_run, utc_now_iso

# Ablation suites (imported from run_ablation to avoid duplication)
try:
    from eval.run_ablation import SUITES, _agg, _augment_dataset_runtime_config, _print_table
except ImportError:
    SUITES = {}  # type: ignore[assignment]

ALL_DATASETS = ["tab", "dbbio", "anonymization", "ratbench", "conll2003", "personalreddit"]
RATBENCH_RISK_THRESHOLD = 0.2  # R_succ threshold: R > 0.2 → re-identified (paper §4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _default_reports_dir() -> str:
    return os.path.join(_PROJECT_ROOT, "eval", "evaluation", "reports")


def _default_runs_dir() -> str:
    return os.path.join(_PROJECT_ROOT, "eval", "evaluation", "runs")


def _f2(p: float, r: float, beta: float = 2.0) -> float:
    if p <= 0.0 and r <= 0.0:
        return 0.0
    b2 = beta * beta
    d = b2 * p + r
    return (1.0 + b2) * p * r / d if d > 0 else 0.0


def _build_config(
    *,
    dataset: str,
    profile: str = "auto",
    eval_mode: str = "both",
    masking_mode: str = "benchmark",
    no_llm: bool = False,
    detection_mode: str = "parallel",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = build_runtime_config(
        enable_detection=True,
        enable_deterministic=True,
        enable_ai=True,
        enable_anonymization=True,
        detection_mode=detection_mode,
        dataset_key=dataset,
        profile=profile,
        eval_mode=eval_mode,
        masking_mode=masking_mode,
    )
    if no_llm:
        cfg.update({
            "disable_llm": True,
            "llm_detection": False,
            "llm_verification": False,
            "llm_audit": False,
            "llm_paraphrase": False,
            "rupta_enabled": False,
        })
    if llm_provider:
        cfg["llm_provider"] = llm_provider
    if llm_model:
        cfg["llm_model"] = llm_model
    if extra:
        cfg.update(extra)
    return cfg


def _print_detection_summary(
    dataset_label: str,
    metrics: Dict[str, Any],
    *,
    label_metrics: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> None:
    print(f"\n{'='*60}")
    print(f"  {dataset_label}")
    print(f"{'='*60}")

    if strict and metrics.get("macro_strict_precision") is not None:
        print(f"  [STRICT]  P={metrics['macro_strict_precision']:.4f}  "
              f"R={metrics['macro_strict_recall']:.4f}  "
              f"F1={metrics['macro_strict_f1']:.4f}  "
              f"F2={metrics['macro_strict_f2']:.4f}")
        total_err = sum(metrics.get(f"total_{k}", 0) for k in ("missed", "spurious", "boundary_error", "type_error"))
        if total_err:
            print(f"  Errors: missed={metrics.get('total_missed',0)}  "
                  f"spurious={metrics.get('total_spurious',0)}  "
                  f"boundary={metrics.get('total_boundary_error',0)}  "
                  f"type={metrics.get('total_type_error',0)}")

    print(f"  [PARTIAL] P={metrics.get('macro_precision',0):.4f}  "
          f"R={metrics.get('macro_recall',0):.4f}  "
          f"F2={metrics.get('macro_f2',0):.4f}")
    print(f"  [MICRO]   P={metrics.get('micro_precision',0):.4f}  "
          f"R={metrics.get('micro_recall',0):.4f}  "
          f"F2={metrics.get('micro_f2',0):.4f}")

    if metrics.get("macro_bleu") is not None:
        print(f"  [UTILITY] BLEU={metrics['macro_bleu']:.4f}")

    # RAT-Bench risk metrics
    if metrics.get("r_succ_rate") is not None:
        print(f"  [RISK]    R_succ={metrics['r_succ_rate']:.4f} (θ=0.2)  avg_risk={metrics.get('avg_risk',0):.4f}")

    print(f"  n_docs={metrics.get('n_documents',0)}  "
          f"GT={metrics.get('total_ground_truth',0)}  "
          f"preds={metrics.get('total_predictions',0)}  "
          f"leaks={metrics.get('total_leaks',0)}")


# ---------------------------------------------------------------------------
# Benchmark subcommand
# ---------------------------------------------------------------------------

def _run_benchmark(args: argparse.Namespace) -> int:
    ensure_pipegraph_importable()
    _create_pipeline, create_initial_state = load_pipegraph()
    pipeline = _create_pipeline()
    root = project_root()
    reports_dir = getattr(args, "out", None) or _default_reports_dir()
    runs_dir = _default_runs_dir()
    os.makedirs(reports_dir, exist_ok=True)
    ts = _utc_ts()

    datasets: List[str] = list(args.datasets)
    levels: List[int] = list(getattr(args, "levels", [1, 2, 3]))
    language: str = getattr(args, "language", "english")
    limit: Optional[int] = getattr(args, "limit", None)
    split: str = getattr(args, "split", "test")
    strict: bool = getattr(args, "strict", False)
    skip_risk: bool = getattr(args, "skip_risk", False)
    save_runs: bool = getattr(args, "save_runs", False)

    all_results: Dict[str, Any] = {"created_at": utc_now_iso(), "datasets": {}}

    for dataset in datasets:
        if dataset == "ratbench":
            for level in levels:
                key = f"ratbench/{language}/L{level}"
                result = _benchmark_one_dataset(
                    dataset=dataset, level=level, language=language,
                    limit=limit, split=split, args=args,
                    pipeline=pipeline, create_initial_state=create_initial_state,
                    root=root, reports_dir=reports_dir, runs_dir=runs_dir,
                    ts=ts, save_runs=save_runs, skip_risk=skip_risk,
                )
                all_results["datasets"][key] = result
                if "metrics" in result:
                    _print_detection_summary(key, result["metrics"], strict=strict)
                    if result["metrics"].get("by_difficulty"):
                        for diff, dm in result["metrics"]["by_difficulty"].items():
                            print(f"    L{diff}: P={dm.get('macro_precision',0):.4f}  "
                                  f"R={dm.get('macro_recall',0):.4f}  "
                                  f"F2={dm.get('macro_f2',0):.4f}")
        else:
            result = _benchmark_one_dataset(
                dataset=dataset, level=None, language=language,
                limit=limit, split=split, args=args,
                pipeline=pipeline, create_initial_state=create_initial_state,
                root=root, reports_dir=reports_dir, runs_dir=runs_dir,
                ts=ts, save_runs=save_runs, skip_risk=skip_risk,
            )
            all_results["datasets"][dataset] = result
            if "metrics" in result:
                _print_detection_summary(dataset, result["metrics"], strict=strict)

    summary_path = os.path.join(reports_dir, f"benchmark_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nRapport complet : {summary_path}")
    return 0


def _benchmark_one_dataset(
    *,
    dataset: str,
    level: Optional[int],
    language: str,
    limit: Optional[int],
    split: str,
    args: argparse.Namespace,
    pipeline: Any,
    create_initial_state: Any,
    root: str,
    reports_dir: str,
    runs_dir: str,
    ts: str,
    save_runs: bool,
    skip_risk: bool,
) -> Dict[str, Any]:
    try:
        docs, dataset_name = load_benchmark_docs(
            dataset=dataset,
            project_root=root,
            limit=limit,
            level=level,
            language=language,
            split=split,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    if not docs:
        return {"status": "empty", "error": "No documents loaded"}

    config = _build_config(
        dataset=dataset,
        profile=getattr(args, "profile", "auto"),
        eval_mode=getattr(args, "eval_mode", "both"),
        masking_mode=getattr(args, "masking_mode", "benchmark"),
        no_llm=getattr(args, "no_llm", False),
        llm_provider=getattr(args, "llm_provider", None),
        llm_model=getattr(args, "llm_model", None),
    )
    allowed_labels = get_allowed_labels(dataset)

    doc_workers: int = getattr(args, "doc_workers", 1)
    t0 = time.time()
    report = build_report(docs, pipeline, create_initial_state, config=config,
                          allowed_labels=allowed_labels, max_workers=doc_workers)
    elapsed = round(time.time() - t0, 2)

    # Attach RAT-Bench metadata
    if dataset == "ratbench":
        try:
            from eval.ratbench_loader import evaluate_text_leaks, get_ratbench_metadata, load_ratbench_profiles
            profiles = load_ratbench_profiles(language=language, level=level, limit=limit)
            for doc, prof_data in zip(report, profiles):
                doc["ratbench_metadata"] = get_ratbench_metadata(prof_data.get("profile", prof_data))
                doc["text_leak_analysis"] = evaluate_text_leaks(
                    doc.get("full_text", ""),
                    doc.get("anonymized_text", ""),
                    prof_data.get("profile", prof_data),
                )
        except Exception:
            pass

    # Aggregate
    if dataset == "ratbench":
        metrics = aggregate_ratbench_metrics(report)
        try:
            leak_summary = compute_leak_summary(report)
            metrics.update(leak_summary)
            metrics["by_difficulty"] = metrics_by_difficulty(report)
            metrics["by_scenario"] = metrics_by_scenario(report)
            metrics["direct_id_detection_rates"] = direct_id_detection_rate(report)
        except Exception:
            pass
    else:
        metrics = aggregate_document_metrics(report)

    # Label metrics
    try:
        from eval.core.metrics import label_metrics
        metrics["label_metrics"] = label_metrics(report)
    except Exception:
        pass

    # Save report
    safe_name = dataset_name.replace("/", "_")
    report_path = os.path.join(reports_dir, f"report_{safe_name}_{ts}.json")
    meta = build_report_meta(dataset_name=dataset_name, limit=limit, config=config)
    save_report_payload(report_path, meta=meta, data=report)

    if save_runs:
        os.makedirs(runs_dir, exist_ok=True)
        save_run(runs_dir, meta=meta, data=report)

    return {
        "status": "ok",
        "dataset_name": dataset_name,
        "n_documents": len(report),
        "elapsed_s": elapsed,
        "report_path": report_path,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Ablation subcommand
# ---------------------------------------------------------------------------

def _run_ablation(args: argparse.Namespace) -> int:
    if not SUITES:
        print("[ERROR] run_ablation.py introuvable — impossible de charger les suites.", file=sys.stderr)
        return 1

    suite_name: str = getattr(args, "suite", "nodes")
    dataset: str = getattr(args, "dataset", "tab")
    limit: Optional[int] = getattr(args, "limit", 50)
    split: str = getattr(args, "split", "test")
    language: str = getattr(args, "language", "english")
    level: Optional[int] = getattr(args, "level", None)
    parallel: Optional[int] = getattr(args, "parallel_configs", None)
    save_runs: bool = getattr(args, "save_runs", False)
    reports_dir = getattr(args, "out", None) or _default_reports_dir()
    runs_dir = _default_runs_dir()
    os.makedirs(reports_dir, exist_ok=True)

    if suite_name == "list":
        print("Suites disponibles:", ", ".join(SUITES.keys()))
        return 0

    if suite_name == "custom":
        custom_path = getattr(args, "custom_config", None)
        if not custom_path:
            print("[ERROR] --custom-config requis pour la suite 'custom'", file=sys.stderr)
            return 1
        with open(custom_path, "r", encoding="utf-8") as fh:
            grid = json.load(fh)
    elif suite_name == "full":
        grid = [cfg for suite in SUITES.values() for cfg in suite]
    else:
        grid = SUITES.get(suite_name)
        if grid is None:
            print(f"[ERROR] Suite inconnue: {suite_name!r}. Disponibles: {', '.join(SUITES.keys())}", file=sys.stderr)
            return 1

    ensure_pipegraph_importable()
    _create_pipeline, create_initial_state = load_pipegraph()
    pipeline = _create_pipeline()
    root = project_root()

    docs, dataset_name = load_benchmark_docs(
        dataset=dataset,
        project_root=root,
        limit=limit,
        level=level,
        language=language,
        split=split,
    )
    if not docs:
        print(f"[ERROR] Aucun document chargé pour {dataset}", file=sys.stderr)
        return 1

    allowed_labels = get_allowed_labels(dataset)
    profile = getattr(args, "profile", "auto")
    eval_mode = getattr(args, "eval_mode", "both")
    masking_mode = getattr(args, "masking_mode", "benchmark")

    results: List[Dict[str, Any]] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(cfg_entry: Dict[str, Any]) -> Dict[str, Any]:
        name = cfg_entry.get("name", "?")
        desc = cfg_entry.get("description", "")
        raw_cfg = dict(cfg_entry.get("config", {}))
        runtime_cfg = _augment_dataset_runtime_config(
            dataset, raw_cfg, profile=profile, eval_mode=eval_mode, masking_mode=masking_mode,
        )
        t0 = time.time()
        try:
            rep = build_report(docs, pipeline, create_initial_state, config=runtime_cfg, allowed_labels=allowed_labels)
        except Exception as exc:
            return {"name": name, "description": desc, "error": str(exc)}
        elapsed = round(time.time() - t0, 2)
        agg = _agg(rep)
        row: Dict[str, Any] = {"name": name, "description": desc, "elapsed_s": elapsed, **agg}
        # Add strict metrics if available
        m_agg = aggregate_document_metrics(rep)
        if m_agg.get("macro_strict_precision") is not None:
            row["strict_precision"] = m_agg["macro_strict_precision"]
            row["strict_recall"] = m_agg["macro_strict_recall"]
            row["strict_f2"] = m_agg.get("macro_strict_f2")
        if m_agg.get("macro_bleu") is not None:
            row["bleu"] = m_agg["macro_bleu"]
        if save_runs:
            os.makedirs(runs_dir, exist_ok=True)
            meta = build_report_meta(dataset_name=dataset_name, limit=limit, config=runtime_cfg,
                                     run_name=name)
            save_run(runs_dir, meta=meta, data=rep, run_name=name)
        return row

    workers = parallel if isinstance(parallel, int) and parallel > 1 else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, cfg): cfg for cfg in grid}
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for cfg_entry in grid:
            results.append(_run_one(cfg_entry))

    _print_table(results)

    ts = _utc_ts()
    summary_path = os.path.join(reports_dir, f"ablation_{suite_name}_{dataset}_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"suite": suite_name, "dataset": dataset_name, "results": results}, fh,
                  ensure_ascii=False, indent=2, default=str)
    print(f"Rapport ablation : {summary_path}")
    return 0


# ---------------------------------------------------------------------------
# Dataset subcommand (standalone per-dataset evaluation)
# ---------------------------------------------------------------------------

def _run_dataset(args: argparse.Namespace) -> int:
    ensure_pipegraph_importable()
    _create_pipeline, create_initial_state = load_pipegraph()
    pipeline = _create_pipeline()
    root = project_root()
    reports_dir = getattr(args, "out", None) or _default_reports_dir()
    runs_dir = _default_runs_dir()
    os.makedirs(reports_dir, exist_ok=True)

    dataset: str = getattr(args, "dataset", "tab")
    split: str = getattr(args, "split", "test")
    language: str = getattr(args, "language", "english")
    level: Optional[int] = getattr(args, "level", None)
    limit: Optional[int] = getattr(args, "limit", None)
    strict: bool = getattr(args, "strict", False)
    save_run_flag: bool = getattr(args, "save_run", False)

    docs, dataset_name = load_benchmark_docs(
        dataset=dataset,
        project_root=root,
        limit=limit,
        level=level,
        language=language,
        split=split,
    )
    if not docs:
        print(f"[ERROR] Aucun document chargé pour {dataset}", file=sys.stderr)
        return 1

    config = _build_config(
        dataset=dataset,
        profile=getattr(args, "profile", "auto"),
        eval_mode=getattr(args, "eval_mode", "both"),
        masking_mode=getattr(args, "masking_mode", "benchmark"),
        no_llm=getattr(args, "no_llm", False),
    )
    allowed_labels = get_allowed_labels(dataset)

    doc_workers: int = getattr(args, "doc_workers", 1)
    t0 = time.time()
    report = build_report(docs, pipeline, create_initial_state, config=config,
                          allowed_labels=allowed_labels, max_workers=doc_workers)
    elapsed = round(time.time() - t0, 2)

    metrics = aggregate_document_metrics(report)
    _print_detection_summary(dataset_name, metrics, strict=strict)
    print(f"  Temps total : {elapsed}s  ({elapsed/max(len(report),1):.2f}s/doc)")

    ts = _utc_ts()
    safe_name = dataset_name.replace("/", "_")
    report_path = os.path.join(reports_dir, f"report_{safe_name}_{ts}.json")
    meta = build_report_meta(dataset_name=dataset_name, limit=limit, config=config)
    save_report_payload(report_path, meta=meta, data=report)
    print(f"\nRapport : {report_path}")

    if save_run_flag:
        os.makedirs(runs_dir, exist_ok=True)
        save_run(runs_dir, meta=meta, data=report)

    return 0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _common_llm_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--no-llm", action="store_true", help="Désactiver tous les modules LLM")
    grp.add_argument("--with-llm", action="store_true", help="Forcer LLM activé (défaut selon config)")
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)


def _common_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="auto")
    parser.add_argument("--eval-mode", choices=EVAL_MODE_CHOICES, default="both")
    parser.add_argument("--masking-mode", choices=MASKING_MODE_CHOICES, default="benchmark")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- benchmark ----
    bm = sub.add_parser("benchmark", help="Évaluation multi-dataset")
    bm.add_argument("--datasets", nargs="+",
                    choices=ALL_DATASETS,
                    default=["tab", "dbbio", "anonymization"])
    bm.add_argument("--levels", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    bm.add_argument("--language", choices=["english", "mandarin", "spanish"], default="english")
    bm.add_argument("--limit", type=int, default=None)
    bm.add_argument("--split", choices=["test", "dev", "train"], default="test")
    bm.add_argument("--skip-risk", action="store_true", help="Ignorer l'axe risque RAT-Bench")
    bm.add_argument("--save-runs", action="store_true")
    bm.add_argument("--strict", action="store_true", help="Afficher les métriques strict-match en priorité")
    bm.add_argument("--doc-workers", type=int, default=1,
                    help="Workers parallèles pour le traitement des documents (défaut: 1, recommandé local GPU)")
    bm.add_argument("--out", default=None, help="Répertoire de sortie des rapports")
    _common_llm_args(bm)
    _common_profile_args(bm)

    # ---- ablation ----
    ab = sub.add_parser("ablation", help="Grille d'ablation sur un dataset")
    ab.add_argument("--dataset", choices=ALL_DATASETS, default="tab")
    ab.add_argument("--suite",
                    choices=list(SUITES.keys()) + ["full", "custom", "list"],
                    default="nodes")
    ab.add_argument("--custom-config", default=None, help="Chemin JSON pour suite=custom")
    ab.add_argument("--limit", type=int, default=50)
    ab.add_argument("--split", choices=["test", "dev", "train"], default="test")
    ab.add_argument("--language", choices=["english", "mandarin", "spanish"], default="english")
    ab.add_argument("--level", type=int, choices=[1, 2, 3], default=None)
    ab.add_argument("--parallel-configs", type=int, default=1,
                    help="Nombre de configs à évaluer en parallèle (défaut: 1)")
    ab.add_argument("--doc-workers", type=int, default=1,
                    help="Workers parallèles pour le traitement des documents (défaut: 1)")
    ab.add_argument("--save-runs", action="store_true")
    ab.add_argument("--out", default=None)
    _common_llm_args(ab)
    _common_profile_args(ab)

    # ---- dataset ----
    ds = sub.add_parser("dataset", help="Évaluation standalone d'un seul dataset")
    ds.add_argument("--dataset", choices=ALL_DATASETS, required=True)
    ds.add_argument("--split", choices=["test", "dev", "train"], default="test")
    ds.add_argument("--language", choices=["english", "mandarin", "spanish"], default="english")
    ds.add_argument("--level", type=int, choices=[1, 2, 3], default=None)
    ds.add_argument("--limit", type=int, default=None)
    ds.add_argument("--save-run", action="store_true")
    ds.add_argument("--strict", action="store_true")
    ds.add_argument("--doc-workers", type=int, default=1,
                    help="Workers parallèles pour le traitement des documents (défaut: 1)")
    ds.add_argument("--out", default=None)
    _common_llm_args(ds)
    _common_profile_args(ds)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cmd = args.command
    if cmd == "benchmark":
        return _run_benchmark(args)
    if cmd == "ablation":
        return _run_ablation(args)
    if cmd == "dataset":
        return _run_dataset(args)
    print(f"[ERROR] Sous-commande inconnue: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
