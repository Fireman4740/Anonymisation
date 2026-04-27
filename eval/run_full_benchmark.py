#!/usr/bin/env python3
"""
run_full_benchmark.py — Évaluation complète unifiée du pipeline PipeGraph
==========================================================================

Ce script évalue le pipeline d'anonymisation sur **tous les axes** :

  1. DÉTECTION D'ENTITÉS (span-level)
     - Precision / Recall / F2 (overlap partiel)
     - Métriques par type d'entité
     → sur TAB, DB-bio, Anonymization dataset, RAT-Bench (L1/L2/L3)

  2. PROTECTION DES ATTRIBUTS (text-leak analysis)
     - Taux de fuite direct / indirect dans le texte anonymisé
     → sur RAT-Bench uniquement (profils structurés)

  3. RISQUE DE RÉ-IDENTIFICATION (algorithme Staab / RAT-Bench paper)
     - Attaque LLM per-attribute (Staab et al.)
     - k-anonymity via population PUMS
     - Risk R(x,t,T) = 1/k + detection directe
     → sur RAT-Bench (nécessite OPENROUTER_API_KEY)

Usage :
    # Évaluation complète (détection + leaks + risque)
    python eval/run_full_benchmark.py

    # Détection uniquement, rapide
    python eval/run_full_benchmark.py --skip-risk --limit 50

    # Un seul dataset
    python eval/run_full_benchmark.py --datasets tab

    # RAT-Bench L1 uniquement avec risque
    python eval/run_full_benchmark.py --datasets ratbench --levels 1

    # Sans LLM pipeline (détection regex+NER seulement)
    python eval/run_full_benchmark.py --no-llm

    # Sauvegarde des runs pour Streamlit
    python eval/run_full_benchmark.py --save-runs
    
    conda activate ano && cd /mnt/f/IA/Anonymisation && python eval/run_full_benchmark.py --datasets anonymization tab dbbio --skip-risk --limit 50 2>&1
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
# Bootstrap : s'assurer que le projet est importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.core.bootstrap import ensure_pipegraph_importable, load_pipegraph, project_root
from eval.core.config import build_runtime_config
from eval.core.datasets import get_allowed_labels, load_benchmark_docs
from eval.core.ratbench import (
    aggregate_ratbench_metrics,
    build_ratbench_report,
    build_ratbench_result,
    compute_leak_summary,
    direct_id_detection_rate,
    metrics_by_difficulty,
    metrics_by_scenario,
)
from eval.core.reporting import (
    aggregate_document_metrics,
    build_report_meta,
    build_report_payload,
    save_report_payload,
)
from eval.pipegraph_eval_local import build_report
from eval.ratbench_loader import load_ratbench_profiles
from eval.run_store import save_run, utc_now_iso


# ═══════════════════════════════════════════════════════════════════════════
# Helpers d'affichage
# ═══════════════════════════════════════════════════════════════════════════

_CYAN = "\033[0;36m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[1;33m"
_RED = "\033[0;31m"
_BOLD = "\033[1m"
_NC = "\033[0m"


def _section(title: str) -> None:
    print(f"\n{_CYAN}{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}{_NC}\n")


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✅ {msg}{_NC}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠️  {msg}{_NC}")


def _err(msg: str) -> None:
    print(f"  {_RED}❌ {msg}{_NC}")


def _info(msg: str) -> None:
    print(f"  {msg}")


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _fmt_metric(val: float, width: int = 6) -> str:
    return f"{val:.4f}".rjust(width)


# ═══════════════════════════════════════════════════════════════════════════
# Structures de résultats
# ═══════════════════════════════════════════════════════════════════════════

class BenchmarkResult:
    """Résultat pour un dataset/axe d'évaluation."""

    def __init__(
        self,
        dataset_name: str,
        axis: str,
        metrics: Dict[str, Any],
        elapsed_s: float,
        report: Optional[List[Dict[str, Any]]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self.axis = axis
        self.metrics = metrics
        self.elapsed_s = elapsed_s
        self.report = report or []
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "axis": self.axis,
            "metrics": self.metrics,
            "elapsed_s": round(self.elapsed_s, 1),
            **self.extra,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Métriques par type d'entité
# ═══════════════════════════════════════════════════════════════════════════

def compute_label_metrics(report: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Agrège TP/FP/FN par label depuis les rapports per-doc pour obtenir
    Precision/Recall/F1 par type d'entité."""
    label_tp: Dict[str, int] = {}
    label_fp: Dict[str, int] = {}
    label_fn: Dict[str, int] = {}
    label_exact_tp: Dict[str, int] = {}
    label_exact_fn: Dict[str, int] = {}

    for doc in report:
        for label, count in doc.get("tp_by_label", {}).items():
            label_tp[label] = label_tp.get(label, 0) + count
        for label, count in doc.get("fp_by_label", {}).items():
            label_fp[label] = label_fp.get(label, 0) + count
        for label, count in doc.get("fn_by_label", {}).items():
            label_fn[label] = label_fn.get(label, 0) + count
        for label, count in doc.get("exact_tp_by_label", {}).items():
            label_exact_tp[label] = label_exact_tp.get(label, 0) + count
        for label, count in doc.get("exact_fn_by_label", {}).items():
            label_exact_fn[label] = label_exact_fn.get(label, 0) + count

    all_labels = sorted(
        set(label_tp) | set(label_fp) | set(label_fn) | set(label_exact_tp) | set(label_exact_fn)
    )
    result: Dict[str, Dict[str, float]] = {}
    for label in all_labels:
        tp = label_tp.get(label, 0)
        fp = label_fp.get(label, 0)
        fn = label_fn.get(label, 0)
        exact_tp = label_exact_tp.get(label, 0)
        exact_fn = label_exact_fn.get(label, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        exact_recall = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "exact_recall": round(exact_recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════
# AXE 1 — Détection d'entités (span-level)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_entity_detection(
    dataset: str,
    pipeline: Any,
    create_initial_state: Any,
    config: Dict[str, Any],
    *,
    limit: int,
    split: str = "test",
    language: str = "english",
    level: Optional[int] = None,
) -> BenchmarkResult:
    """Évalue la détection d'entités (Precision/Recall/F2) sur un dataset."""

    _info(f"📥 Chargement du dataset '{dataset}'...")
    runtime_config = dict(config)

    if dataset == "conll2003":
        runtime_config.setdefault("entity_profile", "news_ner")
        runtime_config.setdefault("gliner_label_profile", "news_ner")
    else:
        runtime_config.setdefault("entity_profile", "pii")
        runtime_config.setdefault("gliner_label_profile", "pii")

    if dataset == "ratbench":
        # Pour RAT-Bench, on passe par le loader dédié
        profiles = load_ratbench_profiles(language=language, level=level, limit=limit)
        from eval.ratbench_loader import build_docs_from_ratbench
        docs = build_docs_from_ratbench(language=language, level=level, limit=limit)
        level_str = f"L{level}" if level else "all"
        dataset_name = f"RAT-Bench/{language}/{level_str}"
    else:
        docs, dataset_name = load_benchmark_docs(
            dataset=dataset,
            project_root=project_root(),
            limit=limit,
            split=split,
        )
        profiles = None

    # Label-scope filtering: drop predictions whose label doesn't exist
    # in the dataset's annotation scope (avoids inflated FP counts).
    ds_allowed = get_allowed_labels(dataset)
    if ds_allowed is not None:
        _info(f"   → Filtrage des prédictions : labels autorisés = {sorted(ds_allowed)}")

    _info(f"   → {len(docs)} documents chargés pour '{dataset_name}'")

    t0 = time.time()

    if dataset == "ratbench" and profiles is not None:
        # Utilise le builder RAT-Bench enrichi (metadata + leak analysis)
        report = build_ratbench_report(
            docs, profiles, pipeline, create_initial_state,
            config=config,
        )
    else:
        report = build_report(
            docs, pipeline, create_initial_state,
            config=runtime_config,
            allowed_labels=ds_allowed,
        )

    elapsed = time.time() - t0

    # Métriques agrégées
    if dataset == "ratbench":
        agg = aggregate_ratbench_metrics(report)
    else:
        agg = aggregate_document_metrics(report)

    # Métriques par type d'entité
    label_metrics = compute_label_metrics(report)

    metrics: Dict[str, Any] = {
        **agg,
        "label_metrics": label_metrics,
    }

    extra: Dict[str, Any] = {}

    # Enrichissements RAT-Bench
    if dataset == "ratbench":
        metrics["by_difficulty"] = metrics_by_difficulty(report)
        metrics["by_scenario"] = metrics_by_scenario(report)
        metrics["direct_id_detection_rates"] = direct_id_detection_rate(report)
        metrics["leak_summary"] = compute_leak_summary(report)
        extra["language"] = language
        extra["level"] = level

    return BenchmarkResult(
        dataset_name=dataset_name,
        axis="entity_detection",
        metrics=metrics,
        elapsed_s=elapsed,
        report=report,
        extra=extra,
    )


# ═══════════════════════════════════════════════════════════════════════════
# AXE 2 — Protection des attributs (text-leak analysis, RAT-Bench only)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_attribute_protection(
    report: List[Dict[str, Any]],
    dataset_name: str,
) -> BenchmarkResult:
    """Analyse les fuites d'attributs dans les textes anonymisés.

    Utilise les résultats de `text_leak_analysis` déjà attachés par
    `build_ratbench_report` — pas de re-exécution du pipeline.
    """
    t0 = time.time()

    leak_entries = [
        doc.get("text_leak_analysis", {})
        for doc in report
        if doc.get("text_leak_analysis")
    ]

    if not leak_entries:
        return BenchmarkResult(
            dataset_name=dataset_name,
            axis="attribute_protection",
            metrics={"error": "no_leak_data"},
            elapsed_s=0.0,
        )

    n = len(leak_entries)

    # Taux de fuite agrégés
    avg_leak_rate = sum(e.get("leak_rate", 0.0) for e in leak_entries) / n
    avg_direct_leak = sum(e.get("direct_leak_rate", 0.0) for e in leak_entries) / n
    avg_indirect_leak = sum(e.get("indirect_leak_rate", 0.0) for e in leak_entries) / n
    total_attrs = sum(e.get("n_total_attributes", 0) for e in leak_entries)
    total_leaked = sum(e.get("n_leaked", 0) for e in leak_entries)
    total_protected = sum(e.get("n_protected", 0) for e in leak_entries)

    # Taux de fuite par catégorie d'attribut
    attr_leak_counts: Dict[str, Dict[str, int]] = {}
    for entry in leak_entries:
        for attr_name, attr_info in entry.get("details", {}).items():
            stats = attr_leak_counts.setdefault(attr_name, {"total": 0, "leaked": 0})
            stats["total"] += 1
            if attr_info.get("leaked"):
                stats["leaked"] += 1

    per_attribute: Dict[str, Dict[str, Any]] = {}
    for attr_name, stats in sorted(attr_leak_counts.items()):
        total = stats["total"]
        leaked = stats["leaked"]
        per_attribute[attr_name] = {
            "total": total,
            "leaked": leaked,
            "protection_rate": round(1 - leaked / total, 4) if total > 0 else 1.0,
            "leak_rate": round(leaked / total, 4) if total > 0 else 0.0,
        }

    # Taux de protection = 1 - taux de fuite (plus intuitif)
    metrics: Dict[str, Any] = {
        "n_documents": n,
        "avg_leak_rate": round(avg_leak_rate, 4),
        "avg_protection_rate": round(1 - avg_leak_rate, 4),
        "avg_direct_leak_rate": round(avg_direct_leak, 4),
        "avg_direct_protection_rate": round(1 - avg_direct_leak, 4),
        "avg_indirect_leak_rate": round(avg_indirect_leak, 4),
        "avg_indirect_protection_rate": round(1 - avg_indirect_leak, 4),
        "total_attributes": total_attrs,
        "total_leaked": total_leaked,
        "total_protected": total_protected,
        "per_attribute": per_attribute,
    }

    return BenchmarkResult(
        dataset_name=dataset_name,
        axis="attribute_protection",
        metrics=metrics,
        elapsed_s=time.time() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# AXE 3 — Risque de ré-identification (RAT-Bench paper Algorithm 1-2)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_reidentification_risk(
    pipeline: Any,
    create_initial_state: Any,
    config: Dict[str, Any],
    report: List[Dict[str, Any]],
    *,
    language: str = "english",
    level: Optional[int] = None,
    limit: int = 50,
) -> BenchmarkResult:
    """Évalue le risque de ré-identification selon l'algorithme RAT-Bench.

    Algorithme (Staab et al.) :
      1. Anonymise le texte (réutilise les résultats de l'axe 1 si dispo)
      2. Attaque LLM : infère les attributs indirects (per-attribute Staab prompting)
      3. Vérifie les fuites directes (nom, SSN, email…)
      4. Calcule la classe d'équivalence k sur la population PUMS
      5. Risk R = 1/k (ou 1.0 si fuite directe)

    Nécessite OPENROUTER_API_KEY dans l'environnement.
    """
    from eval.cli.evaluate_ratbench_risk import (
        evaluate_on_ratbench,
        openrouter_llm_attacker,
        RatBenchRecord,
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        _warn("OPENROUTER_API_KEY non définie — évaluation du risque de ré-identification ignorée")
        return BenchmarkResult(
            dataset_name=f"RAT-Bench/{language}",
            axis="reidentification_risk",
            metrics={"error": "OPENROUTER_API_KEY not set"},
            elapsed_s=0.0,
        )

    _info("📥 Chargement des profils RAT-Bench pour l'évaluation du risque...")
    profiles = load_ratbench_profiles(language=language, level=level, limit=limit)
    _info(f"   → {len(profiles)} profils chargés")

    # Charger la population PUMS pour le calcul de k-anonymity
    from eval.cli.evaluate_ratbench_risk import _load_population
    population_df, is_real_pums = _load_population(profiles)
    pop_source = "PUMS" if is_real_pums else "dataset_profiles"
    _info(f"   → Population : {len(population_df):,} records ({pop_source})")

    if not is_real_pums:
        _warn("Population PUMS non disponible — les estimations de risque seront IMPRÉCISES")

    # Construire les records
    test_records: List[RatBenchRecord] = []
    for p in profiles:
        test_records.append(RatBenchRecord(
            id=str(p.get("id", "")),
            text=p["text"],
            profile=p.get("profile", p),
            difficulty=str(p.get("difficulty", "1")),
            scenario=p.get("scenario", "unknown"),
        ))

    # Réutiliser les textes déjà anonymisés de l'axe 1
    pre_anonymized_texts: Dict[str, str] = {}
    if report:
        for doc in report:
            orig = doc.get("full_text", "")
            anon = doc.get("anonymized_text", "")
            if orig and anon and orig != anon:
                pre_anonymized_texts[orig] = anon

    n_reused = sum(1 for r in test_records if r.text in pre_anonymized_texts)
    if n_reused:
        _info(f"   → Réutilisation de {n_reused}/{len(test_records)} textes déjà anonymisés")

    from eval.pipegraph_eval_local import run_pipegraph_on_text

    def my_anonymizer(text: str) -> str:
        state = run_pipegraph_on_text(pipeline, create_initial_state, text, config)
        if isinstance(state, dict):
            return state.get("text", text)
        return getattr(state, "text", text)

    t0 = time.time()

    risk_metrics, detailed_df = evaluate_on_ratbench(
        test_records,
        my_anonymizer,
        openrouter_llm_attacker,
        population_df,
        pre_anonymized_texts=pre_anonymized_texts,
        is_real_pums=is_real_pums,
    )

    elapsed = time.time() - t0

    level_str = f"L{level}" if level else "all"
    risk_metrics["population_source"] = pop_source
    risk_metrics["population_size"] = len(population_df)

    # Convertir le DataFrame en enregistrements sérialisables
    detailed_results = detailed_df.to_dict(orient="records") if not detailed_df.empty else []

    return BenchmarkResult(
        dataset_name=f"RAT-Bench/{language}/{level_str}",
        axis="reidentification_risk",
        metrics=risk_metrics,
        elapsed_s=elapsed,
        extra={"detailed_results": detailed_results},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Affichage des résultats
# ═══════════════════════════════════════════════════════════════════════════

def print_entity_detection_results(result: BenchmarkResult) -> None:
    """Affiche les résultats de détection d'entités."""
    m = result.metrics
    _section(f"DÉTECTION D'ENTITÉS — {result.dataset_name}")

    _info(f"  {'Documents':20s} : {m.get('n_documents', 0)}")
    _info(f"  {'Durée':20s} : {result.elapsed_s:.0f}s")
    print()
    _info(f"  {_BOLD}{'Métrique':20s}   {'Valeur':>10s}{_NC}")
    _info(f"  {'─' * 35}")
    _info(f"  {'Macro Precision':20s}   {_fmt_metric(m.get('macro_precision', 0.0)):>10s}")
    _info(f"  {'Macro Recall':20s}   {_fmt_metric(m.get('macro_recall', 0.0)):>10s}")
    _info(f"  {'Macro Recall exact':20s}   {_fmt_metric(m.get('macro_exact_label_recall', 0.0)):>10s}")
    _info(f"  {'Macro F2':20s}   {_fmt_metric(m.get('macro_f2', 0.0)):>10s}")
    _info(f"  {'Micro Precision':20s}   {_fmt_metric(m.get('micro_precision', 0.0)):>10s}")
    _info(f"  {'Micro Recall':20s}   {_fmt_metric(m.get('micro_recall', 0.0)):>10s}")
    _info(f"  {'Micro Recall exact':20s}   {_fmt_metric(m.get('micro_exact_label_recall', 0.0)):>10s}")
    _info(f"  {'Micro F2':20s}   {_fmt_metric(m.get('micro_f2', 0.0)):>10s}")
    _info(f"  {'Total prédictions':20s}   {m.get('total_predictions', 0):>10d}")
    _info(f"  {'Total ground-truth':20s}   {m.get('total_ground_truth', 0):>10d}")
    _info(f"  {'Total fuites (leaks)':20s}   {m.get('total_leaks', 0):>10d}")

    # Métriques par type d'entité
    label_metrics = m.get("label_metrics", {})
    if label_metrics:
        print()
        header_label = "Type d'entité"
        _info(f"  {_BOLD}{header_label:20s}  {'P':>7s}  {'R':>7s}  {'R_exact':>7s}  {'F1':>7s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  {'Support':>7s}{_NC}")
        _info(f"  {'─' * 85}")
        for label, lm in sorted(label_metrics.items(), key=lambda x: -x[1]["support"]):
            _info(
                f"  {label:20s}"
                f"  {_fmt_metric(lm['precision'], 7)}"
                f"  {_fmt_metric(lm['recall'], 7)}"
                f"  {_fmt_metric(lm.get('exact_recall', 0.0), 7)}"
                f"  {_fmt_metric(lm['f1'], 7)}"
                f"  {lm['tp']:5d}"
                f"  {lm['fp']:5d}"
                f"  {lm['fn']:5d}"
                f"  {lm['support']:7d}"
            )

    # RAT-Bench : breakdown par difficulté
    by_diff = m.get("by_difficulty", {})
    if by_diff:
        print()
        _info(f"  {_BOLD}Par niveau de difficulté :{_NC}")
        _info(f"  {'Niveau':10s}  {'Docs':>5s}  {'P':>7s}  {'R':>7s}  {'F2':>7s}  {'Leaks':>6s}")
        _info(f"  {'─' * 50}")
        for level_key, dm in sorted(by_diff.items()):
            _info(
                f"  L{str(level_key):9s}"
                f"  {dm.get('n_documents', 0):5d}"
                f"  {_fmt_metric(dm.get('macro_precision', 0), 7)}"
                f"  {_fmt_metric(dm.get('macro_recall', 0), 7)}"
                f"  {_fmt_metric(dm.get('macro_f2', 0), 7)}"
                f"  {dm.get('total_leaks', 0):6d}"
            )

    # RAT-Bench : breakdown par scénario
    by_scene = m.get("by_scenario", {})
    if by_scene:
        print()
        _info(f"  {_BOLD}Par scénario :{_NC}")
        _info(f"  {'Scénario':30s}  {'Docs':>5s}  {'P':>7s}  {'R':>7s}  {'F2':>7s}")
        _info(f"  {'─' * 60}")
        for sc, sm in sorted(by_scene.items()):
            _info(
                f"  {sc:30s}"
                f"  {sm.get('n_documents', 0):5d}"
                f"  {_fmt_metric(sm.get('macro_precision', 0), 7)}"
                f"  {_fmt_metric(sm.get('macro_recall', 0), 7)}"
                f"  {_fmt_metric(sm.get('macro_f2', 0), 7)}"
            )

    # RAT-Bench : taux de détection des identifiants directs
    direct_rates = m.get("direct_id_detection_rates", {})
    if direct_rates:
        print()
        _info(f"  {_BOLD}Taux de détection des identifiants directs :{_NC}")
        _info(f"  {'Type':20s}  {'Total':>6s}  {'Détectés':>9s}  {'Taux':>8s}")
        _info(f"  {'─' * 50}")
        for id_type, dr in sorted(direct_rates.items()):
            _info(
                f"  {id_type:20s}"
                f"  {dr['total']:6d}"
                f"  {dr['detected']:9d}"
                f"  {_pct(dr['detection_rate']):>8s}"
            )

    # RAT-Bench : leak summary
    leak_sum = m.get("leak_summary", {})
    if leak_sum:
        print()
        _info(f"  {_BOLD}Analyse des fuites textuelles (text-leak) :{_NC}")
        _info(f"  {'Taux de fuite moyen':30s} : {_pct(leak_sum.get('avg_leak_rate', 0))}")
        _info(f"  {'Taux direct':30s} : {_pct(leak_sum.get('avg_direct_leak_rate', 0))}")
        _info(f"  {'Taux indirect':30s} : {_pct(leak_sum.get('avg_indirect_leak_rate', 0))}")


def print_attribute_protection_results(result: BenchmarkResult) -> None:
    """Affiche les résultats de protection des attributs."""
    m = result.metrics
    if m.get("error"):
        return

    _section(f"PROTECTION DES ATTRIBUTS — {result.dataset_name}")

    _info(f"  {'Documents':30s} : {m.get('n_documents', 0)}")
    _info(f"  {'Attributs total':30s} : {m.get('total_attributes', 0)}")
    _info(f"  {'Attributs protégés':30s} : {m.get('total_protected', 0)}")
    _info(f"  {'Attributs fuités':30s} : {m.get('total_leaked', 0)}")
    print()
    _info(f"  {_BOLD}{'Métrique':35s}   {'Valeur':>10s}{_NC}")
    _info(f"  {'─' * 50}")
    _info(f"  {'Taux de protection global':35s}   {_pct(m.get('avg_protection_rate', 0)):>10s}")
    _info(f"  {'Taux de fuite global':35s}   {_pct(m.get('avg_leak_rate', 0)):>10s}")
    _info(f"  {'Protection identifiants directs':35s}   {_pct(m.get('avg_direct_protection_rate', 0)):>10s}")
    _info(f"  {'Fuite identifiants directs':35s}   {_pct(m.get('avg_direct_leak_rate', 0)):>10s}")
    _info(f"  {'Protection identifiants indirects':35s}   {_pct(m.get('avg_indirect_protection_rate', 0)):>10s}")
    _info(f"  {'Fuite identifiants indirects':35s}   {_pct(m.get('avg_indirect_leak_rate', 0)):>10s}")

    per_attr = m.get("per_attribute", {})
    if per_attr:
        print()
        _info(f"  {_BOLD}{'Attribut':30s}  {'Total':>6s}  {'Fuité':>6s}  {'Protection':>10s}{_NC}")
        _info(f"  {'─' * 60}")
        for attr_name, stats in sorted(per_attr.items(), key=lambda x: -x[1]["leak_rate"]):
            _info(
                f"  {attr_name:30s}"
                f"  {stats['total']:6d}"
                f"  {stats['leaked']:6d}"
                f"  {_pct(stats['protection_rate']):>10s}"
            )


def print_reidentification_results(result: BenchmarkResult) -> None:
    """Affiche les résultats du risque de ré-identification."""
    m = result.metrics
    if m.get("error"):
        _warn(f"Ré-identification : {m['error']}")
        return

    _section(f"RISQUE DE RÉ-IDENTIFICATION — {result.dataset_name}")

    _info(f"  {'Durée':30s} : {result.elapsed_s:.0f}s")
    _info(f"  {'Population':30s} : {m.get('population_size', '?'):,} ({m.get('population_source', '?')})")
    print()
    _info(f"  {_BOLD}{'Métrique':40s}   {'Valeur':>10s}{_NC}")
    _info(f"  {'─' * 55}")
    _info(f"  {'Risque moyen R(x,t,T)':40s}   {_fmt_metric(m.get('avg_risk', 0)):>10s}")
    _info(f"  {'Fraction ré-identifiés (R≥1)':40s}   {_pct(m.get('frac_re_identified', 0)):>10s}")
    _info(f"  {'Fraction haut risque (R≥0.09)':40s}   {_pct(m.get('frac_high_risk_geq_0_09', 0)):>10s}")
    _info(f"  {'Attributs correctement inférés (moy)':40s}   {m.get('avg_correct_attrs', 0):>10.2f}")

    by_diff = m.get("by_difficulty", {})
    if by_diff:
        print()
        _info(f"  {_BOLD}Par niveau de difficulté :{_NC}")
        _info(f"  {'Niveau':10s}  {'Count':>6s}  {'Risque moy':>12s}")
        _info(f"  {'─' * 35}")
        for lev, stats in sorted(by_diff.items()):
            mean_r = stats.get("mean", 0)
            count = int(stats.get("count", 0))
            _info(f"  L{str(lev):9s}  {count:6d}  {mean_r:12.4f}")

    by_scenario = m.get("by_scenario", {})
    if by_scenario:
        print()
        _info(f"  {_BOLD}Par scénario :{_NC}")
        _info(f"  {'Scénario':30s}  {'Count':>6s}  {'Risque moy':>12s}")
        _info(f"  {'─' * 55}")
        for sc, stats in sorted(by_scenario.items()):
            mean_r = stats.get("mean", 0)
            count = int(stats.get("count", 0))
            _info(f"  {sc:30s}  {count:6d}  {mean_r:12.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Tableau récapitulatif
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(results: List[BenchmarkResult]) -> None:
    """Affiche un tableau récapitulatif de tous les résultats."""
    _section("RÉCAPITULATIF")

    # --- Entity detection ---
    detection_results = [r for r in results if r.axis == "entity_detection"]
    if detection_results:
        _info(f"  {_BOLD}{'Dataset':30s}  {'Pmacro':>7s}  {'Rmacro':>7s}  {'F2macro':>7s}  {'Pmicro':>7s}  {'Rmicro':>7s}  {'F2micro':>7s}  {'Leaks':>6s}  {'Durée':>8s}{_NC}")
        _info(f"  {'─' * 120}")
        for r in detection_results:
            m = r.metrics
            _info(
                f"  {r.dataset_name:30s}"
                f"  {_fmt_metric(m.get('macro_precision', 0), 7)}"
                f"  {_fmt_metric(m.get('macro_recall', 0), 7)}"
                f"  {_fmt_metric(m.get('macro_f2', 0), 7)}"
                f"  {_fmt_metric(m.get('micro_precision', 0), 7)}"
                f"  {_fmt_metric(m.get('micro_recall', 0), 7)}"
                f"  {_fmt_metric(m.get('micro_f2', 0), 7)}"
                f"  {m.get('total_leaks', 0):6d}"
                f"  {r.elapsed_s:7.0f}s"
            )

    # --- Attribute protection ---
    protection_results = [r for r in results if r.axis == "attribute_protection" and not r.metrics.get("error")]
    if protection_results:
        print()
        _info(f"  {_BOLD}{'Dataset':30s}  {'Protection':>10s}  {'Direct prot':>12s}  {'Indirect prot':>14s}{_NC}")
        _info(f"  {'─' * 75}")
        for r in protection_results:
            m = r.metrics
            _info(
                f"  {r.dataset_name:30s}"
                f"  {_pct(m.get('avg_protection_rate', 0)):>10s}"
                f"  {_pct(m.get('avg_direct_protection_rate', 0)):>12s}"
                f"  {_pct(m.get('avg_indirect_protection_rate', 0)):>14s}"
            )

    # --- Re-identification risk ---
    risk_results = [r for r in results if r.axis == "reidentification_risk" and not r.metrics.get("error")]
    if risk_results:
        print()
        _info(f"  {_BOLD}{'Dataset':30s}  {'Risque moy':>10s}  {'%Ré-ID':>8s}  {'%Haut risque':>12s}  {'Attrs inf':>10s}{_NC}")
        _info(f"  {'─' * 80}")
        for r in risk_results:
            m = r.metrics
            _info(
                f"  {r.dataset_name:30s}"
                f"  {_fmt_metric(m.get('avg_risk', 0)):>10s}"
                f"  {_pct(m.get('frac_re_identified', 0)):>8s}"
                f"  {_pct(m.get('frac_high_risk_geq_0_09', 0)):>12s}"
                f"  {m.get('avg_correct_attrs', 0):>10.2f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Sauvegarde
# ═══════════════════════════════════════════════════════════════════════════

def save_benchmark_report(
    results: List[BenchmarkResult],
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Sauvegarde le rapport complet du benchmark."""
    reports_dir = os.path.join(project_root(), "eval", "evaluation", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(reports_dir, f"full_benchmark_{ts}.json")

    payload = {
        "meta": {
            "created_at": utc_now_iso(),
            "pipeline": "pipegraph",
            "benchmark_type": "full",
            "config": config,
            "args": {
                "datasets": args.datasets,
                "limit": args.limit,
                "levels": args.levels,
                "language": args.language,
                "skip_risk": args.skip_risk,
                "no_llm": args.no_llm,
            },
        },
        "results": {},
    }

    for result in results:
        key = f"{result.dataset_name}_{result.axis}"
        entry = result.to_dict()
        # Ne pas inclure les rapports détaillés (trop volumineux)
        entry.pop("detailed_results", None)
        payload["results"][key] = entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    return out_path


def generate_llm_report(
    results: List[BenchmarkResult],
    config: Dict[str, Any],
    args: argparse.Namespace,
    elapsed_total: float,
) -> str:
    """Génère un rapport Markdown exploitable par un LLM.

    Le rapport est structuré pour qu'un LLM puisse analyser les forces et
    faiblesses du pipeline d'anonymisation, identifier les axes d'amélioration,
    et produire des recommandations.
    """
    lines: List[str] = []

    def h1(title: str) -> None:
        lines.append(f"\n# {title}\n")

    def h2(title: str) -> None:
        lines.append(f"\n## {title}\n")

    def h3(title: str) -> None:
        lines.append(f"\n### {title}\n")

    def p(text: str) -> None:
        lines.append(f"{text}\n")

    def bullet(text: str) -> None:
        lines.append(f"- {text}")

    def table_header(cols: List[str]) -> None:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    def table_row(vals: List[str]) -> None:
        lines.append("| " + " | ".join(vals) + " |")

    def pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def fmt(v: float, decimals: int = 4) -> str:
        return f"{v:.{decimals}f}"

    # ── En-tête ────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h1("Rapport de benchmark du pipeline d'anonymisation PipeGraph")
    p(f"**Date** : {ts}")
    p(f"**Duree totale** : {elapsed_total / 60:.1f} min ({elapsed_total:.0f}s)")
    p(f"**Pipeline** : PipeGraph v2.1 (architecture hybride Regex + NER + LLM)")

    # ── 1. Configuration du run ────────────────────────────────────────────
    h2("1. Configuration du run")

    h3("1.1 Arguments CLI")
    table_header(["Parametre", "Valeur"])
    table_row(["Datasets evalues", ", ".join(args.datasets)])
    table_row(["Limit / dataset", str(args.limit or "all")])
    table_row(["Risk limit (profils)", str(args.risk_limit)])
    table_row(["Split (TAB)", args.split])
    table_row(["LLM desactive (--no-llm)", str(args.no_llm)])
    table_row(["Mode detection", args.detection_mode])
    table_row(["Skip risk", str(args.skip_risk)])
    table_row(["Risk only", str(args.risk_only)])
    if "ratbench" in args.datasets:
        table_row(["RAT-Bench niveaux", str(args.levels)])
        table_row(["RAT-Bench langue", args.language])

    h3("1.2 Configuration pipeline (runtime)")
    table_header(["Cle", "Valeur"])
    for k, v in sorted(config.items()):
        table_row([str(k), str(v)])

    h3("1.3 Architecture du pipeline")
    p("Le pipeline PipeGraph suit un StateGraph LangGraph avec les noeuds suivants :")
    bullet("**DetectionNode** : detection hybride (deterministe regex + NER IA GLiNER)")
    if config.get("llm_detection"):
        bullet("**LLMDetectionNode** : detection complementaire par LLM")
    if config.get("llm_audit") or config.get("rupta_enabled"):
        bullet("**LLMAuditNode / RUPTA** : audit LLM avec boucle iterative de reecriture")
    bullet("**AnonymizationNode** : anonymisation (pseudo/mask/generalize/redact)")
    if config.get("llm_paraphrase"):
        bullet("**LLMParaphraseNode** : paraphrase LLM pour fluidite du texte anonymise")
    lines.append("")

    # ── 2. Résultats par axe ───────────────────────────────────────────────
    h2("2. Resultats detailles par axe d'evaluation")

    # ── 2.1 Détection d'entités ───────────────────────────────────────────
    detection_results = [r for r in results if r.axis == "entity_detection"]
    if detection_results:
        h3("2.1 Axe 1 — Detection d'entites (span-level)")
        p("Metriques : Precision (exactitude des predictions), Recall (couverture des entites reelles), F2 (favorise le recall).\n"
          "Les metriques sont calculees au niveau span avec overlap partiel.")

        table_header([
            "Dataset",
            "Documents",
            "Macro Precision",
            "Macro Recall",
            "Macro Recall exact",
            "Macro F2",
            "Micro Precision",
            "Micro Recall",
            "Micro Recall exact",
            "Micro F2",
            "Predictions",
            "Ground-truth",
            "Leaks",
            "Duree (s)",
        ])
        for r in detection_results:
            m = r.metrics
            table_row([
                r.dataset_name,
                str(m.get("n_documents", 0)),
                fmt(m.get("macro_precision", 0)),
                fmt(m.get("macro_recall", 0)),
                fmt(m.get("macro_exact_label_recall", 0)),
                fmt(m.get("macro_f2", 0)),
                fmt(m.get("micro_precision", 0)),
                fmt(m.get("micro_recall", 0)),
                fmt(m.get("micro_exact_label_recall", 0)),
                fmt(m.get("micro_f2", 0)),
                str(m.get("total_predictions", 0)),
                str(m.get("total_ground_truth", 0)),
                str(m.get("total_leaks", 0)),
                fmt(r.elapsed_s, 0),
            ])
        lines.append("")

        # Per-label pour chaque dataset
        for r in detection_results:
            label_metrics = r.metrics.get("label_metrics", {})
            if label_metrics:
                h3(f"2.1.{detection_results.index(r) + 1} Metriques par type d'entite — {r.dataset_name}")
                table_header(["Type entite", "Precision", "Recall", "Recall exact", "F1", "TP", "FP", "FN", "Support"])
                for label, lm in sorted(label_metrics.items(), key=lambda x: -x[1]["support"]):
                    table_row([
                        label,
                        fmt(lm["precision"]),
                        fmt(lm["recall"]),
                        fmt(lm.get("exact_recall", 0)),
                        fmt(lm["f1"]),
                        str(lm["tp"]),
                        str(lm["fp"]),
                        str(lm["fn"]),
                        str(lm["support"]),
                    ])
                lines.append("")

        # RAT-Bench : par difficulté
        for r in detection_results:
            by_diff = r.metrics.get("by_difficulty", {})
            if by_diff:
                h3(f"Par difficulte — {r.dataset_name}")
                table_header([
                    "Niveau",
                    "Documents",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F2",
                    "Micro Precision",
                    "Micro Recall",
                    "Micro F2",
                    "Leaks",
                ])
                for lev, dm in sorted(by_diff.items()):
                    table_row([
                        f"L{lev}",
                        str(dm.get("n_documents", 0)),
                        fmt(dm.get("macro_precision", 0)),
                        fmt(dm.get("macro_recall", 0)),
                        fmt(dm.get("macro_f2", 0)),
                        fmt(dm.get("micro_precision", 0)),
                        fmt(dm.get("micro_recall", 0)),
                        fmt(dm.get("micro_f2", 0)),
                        str(dm.get("total_leaks", 0)),
                    ])
                lines.append("")

            by_scene = r.metrics.get("by_scenario", {})
            if by_scene:
                h3(f"Par scenario — {r.dataset_name}")
                table_header([
                    "Scenario",
                    "Documents",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F2",
                    "Micro Precision",
                    "Micro Recall",
                    "Micro F2",
                ])
                for sc, sm in sorted(by_scene.items()):
                    table_row([
                        sc,
                        str(sm.get("n_documents", 0)),
                        fmt(sm.get("macro_precision", 0)),
                        fmt(sm.get("macro_recall", 0)),
                        fmt(sm.get("macro_f2", 0)),
                        fmt(sm.get("micro_precision", 0)),
                        fmt(sm.get("micro_recall", 0)),
                        fmt(sm.get("micro_f2", 0)),
                    ])
                lines.append("")

            direct_rates = r.metrics.get("direct_id_detection_rates", {})
            if direct_rates:
                h3(f"Detection des identifiants directs — {r.dataset_name}")
                table_header(["Type ID", "Total", "Detectes", "Taux detection"])
                for id_type, dr in sorted(direct_rates.items()):
                    table_row([
                        id_type,
                        str(dr["total"]),
                        str(dr["detected"]),
                        pct(dr["detection_rate"]),
                    ])
                lines.append("")

    # ── 2.2 Protection des attributs ──────────────────────────────────────
    protection_results = [r for r in results if r.axis == "attribute_protection" and not r.metrics.get("error")]
    if protection_results:
        h3("2.2 Axe 2 — Protection des attributs (text-leak analysis)")
        p("Analyse si les attributs personnels (nom, age, ville, etc.) sont encore "
          "lisibles dans le texte anonymise. Protection rate = 1 - leak rate.")

        table_header(["Dataset", "Documents", "Protection globale", "Protection directe", "Protection indirecte", "Attrs total", "Attrs fuites"])
        for r in protection_results:
            m = r.metrics
            table_row([
                r.dataset_name,
                str(m.get("n_documents", 0)),
                pct(m.get("avg_protection_rate", 0)),
                pct(m.get("avg_direct_protection_rate", 0)),
                pct(m.get("avg_indirect_protection_rate", 0)),
                str(m.get("total_attributes", 0)),
                str(m.get("total_leaked", 0)),
            ])
        lines.append("")

        # Fuite par attribut
        for r in protection_results:
            per_attr = r.metrics.get("per_attribute", {})
            if per_attr:
                h3(f"Fuite par attribut — {r.dataset_name}")
                table_header(["Attribut", "Total", "Fuite", "Taux protection", "Taux fuite"])
                for attr_name, stats in sorted(per_attr.items(), key=lambda x: -x[1]["leak_rate"]):
                    table_row([
                        attr_name,
                        str(stats["total"]),
                        str(stats["leaked"]),
                        pct(stats["protection_rate"]),
                        pct(stats["leak_rate"]),
                    ])
                lines.append("")

    # ── 2.3 Risque de ré-identification ───────────────────────────────────
    risk_results = [r for r in results if r.axis == "reidentification_risk" and not r.metrics.get("error")]
    if risk_results:
        h3("2.3 Axe 3 — Risque de re-identification (Staab et al.)")
        p("Algorithme : un attaquant LLM infere les attributs quasi-identifiants depuis le texte anonymise, "
          "puis on calcule la classe d'equivalence k dans la population PUMS. Risk R = 1/k.\n"
          "R = 1.0 signifie re-identification certaine (fuite directe ou k=1).")

        table_header(["Dataset", "Risque moyen", "% re-identifies (R>=1)", "% haut risque (R>=0.09)", "Attrs inferes (moy)", "Population", "Source pop"])
        for r in risk_results:
            m = r.metrics
            table_row([
                r.dataset_name,
                fmt(m.get("avg_risk", 0)),
                pct(m.get("frac_re_identified", 0)),
                pct(m.get("frac_high_risk_geq_0_09", 0)),
                fmt(m.get("avg_correct_attrs", 0), 2),
                str(m.get("population_size", "?")),
                str(m.get("population_source", "?")),
            ])
        lines.append("")

        for r in risk_results:
            by_diff = r.metrics.get("by_difficulty", {})
            if by_diff:
                h3(f"Risque par difficulte — {r.dataset_name}")
                table_header(["Niveau", "Nombre", "Risque moyen"])
                for lev, stats in sorted(by_diff.items()):
                    table_row([
                        f"L{lev}",
                        str(int(stats.get("count", 0))),
                        fmt(stats.get("mean", 0)),
                    ])
                lines.append("")

            by_scenario = r.metrics.get("by_scenario", {})
            if by_scenario:
                h3(f"Risque par scenario — {r.dataset_name}")
                table_header(["Scenario", "Nombre", "Risque moyen"])
                for sc, stats in sorted(by_scenario.items()):
                    table_row([
                        sc,
                        str(int(stats.get("count", 0))),
                        fmt(stats.get("mean", 0)),
                    ])
                lines.append("")

    # ── 3. Analyse des erreurs & faiblesses ───────────────────────────────
    h2("3. Analyse des erreurs et faiblesses")

    # 3.1 Types d'entités les plus faibles
    all_label_metrics: Dict[str, Dict[str, Any]] = {}
    for r in detection_results:
        for label, lm in r.metrics.get("label_metrics", {}).items():
            key = f"{r.dataset_name}/{label}"
            all_label_metrics[key] = {**lm, "dataset": r.dataset_name, "label": label}

    if all_label_metrics:
        h3("3.1 Types d'entites avec le plus faible F1")
        p("Entites triees par F1 croissant (les plus faibles en premier). "
          "Un F1 bas avec un support eleve indique un vrai probleme de detection.")

        # Filtrer ceux avec support >= 2 pour éviter le bruit
        weak = sorted(
            [(k, v) for k, v in all_label_metrics.items() if v["support"] >= 2],
            key=lambda x: x[1]["f1"],
        )
        if weak:
            table_header(["Dataset", "Type entite", "F1", "Precision", "Recall", "Support"])
            for k, v in weak[:15]:
                table_row([
                    v["dataset"],
                    v["label"],
                    fmt(v["f1"]),
                    fmt(v["precision"]),
                    fmt(v["recall"]),
                    str(v["support"]),
                ])
            lines.append("")

    # 3.2 Types d'entités avec le plus de faux positifs
    if all_label_metrics:
        h3("3.2 Types d'entites avec le plus de faux positifs")
        p("Un nombre eleve de FP indique que le pipeline sur-detecte ce type. "
          "Cela peut degrader l'utilite du texte anonymise.")

        by_fp = sorted(
            [(k, v) for k, v in all_label_metrics.items() if v["fp"] > 0],
            key=lambda x: -x[1]["fp"],
        )
        if by_fp:
            table_header(["Dataset", "Type entite", "FP", "Precision", "Support"])
            for k, v in by_fp[:10]:
                table_row([
                    v["dataset"],
                    v["label"],
                    str(v["fp"]),
                    fmt(v["precision"]),
                    str(v["support"]),
                ])
            lines.append("")

    # 3.3 Types d'entités avec le plus de faux négatifs (entités manquées)
    if all_label_metrics:
        h3("3.3 Types d'entites avec le plus de faux negatifs (entites manquees)")
        p("Un FN eleve signifie que le pipeline rate des entites de ce type. "
          "C'est critique pour la vie privee : une entite manquee = une fuite potentielle.")

        by_fn = sorted(
            [(k, v) for k, v in all_label_metrics.items() if v["fn"] > 0],
            key=lambda x: -x[1]["fn"],
        )
        if by_fn:
            table_header(["Dataset", "Type entite", "FN", "Recall", "Support"])
            for k, v in by_fn[:10]:
                table_row([
                    v["dataset"],
                    v["label"],
                    str(v["fn"]),
                    fmt(v["recall"]),
                    str(v["support"]),
                ])
            lines.append("")

    # 3.4 Attributs les plus fréquemment fuités
    worst_attrs: List[Dict[str, Any]] = []
    for r in protection_results:
        for attr_name, stats in r.metrics.get("per_attribute", {}).items():
            if stats["leaked"] > 0:
                worst_attrs.append({
                    "dataset": r.dataset_name,
                    "attribute": attr_name,
                    **stats,
                })
    if worst_attrs:
        h3("3.4 Attributs les plus frequemment fuites")
        p("Ces attributs apparaissent encore lisiblement dans le texte anonymise.")
        worst_attrs.sort(key=lambda x: -x["leak_rate"])
        table_header(["Dataset", "Attribut", "Fuite", "Total", "Taux fuite"])
        for a in worst_attrs[:15]:
            table_row([
                a["dataset"],
                a["attribute"],
                str(a["leaked"]),
                str(a["total"]),
                pct(a["leak_rate"]),
            ])
        lines.append("")

    # 3.5 Leaks (entités détectées dans le ground-truth qui restent dans le texte anonymisé)
    total_leaks_all = sum(r.metrics.get("total_leaks", 0) for r in detection_results)
    if total_leaks_all > 0:
        h3("3.5 Fuites textuelles (leaks)")
        p(f"**Total de fuites sur tous les datasets** : {total_leaks_all}")
        p("Une fuite signifie qu'une entite du ground-truth apparait encore telle quelle dans le texte anonymise.")
        table_header(["Dataset", "Leaks", "Documents", "Leaks/doc"])
        for r in detection_results:
            n_leaks = r.metrics.get("total_leaks", 0)
            n_docs = r.metrics.get("n_documents", 1)
            table_row([
                r.dataset_name,
                str(n_leaks),
                str(n_docs),
                fmt(n_leaks / n_docs, 1),
            ])
        lines.append("")

    # ── 4. Résumé exécutif ─────────────────────────────────────────────────
    h2("4. Resume executif")

    # Calcul de stats résumées
    avg_f2 = (
        sum(r.metrics.get("macro_f2", 0) for r in detection_results) / len(detection_results)
        if detection_results else 0
    )
    avg_precision = (
        sum(r.metrics.get("macro_precision", 0) for r in detection_results) / len(detection_results)
        if detection_results else 0
    )
    avg_recall = (
        sum(r.metrics.get("macro_recall", 0) for r in detection_results) / len(detection_results)
        if detection_results else 0
    )
    avg_protection = (
        sum(r.metrics.get("avg_protection_rate", 0) for r in protection_results) / len(protection_results)
        if protection_results else None
    )
    avg_risk = (
        sum(r.metrics.get("avg_risk", 0) for r in risk_results) / len(risk_results)
        if risk_results else None
    )

    p(f"**Detection d'entites** : F2 moyen = {fmt(avg_f2)} (P = {fmt(avg_precision)}, R = {fmt(avg_recall)}) sur {len(detection_results)} datasets")
    if avg_protection is not None:
        p(f"**Protection des attributs** : taux de protection moyen = {pct(avg_protection)} sur {len(protection_results)} datasets")
    if avg_risk is not None:
        p(f"**Risque de re-identification** : risque moyen = {fmt(avg_risk)} sur {len(risk_results)} datasets")

    p(f"**Configuration** : LLM {'actif' if not args.no_llm else 'inactif'}, mode {args.detection_mode}, "
      f"RUPTA {'actif' if config.get('rupta_enabled') else 'inactif'}")

    # Verdict rapide
    h3("Verdict")
    verdicts: List[str] = []
    if avg_recall < 0.7:
        verdicts.append("ALERTE : Recall moyen < 70% — le pipeline manque trop d'entites, risque de fuites important.")
    elif avg_recall < 0.85:
        verdicts.append("ATTENTION : Recall moyen < 85% — des entites sont manquees, marge d'amelioration.")
    else:
        verdicts.append("OK : Recall moyen >= 85% — bonne couverture de detection.")

    if avg_precision < 0.3:
        verdicts.append("ALERTE : Precision moyenne < 30% — sur-detection massive, le texte anonymise perd son utilite.")
    elif avg_precision < 0.5:
        verdicts.append("ATTENTION : Precision moyenne < 50% — sur-detection notable, le texte est degrade.")
    else:
        verdicts.append("OK : Precision moyenne >= 50% — equilibre acceptable detection/utilite.")

    if total_leaks_all > 0:
        n_docs_total = sum(r.metrics.get("n_documents", 0) for r in detection_results)
        leak_ratio = total_leaks_all / n_docs_total if n_docs_total else 0
        if leak_ratio > 1.0:
            verdicts.append(f"ALERTE : {total_leaks_all} fuites sur {n_docs_total} docs ({fmt(leak_ratio, 1)} leaks/doc) — probleme critique.")
        else:
            verdicts.append(f"ATTENTION : {total_leaks_all} fuites detectees — a investiguer.")

    if avg_protection is not None:
        if avg_protection < 0.7:
            verdicts.append(f"ALERTE : Protection attributs < 70% ({pct(avg_protection)}) — anonymisation insuffisante.")
        elif avg_protection < 0.9:
            verdicts.append(f"ATTENTION : Protection attributs < 90% ({pct(avg_protection)}) — certains attributs fuient.")
        else:
            verdicts.append(f"OK : Protection attributs >= 90% ({pct(avg_protection)}).")

    if avg_risk is not None:
        if avg_risk > 0.5:
            verdicts.append(f"ALERTE : Risque re-identification moyen > 50% ({fmt(avg_risk)}) — pipeline facilement attaquable.")
        elif avg_risk > 0.1:
            verdicts.append(f"ATTENTION : Risque re-identification moyen > 10% ({fmt(avg_risk)}) — marge d'amelioration.")
        else:
            verdicts.append(f"OK : Risque re-identification moyen < 10% ({fmt(avg_risk)}).")

    for v in verdicts:
        bullet(v)
    lines.append("")

    # ── 5. Données brutes JSON ─────────────────────────────────────────────
    h2("5. Donnees structurees (JSON)")
    p("Bloc JSON contenant toutes les metriques pour un traitement automatise :")
    lines.append("```json")
    raw_data = {
        "run": {
            "timestamp": ts,
            "elapsed_total_s": round(elapsed_total, 1),
            "config": config,
            "args": {
                "datasets": args.datasets,
                "limit": args.limit,
                "risk_limit": args.risk_limit,
                "levels": args.levels,
                "language": args.language,
                "no_llm": args.no_llm,
                "detection_mode": args.detection_mode,
                "skip_risk": args.skip_risk,
                "split": args.split,
            },
        },
        "results": {
            f"{r.dataset_name}_{r.axis}": r.to_dict()
            for r in results
        },
        "summary": {
            "avg_f2": round(avg_f2, 4),
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_protection_rate": round(avg_protection, 4) if avg_protection is not None else None,
            "avg_risk": round(avg_risk, 4) if avg_risk is not None else None,
            "total_leaks": total_leaks_all,
        },
    }
    lines.append(json.dumps(raw_data, ensure_ascii=False, indent=2, default=str))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def save_llm_report(
    report_md: str,
    out_path: Optional[str] = None,
) -> str:
    """Sauvegarde le rapport LLM au format Markdown."""
    if out_path is None:
        reports_dir = os.path.join(project_root(), "eval", "evaluation", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(reports_dir, f"llm_report_{ts}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return out_path


def save_benchmark_runs(
    results: List[BenchmarkResult],
    config: Dict[str, Any],
) -> List[str]:
    """Sauvegarde les runs individuels pour Streamlit / historique."""
    runs_dir = os.path.join(project_root(), "eval", "evaluation", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    saved: List[str] = []

    for result in results:
        if result.axis != "entity_detection" or not result.report:
            continue

        meta = build_report_meta(
            dataset_name=result.dataset_name,
            limit=len(result.report),
            config=config,
            run_name="full_benchmark",
        )
        path = save_run(
            runs_dir,
            meta=meta,
            data=result.report,
            run_name="full_benchmark",
        )
        saved.append(path)

    return saved


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

AVAILABLE_DATASETS = ["tab", "dbbio", "anonymization", "ratbench", "conll2003"]
DEFAULT_DATASETS = ["tab", "dbbio", "anonymization", "ratbench", "conll2003"]
DEFAULT_LEVELS = [1, 2, 3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Évaluation complète du pipeline PipeGraph sur tous les benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Datasets
    p.add_argument(
        "--datasets", nargs="+",
        choices=AVAILABLE_DATASETS,
        default=DEFAULT_DATASETS,
        help="Datasets à évaluer (défaut: tous)",
    )

    # Limites
    p.add_argument("--limit", type=int, default=None,
                    help="Nombre max de documents par dataset (défaut: tous)")
    p.add_argument("--risk-limit", type=int, default=50,
                    help="Nombre max de profils pour l'évaluation du risque (défaut: 50)")

    # RAT-Bench
    p.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS,
                    help="Niveaux RAT-Bench à évaluer (défaut: 1 2 3)")
    p.add_argument("--language", default="english",
                    help="Langue RAT-Bench (défaut: english)")

    # TAB
    p.add_argument("--split", default="test",
                    help="Split TAB à utiliser (défaut: test)")

    # Pipeline config
    p.add_argument("--no-llm", action="store_true",
                    help="Désactiver LLM (regex + NER uniquement)")
    p.add_argument("--detection-mode", default="parallel",
                    choices=["serial", "parallel"],
                    help="Mode de détection (défaut: parallel)")

    # Axes d'évaluation
    p.add_argument("--skip-risk", action="store_true",
                    help="Ignorer l'évaluation du risque de ré-identification")
    p.add_argument("--risk-only", action="store_true",
                    help="N'exécuter que l'évaluation du risque (nécessite RAT-Bench)")

    # Sorties
    p.add_argument("--save-runs", action="store_true",
                    help="Sauvegarder les runs individuels pour Streamlit")
    p.add_argument("--out", default=None,
                    help="Chemin du rapport de sortie (JSON)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    _section(f"BENCHMARK COMPLET DU PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── Configuration ──────────────────────────────────────────────────────
    llm_active = not args.no_llm

    config = build_runtime_config(
        enable_detection=True,
        enable_deterministic=True,
        enable_ai=True,
        enable_anonymization=True,
        detection_mode=args.detection_mode,
        llm_detection=llm_active,
        llm_verification=llm_active,
        llm_audit=llm_active,
        llm_paraphrase=False,
        rupta_enabled=llm_active,
    )

    _info(f"{'Datasets':25s} : {', '.join(args.datasets)}")
    _info(f"{'Limit/dataset':25s} : {args.limit or 'all'}")
    _info(f"{'LLM actif':25s} : {'Oui' if llm_active else 'Non'}")
    _info(f"{'Mode détection':25s} : {args.detection_mode}")
    _info(f"{'Éval risque':25s} : {'Oui' if not args.skip_risk else 'Non'}")
    if "ratbench" in args.datasets:
        _info(f"{'RAT-Bench niveaux':25s} : {args.levels}")
        _info(f"{'RAT-Bench langue':25s} : {args.language}")
        _info(f"{'Risk limit':25s} : {args.risk_limit}")

    # ── Chargement du pipeline ─────────────────────────────────────────────
    _section("CHARGEMENT DU PIPELINE")
    ensure_pipegraph_importable()
    _create_pipeline, create_initial_state = load_pipegraph()
    pipeline = _create_pipeline()
    _ok("Pipeline PipeGraph chargé")

    all_results: List[BenchmarkResult] = []
    t_global = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # Si --risk-only, on saute la détection et on va directement au risque
    # ══════════════════════════════════════════════════════════════════════
    if args.risk_only:
        _section("MODE RISK-ONLY : évaluation du risque uniquement")
        for level in args.levels:
            _info(f"\n  ▶ RAT-Bench {args.language} L{level}")

            # On doit quand même faire la détection pour avoir les textes anonymisés
            det_result = evaluate_entity_detection(
                "ratbench", pipeline, create_initial_state, config,
                limit=args.risk_limit,
                language=args.language,
                level=level,
            )

            risk_result = evaluate_reidentification_risk(
                pipeline, create_initial_state, config,
                report=det_result.report,
                language=args.language,
                level=level,
                limit=args.risk_limit,
            )
            all_results.append(risk_result)
            print_reidentification_results(risk_result)

        print_summary_table(all_results)
        report_path = save_benchmark_report(all_results, config, args)
        _ok(f"Rapport JSON sauvegardé : {report_path}")

        # Rapport LLM
        elapsed_risk = time.time() - t_global
        llm_md = generate_llm_report(all_results, config, args, elapsed_risk)
        llm_path = save_llm_report(llm_md)
        _ok(f"Rapport LLM sauvegardé : {llm_path}")
        return

    # ══════════════════════════════════════════════════════════════════════
    # AXE 1 : DÉTECTION D'ENTITÉS
    # ══════════════════════════════════════════════════════════════════════
    _section("AXE 1 : DÉTECTION D'ENTITÉS (span-level)")

    # Cache des rapports RAT-Bench pour réutilisation dans les axes 2 et 3
    ratbench_reports: Dict[int, BenchmarkResult] = {}

    for dataset in args.datasets:
        if dataset == "ratbench":
            # Évaluer chaque niveau séparément
            for level in args.levels:
                _info(f"\n  ▶ RAT-Bench {args.language} L{level}")
                result = evaluate_entity_detection(
                    dataset, pipeline, create_initial_state, config,
                    limit=args.limit,
                    language=args.language,
                    level=level,
                )
                all_results.append(result)
                ratbench_reports[level] = result
                print_entity_detection_results(result)
        else:
            _info(f"\n  ▶ {dataset}")
            result = evaluate_entity_detection(
                dataset, pipeline, create_initial_state, config,
                limit=args.limit,
                split=args.split,
            )
            all_results.append(result)
            print_entity_detection_results(result)

    # ══════════════════════════════════════════════════════════════════════
    # AXE 2 : PROTECTION DES ATTRIBUTS (RAT-Bench uniquement)
    # ══════════════════════════════════════════════════════════════════════
    if "ratbench" in args.datasets and ratbench_reports:
        _section("AXE 2 : PROTECTION DES ATTRIBUTS (text-leak analysis)")

        for level, det_result in sorted(ratbench_reports.items()):
            prot_result = evaluate_attribute_protection(
                det_result.report,
                det_result.dataset_name,
            )
            all_results.append(prot_result)
            print_attribute_protection_results(prot_result)

    # ══════════════════════════════════════════════════════════════════════
    # AXE 3 : RISQUE DE RÉ-IDENTIFICATION (RAT-Bench, opt-in)
    # ══════════════════════════════════════════════════════════════════════
    if "ratbench" in args.datasets and not args.skip_risk:
        _section("AXE 3 : RISQUE DE RÉ-IDENTIFICATION (Staab et al.)")

        for level in args.levels:
            _info(f"\n  ▶ RAT-Bench {args.language} L{level}")

            # Réutiliser le rapport de détection si disponible
            det_report = ratbench_reports.get(level, BenchmarkResult(
                dataset_name="", axis="", metrics={}, elapsed_s=0,
            )).report or []

            risk_result = evaluate_reidentification_risk(
                pipeline, create_initial_state, config,
                report=det_report,
                language=args.language,
                level=level,
                limit=args.risk_limit,
            )
            all_results.append(risk_result)
            print_reidentification_results(risk_result)

    # ══════════════════════════════════════════════════════════════════════
    # RÉCAPITULATIF
    # ══════════════════════════════════════════════════════════════════════
    elapsed_total = time.time() - t_global
    print_summary_table(all_results)

    _info(f"\n  Durée totale : {elapsed_total / 60:.1f} min ({elapsed_total:.0f}s)")

    # ── Sauvegarde ─────────────────────────────────────────────────────────
    if args.out:
        report_path = args.out
    else:
        report_path = save_benchmark_report(all_results, config, args)

    _ok(f"Rapport JSON sauvegardé : {report_path}")

    # ── Rapport LLM (Markdown) ────────────────────────────────────────────
    llm_md = generate_llm_report(all_results, config, args, elapsed_total)
    llm_path = save_llm_report(llm_md)
    _ok(f"Rapport LLM sauvegardé : {llm_path}")

    if args.save_runs:
        saved_runs = save_benchmark_runs(all_results, config)
        for p in saved_runs:
            _ok(f"Run sauvegardé : {os.path.basename(p)}")

    print()


if __name__ == "__main__":
    main()
