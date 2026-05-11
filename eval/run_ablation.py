"""
Ablation Study Runner for PipeGraph.

Runs the pipeline on a benchmark dataset across a grid of configurations and
produces a comparison table to identify which components matter most.

Usage examples
--------------
# Quick ablation sur TAB/test avec les 10 premières entrées
    python eval/run_ablation.py --dataset tab --limit 10 --suite nodes

# Étude complète des ensembles NER sur RAT-Bench (anglais, niveau 1)
    python eval/run_ablation.py --dataset ratbench --level 1 --limit 50 --suite ner_ensemble

# Ablation complète avec LLM activé sur TAB
    python eval/run_ablation.py --dataset tab --limit 50 --suite full --with-llm --save-runs

# Ablation personnalisée via fichier JSON
    python eval/run_ablation.py --dataset tab --limit 50 --suite custom --custom-config ablation_grid.json

# Lister les suites disponibles
    python eval/run_ablation.py --list-suites

Suites disponibles
------------------
  nodes         : Ablation des nœuds (detection only, +LLM, +RUPTA, full)
  ner_presets   : Presets GLiNER (fast → full)
  ner_ensemble  : Étude de l'ensemble multi-modèle GLiNER
  ner_threshold : Variation du seuil de confiance GLiNER
  ner_vote      : Variation du seuil de vote (consensus multi-modèle)
  anon_strategy : Comparaison des stratégies d'anonymisation
  detection_mode: serial vs parallel
  full          : Toutes les suites (long)
  custom        : Grille personnalisée depuis --custom-config
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
for _p in (EVAL_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.core.bootstrap import load_pipegraph
from eval.core.datasets import get_allowed_labels, load_benchmark_docs
from eval.core.profiles import (
    EVAL_MODE_CHOICES,
    MASKING_MODE_CHOICES,
    PROFILE_CHOICES,
    apply_profile_to_config,
)
from eval.core.reporting import aggregate_document_metrics
from eval.pipegraph_eval_local import build_report
from eval.run_store import save_run, utc_now_iso

# ---------------------------------------------------------------------------
# Ablation configuration grids
# ---------------------------------------------------------------------------

AblationGrid = List[Dict[str, Any]]


def _suite_nodes() -> AblationGrid:
    """Ablation des nœuds du pipeline — isole la contribution de chaque nœud."""
    return [
        {
            "name": "01_regex_only",
            "description": "Regex déterministe uniquement (pas de NER IA, pas de LLM)",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": False,
                "enable_anonymization": True,
                "disable_llm": True,
            },
        },
        {
            "name": "02_ner_only",
            "description": "NER IA uniquement (pas de regex, pas de LLM)",
            "config": {
                "enable_detection": True,
                "enable_deterministic": False,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": True,
            },
        },
        {
            "name": "03_regex_plus_ner",
            "description": "Regex + NER IA (pas de LLM)",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": True,
            },
        },
        {
            "name": "04_regex_plus_ner_plus_llm_detect",
            "description": "Regex + NER IA + LLM détection (pas d'audit RUPTA)",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": False,
                "llm_detection": True,
                "llm_verification": False,
                "llm_audit": False,
                "llm_paraphrase": False,
                "rupta_enabled": False,
            },
        },
        {
            "name": "04b_regex_ner_llm_detect_verify",
            "description": "Regex + NER + LLM détection + LLM vérification",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": False,
                "llm_detection": True,
                "llm_verification": True,
                "llm_audit": False,
                "llm_paraphrase": False,
                "rupta_enabled": False,
            },
        },
        {
            "name": "04c_regex_ner_llm_verify_only",
            "description": "Regex + NER + LLM vérification (sans LLM détection)",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": False,
                "llm_detection": False,
                "llm_verification": True,
                "llm_audit": False,
                "llm_paraphrase": False,
                "rupta_enabled": False,
            },
        },
        {
            "name": "05_full_pipeline",
            "description": "Pipeline complet (Regex + NER + LLM detect + verify + RUPTA)",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": False,
                "llm_detection": True,
                "llm_verification": True,
                "llm_audit": True,
                "llm_paraphrase": True,
                "rupta_enabled": True,
            },
        },
    ]


def _suite_ner_presets() -> AblationGrid:
    """Ablation des presets GLiNER — compare vitesse vs qualité."""
    presets = [
        ("fast",      "urchade/gliner_small-v2.1 uniquement (le plus rapide)"),
        ("balanced",  "urchade/gliner_medium-v2.1 (équilibré)"),
        ("pii",       "urchade/gliner_multi_pii-v1 (spécialisé PII)"),
        ("multitask", "knowledgator/gliner-multitask-v1.0 (polyvalent)"),
        ("accuracy",  "gliner_large + gliner_multi (précision max sans ensemble PII)"),
        ("best",      "gliner_large + gliner_multi_pii (prod recommandé)"),
        ("full",      "4 modèles — ensemble complet (plus lent)"),
    ]
    return [
        {
            "name": f"preset_{name}",
            "description": desc,
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": True,
                "gliner_preset": name,
            },
        }
        for name, desc in presets
    ]


def _suite_ner_ensemble() -> AblationGrid:
    """
    Étude d'ensemble multi-modèle GLiNER.
    Teste des modèles individuellement, puis en combinaisons croissantes.
    """
    individual_models = [
        "urchade/gliner_small-v2.1",
        "urchade/gliner_medium-v2.1",
        "urchade/gliner_large-v2.1",
        "urchade/gliner_multi-v2.1",
        "urchade/gliner_multi_pii-v1",
        "knowledgator/gliner-multitask-v1.0",
        "EmergentMethods/gliner_medium_news-v2.1",
        "numind/NuNER_Zero-span",
    ]

    base_cfg: Dict[str, Any] = {
        "enable_detection": True,
        "enable_deterministic": True,
        "enable_ai": True,
        "enable_anonymization": True,
        "disable_llm": True,
    }

    grid: AblationGrid = []

    # 1. Modèles individuels
    for model in individual_models:
        short = model.split("/")[-1].replace(".", "_")
        grid.append({
            "name": f"solo_{short}",
            "description": f"Modèle individuel: {model}",
            "config": {**base_cfg, "gliner_models": [model]},
        })

    # 2. Paires clés : meilleur généraliste + spécialiste PII
    grid.append({
        "name": "pair_large_pii",
        "description": "Paire: gliner_large + gliner_multi_pii (preset 'best')",
        "config": {**base_cfg, "gliner_preset": "best"},
    })
    grid.append({
        "name": "pair_large_multi",
        "description": "Paire: gliner_large + gliner_multi",
        "config": {
            **base_cfg,
            "gliner_models": [
                "urchade/gliner_large-v2.1",
                "urchade/gliner_multi-v2.1",
            ],
        },
    })
    grid.append({
        "name": "pair_pii_multitask",
        "description": "Paire: gliner_multi_pii + gliner-multitask",
        "config": {
            **base_cfg,
            "gliner_models": [
                "urchade/gliner_multi_pii-v1",
                "knowledgator/gliner-multitask-v1.0",
            ],
        },
    })

    # 3. Triplets
    grid.append({
        "name": "trio_large_pii_multi",
        "description": "Trio: large + multi_pii + multi",
        "config": {
            **base_cfg,
            "gliner_models": [
                "urchade/gliner_large-v2.1",
                "urchade/gliner_multi_pii-v1",
                "urchade/gliner_multi-v2.1",
            ],
        },
    })
    grid.append({
        "name": "trio_large_pii_multitask",
        "description": "Trio: large + multi_pii + multitask",
        "config": {
            **base_cfg,
            "gliner_models": [
                "urchade/gliner_large-v2.1",
                "urchade/gliner_multi_pii-v1",
                "knowledgator/gliner-multitask-v1.0",
            ],
        },
    })

    # 4. Ensemble complet
    grid.append({
        "name": "ensemble_full",
        "description": "Ensemble complet (4 modèles — preset 'full')",
        "config": {**base_cfg, "gliner_preset": "full"},
    })

    return grid


def _suite_ner_threshold() -> AblationGrid:
    """Ablation du seuil de confiance GLiNER."""
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    base_cfg: Dict[str, Any] = {
        "enable_detection": True,
        "enable_deterministic": True,
        "enable_ai": True,
        "enable_anonymization": True,
        "disable_llm": True,
        "gliner_preset": "best",
    }
    return [
        {
            "name": f"threshold_{str(t).replace('.', '_')}",
            "description": f"Seuil de confiance GLiNER: {t}",
            "config": {**base_cfg, "gliner_threshold": t},
        }
        for t in thresholds
    ]


def _suite_ner_vote() -> AblationGrid:
    """Ablation du seuil de vote (consensus multi-modèle)."""
    # Avec "best" (2 modèles, poids ≥1.0), les votes cumulés varient entre ~1.0 et ~2.25
    # 1.0 = accepter toute détection (même un seul modèle)
    # 1.5 = exiger ≈ consensus de 1.5 poids → à mi-chemin
    # 2.0 = exiger les deux modèles
    votes = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.1, 2.25]
    base_cfg: Dict[str, Any] = {
        "enable_detection": True,
        "enable_deterministic": True,
        "enable_ai": True,
        "enable_anonymization": True,
        "disable_llm": True,
        "gliner_preset": "best",
    }
    return [
        {
            "name": f"min_vote_{str(v).replace('.', '_')}",
            "description": f"Seuil de vote NER minimum: {v}",
            "config": {**base_cfg, "ner_min_vote": v},
        }
        for v in votes
    ]


def _suite_anon_strategy() -> AblationGrid:
    """Ablation des stratégies d'anonymisation."""
    return [
        {
            "name": "strategy_pseudo",
            "description": "Pseudonymisation cohérente (défaut production)",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": True,
                "anon_strategy": "pseudo",
            },
        },
        {
            "name": "strategy_generalize",
            "description": "Généralisation (remplacement par [TYPE])",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": True,
                "anon_strategy": "generalize",
            },
        },
        {
            "name": "strategy_mask",
            "description": "Masquage partiel adapté au type",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": True,
                "anon_strategy": "mask",
            },
        },
        {
            "name": "strategy_redact",
            "description": "Rédaction complète [TYPE_REDACTED]",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": True,
                "anon_strategy": "redact",
            },
        },
        {
            "name": "strategy_policy_mixed",
            "description": "Politique mixte par type (PERSON→pseudo, LOC→generalize, PHONE→mask, IBAN→redact)",
            "config": {
                "enable_detection": True,
                "enable_deterministic": True,
                "enable_ai": True,
                "enable_anonymization": True,
                "disable_llm": True,
                "anon_strategy": "pseudo",
                "anon_policy": {
                    "PERSON": "pseudo",
                    "PER": "pseudo",
                    "LOC": "generalize",
                    "LOCATION": "generalize",
                    "GPE": "generalize",
                    "DATE": "generalize",
                    "PHONE": "mask",
                    "EMAIL": "mask",
                    "IBAN": "redact",
                },
            },
        },
    ]


def _suite_detection_mode() -> AblationGrid:
    """Ablation du mode d'exécution (série vs parallèle)."""
    base_cfg: Dict[str, Any] = {
        "enable_detection": True,
        "enable_deterministic": True,
        "enable_ai": True,
        "enable_anonymization": True,
        "disable_llm": True,
        "gliner_preset": "best",
    }
    return [
        {
            "name": "mode_serial",
            "description": "Exécution série (déterministe, puis NER)",
            "config": {**base_cfg, "detection_mode": "serial"},
        },
        {
            "name": "mode_parallel",
            "description": "Exécution parallèle (déterministe et NER en simultané)",
            "config": {**base_cfg, "detection_mode": "parallel"},
        },
    ]


SUITES: Dict[str, AblationGrid] = {
    "nodes": _suite_nodes(),
    "ner_presets": _suite_ner_presets(),
    "ner_ensemble": _suite_ner_ensemble(),
    "ner_threshold": _suite_ner_threshold(),
    "ner_vote": _suite_ner_vote(),
    "anon_strategy": _suite_anon_strategy(),
    "detection_mode": _suite_detection_mode(),
}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_docs(
    dataset: str,
    project_root: str,
    limit: int,
    level: Optional[int] = None,
    language: str = "english",
    split: str = "test",
) -> Tuple[List[Any], str]:
    """Returns (docs, dataset_name)."""
    return load_benchmark_docs(
        dataset=dataset,
        project_root=project_root,
        limit=limit,
        level=level,
        language=language,
        split=split,
    )


def _augment_dataset_runtime_config(
    dataset: str,
    config: Dict[str, Any],
    *,
    profile: str = "auto",
    eval_mode: str = "both",
    masking_mode: str = "benchmark",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    runtime = dict(config)
    if llm_provider:
        runtime["llm_provider"] = llm_provider
    if llm_model:
        runtime["llm_model"] = llm_model
    return apply_profile_to_config(
        runtime,
        dataset_key=dataset,
        profile_name=runtime.get("profile") or runtime.get("eval_profile") or profile,
        eval_mode=runtime.get("eval_mode") or eval_mode,
        masking_mode=runtime.get("masking_mode") or masking_mode,
    )


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _agg(report: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = aggregate_document_metrics(report)
    return {
        "precision": float(metrics.get("macro_precision", 0.0)),
        "recall": float(metrics.get("macro_recall", 0.0)),
        "f2": float(metrics.get("macro_f2", 0.0)),
        "leaks": int(metrics.get("total_leaks", 0)),
        "n": int(metrics.get("n_documents", 0)),
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    cols = ["name", "precision", "recall", "f2", "leaks", "elapsed_s", "description"]
    widths = {c: len(c) for c in cols}
    for r in results:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))

    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    header = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"

    print("\n" + sep)
    print(header)
    print(sep)

    # Sort by recall desc (priorité au rappel pour l'anonymisation)
    for r in sorted(results, key=lambda x: x.get("recall", 0.0), reverse=True):
        row = "|" + "|".join(f" {str(r.get(c, '')):<{widths[c]}} " for c in cols) + "|"
        print(row)

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ablation Study Runner — compare les configurations du pipeline."
    )
    parser.add_argument(
        "--dataset",
        choices=["tab", "dbbio", "anonymization", "ratbench", "conll2003"],
        default="tab",
        help="Dataset de benchmark (défaut: tab)",
    )
    parser.add_argument("--split", choices=["test", "dev", "train"], default="test")
    parser.add_argument("--language", choices=["english", "mandarin", "spanish"], default="english")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Nombre max de documents à évaluer par configuration (défaut: 50)",
    )
    parser.add_argument(
        "--suite",
        choices=list(SUITES.keys()) + ["full", "custom"],
        default="ner_ensemble",
        help="Suite d'ablation à exécuter",
    )
    parser.add_argument(
        "--custom-config",
        dest="custom_config",
        default=None,
        help="Chemin vers un fichier JSON définissant une grille personnalisée",
    )
    parser.add_argument(
        "--save-runs",
        action="store_true",
        help="Sauvegarder chaque run dans eval/evaluation/runs/",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Chemin du fichier JSON de sortie du résumé (optionnel)",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="Lister les suites disponibles et quitter",
    )
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="Afficher la grille de configs sans l'exécuter",
    )
    parser.add_argument(
        "--with-llm",
        dest="with_llm",
        action="store_true",
        help=(
            "Force l'activation du LLM dans toutes les configurations de la grille "
            "(disable_llm=False, llm_detection=True, llm_audit=True, "
            "llm_paraphrase=True, rupta_enabled=True). "
            "Utile pour mesurer la contribution du LLM sur toutes les variantes NER/threshold."
        ),
    )
    parser.add_argument("--no-llm", action="store_true", help="Force la désactivation LLM sur toutes les configs")
    parser.add_argument("--with-rupta", action="store_true", help="Force RUPTA actif sur toutes les configs LLM")
    parser.add_argument("--llm-provider", default=None, help="Provider LLM runtime (ex: openrouter, ollama)")
    parser.add_argument("--llm-model", default=None, help="Modèle LLM runtime à utiliser pour tous les rôles")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="auto", help="Profil dataset (défaut: auto)")
    parser.add_argument("--eval-mode", choices=EVAL_MODE_CHOICES, default="both", help="Mode d'évaluation")
    parser.add_argument("--masking-mode", choices=MASKING_MODE_CHOICES, default="benchmark", help="Mode de masquage")
    parser.add_argument(
        "-j", "--parallel-configs",
        dest="parallel_configs",
        default="auto",
        help=(
            "Nombre de configs évaluées en parallèle (défaut: auto). "
            "'auto' = détection intelligente selon LLM activé/désactivé. "
            "'1' = séquentiel (debug). Nombre explicite sinon."
        ),
    )
    args = parser.parse_args(argv)

    if args.list_suites:
        print("\nSuites disponibles:")
        for name, grid in SUITES.items():
            print(f"  {name:<20} — {len(grid)} configurations")
        print(f"  {'full':<20} — toutes les suites ci-dessus")
        print(f"  {'custom':<20} — depuis --custom-config fichier.json")
        return 0

    # Résoudre la grille
    if args.suite == "full":
        grid: AblationGrid = []
        for s_grid in SUITES.values():
            grid.extend(s_grid)
    elif args.suite == "custom":
        if not args.custom_config:
            print("❌ --custom-config est requis pour la suite 'custom'")
            return 1
        with open(args.custom_config, "r", encoding="utf-8") as f:
            grid = json.load(f)
    else:
        grid = SUITES[args.suite]

    # Appliquer le patch LLM si --with-llm est activé
    if args.with_llm:
        llm_patch: Dict[str, Any] = {
            "disable_llm": False,
            "llm_detection": True,
            "llm_verification": True,
            "llm_audit": True,
            "llm_paraphrase": True,
            "rupta_enabled": True,
        }
        for entry in grid:
            entry["config"].update(llm_patch)
        print("⚡ Mode --with-llm : LLM forcé actif sur toutes les configs")

    if args.with_rupta:
        rupta_patch: Dict[str, Any] = {
            "disable_llm": False,
            "llm_audit": True,
            "llm_paraphrase": True,
            "rupta_enabled": True,
        }
        for entry in grid:
            entry["config"].update(rupta_patch)
        print("⚡ Mode --with-rupta : RUPTA forcé actif sur toutes les configs")

    if args.no_llm:
        no_llm_patch: Dict[str, Any] = {
            "disable_llm": True,
            "llm_detection": False,
            "llm_verification": False,
            "llm_audit": False,
            "llm_paraphrase": False,
            "rupta_enabled": False,
        }
        for entry in grid:
            entry["config"].update(no_llm_patch)
        print("⚡ Mode --no-llm : LLM désactivé sur toutes les configs")

    if args.configs_only:
        print(f"\nGrille '{args.suite}' — {len(grid)} configurations:")
        for i, cfg_entry in enumerate(grid, 1):
            effective_cfg = _augment_dataset_runtime_config(
                args.dataset,
                cfg_entry["config"],
                profile=args.profile,
                eval_mode=args.eval_mode,
                masking_mode=args.masking_mode,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
            )
            print(f"\n  [{i}] {cfg_entry['name']}")
            print(f"       {cfg_entry.get('description', '')}")
            print(f"       config: {json.dumps(effective_cfg)}")
        return 0

    # Charger le pipeline et les documents
    print(f"\n🔧 Chargement du pipeline PipeGraph...", flush=True)
    create_pipeline_graph, create_initial_state = load_pipegraph()
    pipeline = create_pipeline_graph()

    print(f"📥 Chargement du dataset '{args.dataset}' (limit={args.limit})...", flush=True)
    docs, dataset_name = _load_docs(
        dataset=args.dataset,
        project_root=PROJECT_ROOT,
        limit=args.limit,
        level=args.level,
        language=args.language,
        split=args.split,
    )
    print(f"   → {len(docs)} documents chargés", flush=True)

    if not docs:
        print("❌ Aucun document chargé. Vérifiez le dataset.", flush=True)
        return 1

    # Label-scope filtering
    ds_allowed = get_allowed_labels(args.dataset, profile=args.profile)
    if ds_allowed is not None:
        print(f"   → Filtrage des prédictions : labels autorisés = {sorted(ds_allowed)}", flush=True)

    runs_dir = os.path.join(PROJECT_ROOT, "eval", "evaluation", "runs")

    print(f"\n🧪 Suite '{args.suite}' — {len(grid)} configurations à tester\n", flush=True)
    print("=" * 70, flush=True)

    # --- Determine parallelism strategy -----------------------------------
    n_llm = sum(1 for e in grid if not e["config"].get("disable_llm", False))
    n_no_llm = len(grid) - n_llm

    p_str = args.parallel_configs
    if p_str == "auto":
        if len(grid) <= 1:
            parallel_configs = 1
        elif n_llm == 0:
            # All non-LLM: aggressively parallel (GPU-bound, overlap CPU work)
            parallel_configs = min(len(grid), 4)
        else:
            # LLM present: moderate parallelism to share API bandwidth
            parallel_configs = min(len(grid), 3)
    else:
        parallel_configs = max(1, int(p_str))

    _print_lock = threading.Lock()

    # --- Worker: evaluate a single config ---------------------------------
    def _eval_one(
        i: int,
        cfg_entry: Dict[str, Any],
        doc_max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        name = cfg_entry["name"]
        desc = cfg_entry.get("description", "")
        run_config = _augment_dataset_runtime_config(
            args.dataset,
            cfg_entry["config"],
            profile=args.profile,
            eval_mode=args.eval_mode,
            masking_mode=args.masking_mode,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
        )

        with _print_lock:
            print(f"\n[{i:02d}/{len(grid)}] ⏳ {name}", flush=True)
            print(f"       {desc}", flush=True)

        t0 = time.time()
        try:
            report = build_report(
                docs, pipeline, create_initial_state,
                config=run_config,
                max_workers=doc_max_workers,
                allowed_labels=ds_allowed,
            )
            elapsed = round(time.time() - t0, 1)
            agg = _agg(report)

            row: Dict[str, Any] = {
                "name": name,
                "description": desc,
                "config": run_config,
                "precision": agg["precision"],
                "recall": agg["recall"],
                "f2": agg["f2"],
                "leaks": agg["leaks"],
                "n": agg["n"],
                "elapsed_s": elapsed,
            }

            with _print_lock:
                print(
                    f"       ✅ [{i:02d}] P={agg['precision']:.3f}  R={agg['recall']:.3f}  "
                    f"F2={agg['f2']:.3f}  leaks={agg['leaks']}  ({elapsed}s)",
                    flush=True,
                )

            if args.save_runs:
                meta = {
                    "created_at": utc_now_iso(),
                    "pipeline": "pipegraph",
                    "run_name": f"ablation_{args.suite}_{name}",
                    "dataset": {"name": dataset_name},
                    "limit": args.limit,
                    "config": run_config,
                    "ablation": {"suite": args.suite, "entry": name},
                    "aggregate_metrics": agg,
                }
                saved = save_run(
                    runs_dir, meta=meta, data=report,
                    run_name=f"ablation_{args.suite}_{name}",
                )
                with _print_lock:
                    print(f"       💾 Run sauvegardé: {os.path.basename(saved)}", flush=True)

            return row

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            with _print_lock:
                print(f"       ❌ [{i:02d}] Erreur: {exc}", flush=True)
            return {
                "name": name,
                "description": desc,
                "config": run_config,
                "precision": 0.0,
                "recall": 0.0,
                "f2": 0.0,
                "leaks": -1,
                "n": 0,
                "elapsed_s": elapsed,
                "error": str(exc),
            }

    # --- Helper: run a batch of indices in a thread pool ------------------
    def _run_batch(
        indices: List[int],
        n_workers: int,
        doc_max_workers: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        result_map: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_eval_one, idx + 1, grid[idx], doc_max_workers): idx
                for idx in indices
            }
            for fut in as_completed(futs):
                grid_idx = futs[fut]
                try:
                    result_map[grid_idx] = fut.result(timeout=600)
                except Exception as exc:
                    result_map[grid_idx] = {
                        "name": grid[grid_idx]["name"],
                        "description": grid[grid_idx].get("description", ""),
                        "config": grid[grid_idx]["config"],
                        "precision": 0.0, "recall": 0.0, "f2": 0.0,
                        "leaks": -1, "n": 0, "elapsed_s": 0,
                        "error": str(exc),
                    }
        return result_map

    # --- Execute ----------------------------------------------------------
    if parallel_configs <= 1 or len(grid) <= 1:
        # Sequential mode (with per-doc progress)
        def _progress(done: int, total: int, doc_id: str) -> None:
            print(f"       📄 {done}/{total} ({doc_id})", flush=True)

        results: List[Dict[str, Any]] = []
        for i, cfg_entry in enumerate(grid, 1):
            name = cfg_entry["name"]
            desc = cfg_entry.get("description", "")
            run_config = _augment_dataset_runtime_config(
                args.dataset,
                cfg_entry["config"],
                profile=args.profile,
                eval_mode=args.eval_mode,
                masking_mode=args.masking_mode,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
            )
            print(f"\n[{i:02d}/{len(grid)}] {name}", flush=True)
            print(f"       {desc}", flush=True)
            print(f"       config: {json.dumps(run_config)}", flush=True)
            t0 = time.time()
            try:
                report = build_report(
                    docs, pipeline, create_initial_state,
                    config=run_config, progress_cb=_progress,
                    allowed_labels=ds_allowed,
                )
                elapsed = round(time.time() - t0, 1)
                agg = _agg(report)
                row = {
                    "name": name, "description": desc, "config": run_config,
                    "precision": agg["precision"], "recall": agg["recall"],
                    "f2": agg["f2"], "leaks": agg["leaks"],
                    "n": agg["n"], "elapsed_s": elapsed,
                }
                results.append(row)
                print(
                    f"       ✅ P={agg['precision']:.3f}  R={agg['recall']:.3f}  "
                    f"F2={agg['f2']:.3f}  leaks={agg['leaks']}  ({elapsed}s)",
                    flush=True,
                )
                if args.save_runs:
                    meta = {
                        "created_at": utc_now_iso(), "pipeline": "pipegraph",
                        "run_name": f"ablation_{args.suite}_{name}",
                        "dataset": {"name": dataset_name}, "limit": args.limit,
                        "config": run_config,
                        "ablation": {"suite": args.suite, "entry": name},
                        "aggregate_metrics": agg,
                    }
                    saved = save_run(
                        runs_dir, meta=meta, data=report,
                        run_name=f"ablation_{args.suite}_{name}",
                    )
                    print(f"       💾 Run sauvegardé: {os.path.basename(saved)}", flush=True)
            except Exception as exc:
                elapsed = round(time.time() - t0, 1)
                print(f"       ❌ Erreur: {exc}", flush=True)
                results.append({
                    "name": name, "description": desc, "config": run_config,
                    "precision": 0.0, "recall": 0.0, "f2": 0.0,
                    "leaks": -1, "n": 0, "elapsed_s": elapsed, "error": str(exc),
                })
    else:
        # ── Parallel mode ─────────────────────────────────────────────
        no_llm_idx = [i for i, e in enumerate(grid) if e["config"].get("disable_llm", False)]
        llm_idx = [i for i, e in enumerate(grid) if not e["config"].get("disable_llm", False)]

        print(
            f"\n🚀 Mode parallèle : {parallel_configs} configs simultanées "
            f"({n_no_llm} sans LLM, {n_llm} avec LLM)",
            flush=True,
        )

        result_map: Dict[int, Dict[str, Any]] = {}

        # Phase 1 — Non-LLM configs (GPU-bound → serial docs, parallel configs)
        if no_llm_idx:
            n_w = min(len(no_llm_idx), parallel_configs)
            print(
                f"\n📊 Phase 1 : {len(no_llm_idx)} configs sans LLM "
                f"(×{n_w} parallèle, docs séquentiels)",
                flush=True,
            )
            result_map.update(_run_batch(no_llm_idx, n_w, doc_max_workers=1))

        # Phase 2 — LLM configs (I/O-bound → moderate config parallelism,
        #           share per-doc API workers across configs)
        if llm_idx:
            n_w = min(len(llm_idx), max(2, parallel_configs))
            doc_workers = max(2, 8 // n_w)  # keep ~8 API reqs in flight total
            print(
                f"\n📊 Phase 2 : {len(llm_idx)} configs avec LLM "
                f"(×{n_w} parallèle, {doc_workers} doc-workers chacune)",
                flush=True,
            )
            result_map.update(_run_batch(llm_idx, n_w, doc_max_workers=doc_workers))

        # Reconstruct in original grid order
        results = [result_map[i] for i in range(len(grid))]

    # Résumé final
    print("\n" + "=" * 70)
    print(f"📊 Résumé de l'ablation — suite '{args.suite}' sur '{dataset_name}'")
    _print_table(results)

    # Meilleure configuration par rappel
    best_recall = max((r for r in results if r.get("n", 0) > 0), key=lambda x: x["recall"], default=None)
    best_f2 = max((r for r in results if r.get("n", 0) > 0), key=lambda x: x["f2"], default=None)
    if best_recall:
        print(f"🏆 Meilleur Rappel  : [{best_recall['name']}]  R={best_recall['recall']:.4f}")
    if best_f2:
        print(f"🏆 Meilleur F2      : [{best_f2['name']}]  F2={best_f2['f2']:.4f}\n")

    # Sauvegarder le résumé JSON
    if args.out:
        out_path = args.out
    else:
        os.makedirs(os.path.join(PROJECT_ROOT, "eval", "evaluation", "reports"), exist_ok=True)
        out_path = os.path.join(
            PROJECT_ROOT,
            "eval",
            "evaluation",
            "reports",
            f"ablation_{args.suite}_{args.dataset}.json",
        )

    summary = {
        "suite": args.suite,
        "dataset": dataset_name,
        "limit": args.limit,
        "with_llm": args.with_llm,
        "no_llm": args.no_llm,
        "with_rupta": args.with_rupta,
        "profile": args.profile,
        "eval_mode": args.eval_mode,
        "masking_mode": args.masking_mode,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "n_configs": len(grid),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 Résumé sauvegardé: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
