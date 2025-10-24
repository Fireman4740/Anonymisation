#!/usr/bin/env python3
"""
Script d'évaluation du pipeline complet sur différents datasets.

Ce script évalue:
- DB-Bio (biographies de célébrités)
- PersonalReddit (commentaires synthétiques)
- TAB (ECHR court cases)

Avec deux configurations:
- Baseline (L0): Sans LLM, uniquement NER/regex
- RUPTA (L1): Avec optimisation privacy-utility

Usage:
    python scripts/eval_rupta_pipeline.py --dataset dbbio --split test --n_samples 50 --policy L1
    python scripts/eval_rupta_pipeline.py --dataset reddit --split test --n_samples 50 --policy L0
    python scripts/eval_rupta_pipeline.py --dataset tab --split test --n_samples 10 --use_rupta --policy L1
    python scripts/eval_rupta_pipeline.py --all  # Évalue tous les datasets
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.openrouter_client import OpenRouterClient
from src.rupta.privacy_evaluator import evaluate_reidentification_risk
from src.rupta.utility_evaluator import evaluate_utility_preservation
from src.orchestrator import anonymize_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, split: str = "test") -> List[Dict[str, Any]]:
    """Charge un dataset d'évaluation."""
    base_path = Path(__file__).parent.parent / "Dataset" / "evaluation"
    
    if dataset_name == "dbbio":
        file_path = base_path / "DB-bio" / f"{split}.jsonl"
    elif dataset_name == "reddit":
        file_path = base_path / "PersonalReddit" / "Reddit_synthetic" / f"{split}.jsonl"
    elif dataset_name == "tab":
        file_path = base_path / "TAB" / f"{split}.jsonl"
    else:
        raise ValueError(f"Dataset inconnu: {dataset_name}. Choix possibles: dbbio, reddit, tab")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    logger.info(f"Chargé {len(data)} exemples de {dataset_name}/{split}")
    return data


def evaluate_baseline(
    text: str,
    ground_truth_people: str,
    ground_truth_label: str,
    client: OpenRouterClient,
    policy_level: str = "L0"
) -> Dict[str, Any]:
    """Évalue le pipeline baseline sans RUPTA."""
    
    # Anonymisation baseline
    result = anonymize_text(
        value=text,
        scope_id=f"eval_baseline_{int(time.time()*1000)}",
        secret_salt="eval_secret",
        level=policy_level,
    )
    
    anonymized = result.get("anonymized_text", text)
    
    # Évaluation privacy
    privacy_eval = evaluate_reidentification_risk(
        client=client,
        anonymized_text=anonymized,
        ground_truth_people=ground_truth_people,
        p_threshold=10,
        model="qwen/qwen3-30b-a3b-instruct-2507"
    )
    
    # Évaluation utility
    utility_eval = evaluate_utility_preservation(
        client=client,
        anonymized_text=anonymized,
        ground_truth_label=ground_truth_label,
        model="qwen/qwen3-30b-a3b-instruct-2507"
    )
    
    return {
        "anonymized_text": anonymized,
        "privacy": privacy_eval,
        "utility": utility_eval,
        "entities_detected": result.get("entities_detected", [])
    }


def evaluate_rupta(
    text: str,
    ground_truth_people: str,
    ground_truth_label: str,
    client: OpenRouterClient,
    policy_level: str = "L1"
) -> Dict[str, Any]:
    """Évalue le pipeline avec RUPTA."""
    
    # Anonymisation avec RUPTA
    result = anonymize_text(
        value=text,
        scope_id=f"eval_rupta_{int(time.time()*1000)}",
        secret_salt="eval_secret",
        level=policy_level,
        overrides={
            "rupta_enabled": True,
            "rupta_ground_truth_people": ground_truth_people,
            "rupta_ground_truth_label": ground_truth_label,
        }
    )
    
    anonymized = result.get("anonymized_text", text)
    
    # Les métriques RUPTA sont déjà dans le résultat
    rupta_metrics = result.get("rupta_metrics", {})
    
    return {
        "anonymized_text": anonymized,
        "privacy": rupta_metrics.get("privacy", {}),
        "utility": rupta_metrics.get("utility", {}),
        "iterations": rupta_metrics.get("iterations", 0),
        "final_reward": rupta_metrics.get("final_reward", 0),
        "entities_detected": result.get("entities_detected", [])
    }


def run_evaluation(
    dataset_name: str,
    split: str = "test",
    n_samples: int = 10,
    use_baseline: bool = False,
    use_rupta: bool = True,
    policy_level: str = "L1"
) -> Dict[str, Any]:
    """Lance l'évaluation sur un dataset."""
    
    logger.info(f"=== Évaluation {dataset_name} ({split}) ===")
    logger.info(f"Échantillons: {n_samples}")
    logger.info(f"Baseline: {use_baseline}, RUPTA: {use_rupta}, Policy: {policy_level}")
    
    # Charger les données
    data = load_dataset(dataset_name, split)
    if n_samples:
        data = data[:n_samples]
    
    # Initialiser le client
    client = OpenRouterClient()
    
    results = {
        "dataset": dataset_name,
        "split": split,
        "n_samples": len(data),
        "policy_level": policy_level,
        "baseline": [] if use_baseline else None,
        "rupta": [] if use_rupta else None,
        "metrics_summary": {}
    }
    
    # Évaluation
    for i, item in enumerate(data):
        logger.info(f"Traitement exemple {i+1}/{len(data)}")
        
        # Extraire les champs selon le dataset
        if dataset_name == "dbbio":
            text = item["text"]
            people = item["people"]
            label = item["label"]
        elif dataset_name == "reddit":
            text = item.get("response", item.get("text", ""))
            people = item.get("personality", {}).get("name", "Unknown")
            label = item.get("personality", {}).get("occupation", "Unknown")
        elif dataset_name == "tab":
            text = item.get("text", "")
            people = item.get("people", "Unknown")
            label = item.get("label", "applicant")
        else:
            logger.warning(f"Dataset {dataset_name} non reconnu, utilisation des champs par défaut")
            text = item.get("text", "")
            people = item.get("people", "Unknown")
            label = item.get("label", "Unknown")
        
        result_item = {
            "id": i,
            "original_text": text,
            "ground_truth_people": people,
            "ground_truth_label": label
        }
        
        # Baseline
        if use_baseline:
            try:
                baseline_result = evaluate_baseline(text, people, label, client, policy_level)
                result_item["baseline"] = baseline_result
                results["baseline"].append(result_item.copy())
            except Exception as e:
                logger.error(f"Erreur baseline exemple {i}: {e}")
                result_item["baseline"] = {"error": str(e)}
        
        # RUPTA
        if use_rupta:
            try:
                rupta_result = evaluate_rupta(text, people, label, client, policy_level)
                result_item["rupta"] = rupta_result
                results["rupta"].append(result_item.copy())
            except Exception as e:
                logger.error(f"Erreur RUPTA exemple {i}: {e}")
                result_item["rupta"] = {"error": str(e)}
    
    # Calcul des métriques de synthèse
    results["metrics_summary"] = compute_summary_metrics(results)
    
    return results


def compute_summary_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calcule les métriques de synthèse."""
    summary = {}
    
    # Métriques baseline
    if results.get("baseline"):
        baseline_privacy_ranks = [
            r.get("baseline", {}).get("privacy", {}).get("rank")
            for r in results["baseline"]
            if r.get("baseline", {}).get("privacy", {}).get("rank") is not None
        ]
        baseline_utility_scores = [
            r.get("baseline", {}).get("utility", {}).get("confidence", 0) * 100
            for r in results["baseline"]
            if r.get("baseline", {}).get("utility", {}).get("confidence") is not None
        ]
        
        summary["baseline"] = {
            "avg_privacy_rank": sum(baseline_privacy_ranks) / len(baseline_privacy_ranks) if baseline_privacy_ranks else None,
            "non_identified_rate": sum(1 for r in results["baseline"] 
                                      if r.get("baseline", {}).get("privacy", {}).get("rank") == 999) / len(results["baseline"]),
            "avg_utility_score": sum(baseline_utility_scores) / len(baseline_utility_scores) if baseline_utility_scores else None,
            "utility_preserved_rate": sum(1 for s in baseline_utility_scores if s >= 80) / len(baseline_utility_scores) if baseline_utility_scores else None
        }
    
    # Métriques RUPTA
    if results.get("rupta"):
        rupta_privacy_ranks = [
            r.get("rupta", {}).get("privacy", {}).get("rank")
            for r in results["rupta"]
            if r.get("rupta", {}).get("privacy", {}).get("rank") is not None
        ]
        rupta_utility_scores = [
            r.get("rupta", {}).get("utility", {}).get("confidence", 0) * 100
            for r in results["rupta"]
            if r.get("rupta", {}).get("utility", {}).get("confidence") is not None
        ]
        rupta_iterations = [
            r.get("rupta", {}).get("iterations", 0)
            for r in results["rupta"]
            if r.get("rupta", {}).get("iterations") is not None
        ]
        
        summary["rupta"] = {
            "avg_privacy_rank": sum(rupta_privacy_ranks) / len(rupta_privacy_ranks) if rupta_privacy_ranks else None,
            "non_identified_rate": sum(1 for r in results["rupta"] 
                                      if r.get("rupta", {}).get("privacy", {}).get("rank") == 999) / len(results["rupta"]),
            "avg_utility_score": sum(rupta_utility_scores) / len(rupta_utility_scores) if rupta_utility_scores else None,
            "utility_preserved_rate": sum(1 for s in rupta_utility_scores if s >= 80) / len(rupta_utility_scores) if rupta_utility_scores else None,
            "avg_iterations": sum(rupta_iterations) / len(rupta_iterations) if rupta_iterations else None
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Évaluation du pipeline avec RUPTA")
    parser.add_argument("--dataset", choices=["dbbio", "reddit", "tab", "all"], default="dbbio",
                       help="Dataset à évaluer (dbbio, reddit, tab, ou all)")
    parser.add_argument("--split", default="test", help="Split du dataset (train/val/test ou dev pour tab)")
    parser.add_argument("--n_samples", type=int, default=10,
                       help="Nombre d'échantillons à évaluer (0 = tous)")
    parser.add_argument("--use_baseline", action="store_true",
                       help="Évaluer le baseline")
    parser.add_argument("--use_rupta", action="store_true", default=True,
                       help="Évaluer RUPTA")
    parser.add_argument("--policy", choices=["L0", "L1"], default="L1",
                       help="Niveau de policy (L0=baseline sans LLM, L1=avec LLM+RUPTA)")
    parser.add_argument("--output", default="results/eval_rupta_pipeline.json",
                       help="Fichier de sortie")
    parser.add_argument("--all", action="store_true",
                       help="Évaluer tous les datasets (dbbio, reddit, tab)")
    
    args = parser.parse_args()

    # Harmoniser les options selon la policy choisie
    use_baseline = args.use_baseline
    use_rupta = args.use_rupta

    if args.policy == "L0":
        if use_rupta:
            logger.info("Policy L0 détectée : désactivation automatique de RUPTA et activation de la baseline.")
        elif not use_baseline:
            logger.info("Policy L0 détectée : activation automatique de la baseline.")
        use_baseline = True
        use_rupta = False
    elif args.policy == "L1" and not (use_baseline or use_rupta):
        # Sécurité : si aucun flag n'est passé explicitement, on garde le comportement par défaut
        use_rupta = True
    
    # Créer le dossier de sortie
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Liste des datasets à évaluer
    datasets = ["dbbio", "reddit", "tab"] if args.all else [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_evaluation(
                dataset_name=dataset,
                split=args.split,
                n_samples=args.n_samples,
                use_baseline=use_baseline,
                use_rupta=use_rupta,
                policy_level=args.policy
            )
            all_results[dataset] = results
            
            # Sauvegarder les résultats intermédiaires
            intermediate_path = output_path.parent / f"eval_{dataset}_{args.split}_{args.policy}.json"
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Résultats {dataset} sauvegardés dans {intermediate_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de {dataset}: {e}")
            all_results[dataset] = {"error": str(e)}
    
    # Sauvegarder tous les résultats
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Résultats complets sauvegardés dans {output_path}")
    
    # Afficher le résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*80)
    
    for dataset_name, dataset_results in all_results.items():
        if "error" in dataset_results:
            print(f"\n❌ {dataset_name.upper()}: ERREUR - {dataset_results['error']}")
            continue
            
        print(f"\n📊 {dataset_name.upper()}")
        print(f"   Policy: {dataset_results.get('policy_level', 'N/A')}")
        print(f"   Échantillons: {dataset_results['n_samples']}")
        
        summary = dataset_results.get("metrics_summary", {})
        
        if summary.get("baseline"):
            baseline = summary["baseline"]
            avg_privacy_rank = baseline.get("avg_privacy_rank")
            non_identified_rate = baseline.get("non_identified_rate")
            avg_utility_score = baseline.get("avg_utility_score")
            utility_preserved_rate = baseline.get("utility_preserved_rate")

            avg_privacy_rank_str = f"{avg_privacy_rank:.2f}" if isinstance(avg_privacy_rank, (int, float)) else "N/A"
            non_identified_str = f"{non_identified_rate * 100:.1f}%" if isinstance(non_identified_rate, (int, float)) else "N/A"
            avg_utility_score_str = f"{avg_utility_score:.1f}%" if isinstance(avg_utility_score, (int, float)) else "N/A"
            utility_preserved_str = f"{utility_preserved_rate * 100:.1f}%" if isinstance(utility_preserved_rate, (int, float)) else "N/A"

            print(f"\n   Baseline:")
            print(f"     • Privacy Rank moyen: {avg_privacy_rank_str}")
            print(f"     • Taux non-identifié: {non_identified_str}")
            print(f"     • Utility Score moyen: {avg_utility_score_str}")
            print(f"     • Taux utilité préservée: {utility_preserved_str}")
        
        if summary.get("rupta"):
            rupta = summary["rupta"]
            avg_privacy_rank = rupta.get("avg_privacy_rank")
            non_identified_rate = rupta.get("non_identified_rate")
            avg_utility_score = rupta.get("avg_utility_score")
            utility_preserved_rate = rupta.get("utility_preserved_rate")
            avg_iterations = rupta.get("avg_iterations")

            avg_privacy_rank_str = f"{avg_privacy_rank:.2f}" if isinstance(avg_privacy_rank, (int, float)) else "N/A"
            non_identified_str = f"{non_identified_rate * 100:.1f}%" if isinstance(non_identified_rate, (int, float)) else "N/A"
            avg_utility_score_str = f"{avg_utility_score:.1f}%" if isinstance(avg_utility_score, (int, float)) else "N/A"
            utility_preserved_str = f"{utility_preserved_rate * 100:.1f}%" if isinstance(utility_preserved_rate, (int, float)) else "N/A"
            avg_iterations_str = f"{avg_iterations:.2f}" if isinstance(avg_iterations, (int, float)) else "N/A"

            print(f"\n   RUPTA:")
            print(f"     • Privacy Rank moyen: {avg_privacy_rank_str}")
            print(f"     • Taux non-identifié: {non_identified_str}")
            print(f"     • Utility Score moyen: {avg_utility_score_str}")
            print(f"     • Taux utilité préservée: {utility_preserved_str}")
            print(f"     • Itérations moyennes: {avg_iterations_str}")
    
    print("\n" + "="*80)
    print(f"✅ Résultats sauvegardés: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
