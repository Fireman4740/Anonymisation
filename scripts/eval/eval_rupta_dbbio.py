"""
Script d'évaluation RUPTA sur le dataset DB-Bio

Ce script évalue les performances du système d'anonymisation sur le dataset 
DB-Bio en mesurant :
- Privacy : Re-identification Rank (position de la vraie personne dans les candidats)
- Utility : Classification Accuracy (préservation de l'occupation)

Usage:
    python eval_rupta_dbbio.py --split test --n_samples 10 --use_baseline
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.openrouter_client import OpenRouterClient
from src.orchestrator import anonymize_text
from src.rupta.privacy_evaluator import evaluate_reidentification_risk
from src.rupta.utility_evaluator import evaluate_classification_utility
from src.rupta.optimizer import optimize_anonymization


def load_dbbio_dataset(split: str = "test") -> List[Dict[str, Any]]:
    """
    Charge le dataset DB-Bio
    
    Args:
        split: 'train', 'dev' ou 'test'
    
    Returns:
        Liste de dictionnaires avec 'text', 'people', 'label'
    """
    dataset_dir = Path("Dataset/evaluation/DB-Bio")
    
    # Chercher le fichier correspondant au split
    possible_files = [
        dataset_dir / f"{split}.jsonl",
        dataset_dir / f"db-bio-{split}.jsonl",
        dataset_dir / f"dbbio_{split}.jsonl"
    ]
    
    for filepath in possible_files:
        if filepath.exists():
            print(f"📂 Chargement de {filepath}")
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            print(f"   ✅ {len(data)} exemples chargés")
            return data
    
    raise FileNotFoundError(
        f"Aucun fichier trouvé pour split='{split}' dans {dataset_dir}\n"
        f"Fichiers recherchés : {[f.name for f in possible_files]}"
    )


def baseline_anonymization(text: str, config_path: str = "config.json") -> str:
    """
    Anonymisation baseline (système actuel sans RUPTA)
    
    Pour éviter les problèmes de dépendances NER avec Python 3.11,
    on utilise une anonymisation regex simple qui masque :
    - Noms propres (première lettre maj + reste du mot)
    - Dates, emails, numéros, etc. (regex)
    
    Args:
        text: Texte à anonymiser
        config_path: Chemin vers config.json
    
    Returns:
        Texte anonymisé
    """
    import re
    
    try:
        # Étape 1: Anonymisation regex basique
        anonymized = text
        
        # Masquer les dates (années 1900-2099)
        anonymized = re.sub(r'\b(19|20)\d{2}\b', '[DATE]', anonymized)
        
        # Masquer les emails  
        anonymized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', anonymized)
        
        # Masquer les numéros de téléphone
        anonymized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', anonymized)
        
        # Masquer les URLs
        anonymized = re.sub(r'https?://[^\s]+', '[URL]', anonymized)
        
        # Étape 2: Masquer les noms propres (mots avec majuscule initiale)
        # Exclure les mots en début de phrase et les mots courants
        common_words = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'And', 
                       'But', 'Or', 'As', 'If', 'When', 'Where', 'Who', 'Which', 'That',
                       'He', 'She', 'It', 'They', 'We', 'I', 'You', 'His', 'Her', 'Their',
                       'French', 'American', 'English', 'Soviet', 'World', 'Communist'}
        
        # Remplacer les noms propres (2+ lettres, commence par majuscule)
        def replace_proper_name(match):
            word = match.group(0)
            # Ne pas remplacer si c'est un mot commun
            if word in common_words:
                return word
            # Ne pas remplacer si c'est juste après un point (début de phrase)
            if match.start() > 0 and anonymized[match.start()-1:match.start()] == '.':
                return word
            # Remplacer par [PERSON]
            return '[PERSON]'
        
        # Pattern: majuscule suivie de minuscules (au moins 2 lettres)
        anonymized = re.sub(r'\b[A-Z][a-z]{1,}\b', replace_proper_name, anonymized)
        
        # Masquer les noms de famille (MAJ MAJ ou Maj-Maj)
        anonymized = re.sub(r'\b[A-Z][a-z]+-[A-Z][a-z]+\b', '[PERSON]', anonymized)
        anonymized = re.sub(r'\b[A-Z]{2,}\b', '[ORG]', anonymized)
        
        return anonymized
        
    except Exception as e:
        print(f"⚠️  Erreur baseline : {e}")
        import traceback
        traceback.print_exc()
        return text


def rupta_anonymization(
    text: str,
    ground_truth_people: str,
    ground_truth_label: str,
    client: OpenRouterClient,
    config_path: str = "config.json",
    max_iterations: int = 3,
    p_threshold: int = 10
) -> Dict[str, Any]:
    """
    Anonymisation avec optimisation RUPTA.
    """

    initial_anonymized = baseline_anonymization(text, config_path)

    result = optimize_anonymization(
        client=client,
        original_text=text,
        initial_anonymized_text=initial_anonymized,
        ground_truth_people=ground_truth_people,
        ground_truth_label=ground_truth_label,
        max_iterations=max_iterations,
        p_threshold=p_threshold
    )

    return result


def evaluate_sample(
    sample: Dict[str, Any],
    client: OpenRouterClient,
    use_rupta: bool = False,
    p_threshold: int = 10
) -> Dict[str, Any]:
    """
    Évalue un exemple en baseline ou avec RUPTA.
    """

    original_text = sample["text"]
    ground_truth_people = sample["people"][0] if sample["people"] else "Unknown"
    ground_truth_label = sample["label"]

    if use_rupta:
        result = rupta_anonymization(
            text=original_text,
            ground_truth_people=ground_truth_people,
            ground_truth_label=ground_truth_label,
            client=client,
            max_iterations=3,
            p_threshold=p_threshold
        )
        anonymized_text = result["final_text"]
        privacy_score = result["privacy_score"]
        utility_score = result["utility_score"]

        return {
            "original": original_text,
            "anonymized": anonymized_text,
            "people": ground_truth_people,
            "label": ground_truth_label,
            "privacy_rank": privacy_score.get("rank"),
            "privacy_identified": privacy_score.get("confirmation"),
            "utility_confidence": utility_score.get("confidence_score"),
            "utility_preserved": utility_score.get("utility_preserved"),
            "iterations": result.get("iterations"),
            "method": "RUPTA"
        }

    anonymized_text = baseline_anonymization(original_text)
    privacy_eval = evaluate_reidentification_risk(
        client=client,
        anonymized_text=anonymized_text,
        ground_truth_people=ground_truth_people,
        p_threshold=p_threshold
    )
    utility_eval = evaluate_classification_utility(
        client=client,
        anonymized_text=anonymized_text,
        ground_truth_label=ground_truth_label
    )

    return {
        "original": original_text,
        "anonymized": anonymized_text,
        "people": ground_truth_people,
        "label": ground_truth_label,
        "privacy_rank": privacy_eval.get("rank"),
        "privacy_identified": privacy_eval.get("confirmation"),
        "utility_confidence": utility_eval.get("confidence_score"),
        "utility_preserved": utility_eval.get("utility_preserved"),
        "method": "Baseline"
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Union[float, int, None]]:
    """
    Calcule les métriques agrégées
    
    Args:
        results: Liste des résultats individuels
    
    Returns:
        Dictionnaire de métriques
    """
    valid_results = [r for r in results if r.get("privacy_rank") is not None]
    
    if not valid_results:
        return {
            "avg_privacy_rank": None,
            "privacy_not_identified_rate": None,
            "avg_utility_confidence": None,
            "utility_preserved_rate": None,
            "n_samples": len(results)
        }
    
    # Privacy metrics
    privacy_ranks = [r["privacy_rank"] for r in valid_results if r["privacy_rank"] is not None]
    not_identified = [r for r in valid_results if r["privacy_rank"] is None or r["privacy_rank"] > 10]
    
    # Utility metrics
    utility_confidences = [r["utility_confidence"] for r in valid_results if r["utility_confidence"] is not None]
    utility_preserved = [r for r in valid_results if r.get("utility_preserved", False)]
    
    return {
        "avg_privacy_rank": sum(privacy_ranks) / len(privacy_ranks) if privacy_ranks else None,
        "privacy_not_identified_rate": len(not_identified) / len(valid_results),
        "avg_utility_confidence": sum(utility_confidences) / len(utility_confidences) if utility_confidences else None,
        "utility_preserved_rate": len(utility_preserved) / len(valid_results),
        "n_samples": len(valid_results)
    }


def main():
    parser = argparse.ArgumentParser(description="Évaluation RUPTA sur DB-Bio")
    parser.add_argument("--split", type=str, default="test", help="Split à évaluer (train/dev/test)")
    parser.add_argument("--n_samples", type=int, default=10, help="Nombre d'exemples à évaluer")
    parser.add_argument("--use_baseline", action="store_true", help="Utiliser baseline au lieu de RUPTA")
    parser.add_argument("--p_threshold", type=int, default=10, help="Seuil p pour privacy evaluation")
    parser.add_argument("--output", type=str, default="results_dbbio.json", help="Fichier de sortie")
    
    args = parser.parse_args()
    
    # Vérifier la clé API
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY non définie !")
        return
    
    print("\n" + "=" * 60)
    print(f"Évaluation RUPTA sur DB-Bio")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Échantillons: {args.n_samples}")
    print(f"Méthode: {'Baseline' if args.use_baseline else 'RUPTA'}")
    print(f"p_threshold: {args.p_threshold}")
    
    # Charger le dataset
    try:
        dataset = load_dbbio_dataset(args.split)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n💡 Téléchargez d'abord le dataset avec : python download_datasets.py")
        return
    
    # Limiter au nombre demandé
    dataset = dataset[:args.n_samples]
    
    # Client OpenRouter
    client = OpenRouterClient.from_config()
    
    # Évaluation
    results = []
    print(f"\n🔄 Évaluation en cours...")
    
    for i, sample in enumerate(tqdm(dataset, desc="Échantillons")):
        try:
            result = evaluate_sample(
                sample=sample,
                client=client,
                use_rupta=not args.use_baseline,
                p_threshold=args.p_threshold
            )
            results.append(result)
            
            # Afficher progression
            if (i + 1) % 5 == 0:
                interim_metrics = compute_metrics(results)
                print(f"\n  Métriques intermédiaires ({i+1} échantillons):")
                print(f"    Privacy rank moyen: {interim_metrics['avg_privacy_rank']:.2f}" if interim_metrics['avg_privacy_rank'] else "    Privacy rank: N/A")
                print(f"    Non-identifié: {interim_metrics['privacy_not_identified_rate']:.2%}")
                print(f"    Utility confidence: {interim_metrics['avg_utility_confidence']:.1f}%" if interim_metrics['avg_utility_confidence'] else "    Utility: N/A")
        
        except Exception as e:
            print(f"\n⚠️  Erreur sur échantillon {i}: {e}")
            continue
    
    # Calculer métriques finales
    metrics = compute_metrics(results)
    
    # Afficher résultats
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"Méthode: {'Baseline' if args.use_baseline else 'RUPTA'}")
    print(f"\nPrivacy:")
    print(f"  Rang moyen: {metrics['avg_privacy_rank']:.2f}" if metrics['avg_privacy_rank'] else "  Rang moyen: N/A")
    print(f"  Taux non-identifié: {metrics['privacy_not_identified_rate']:.2%}")
    print(f"\nUtility:")
    print(f"  Confiance moyenne: {metrics['avg_utility_confidence']:.1f}%" if metrics['avg_utility_confidence'] else "  Confiance moyenne: N/A")
    print(f"  Taux préservation: {metrics['utility_preserved_rate']:.2%}")
    print(f"\nÉchantillons: {metrics['n_samples']}")
    
    # Sauvegarder résultats
    output = {
        "config": {
            "split": args.split,
            "n_samples": args.n_samples,
            "method": "Baseline" if args.use_baseline else "RUPTA",
            "p_threshold": args.p_threshold
        },
        "metrics": metrics,
        "results": results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résultats sauvegardés dans {args.output}")
    print("\n✅ Évaluation terminée !")


if __name__ == "__main__":
    main()
