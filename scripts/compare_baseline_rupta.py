"""
Script de comparaison Baseline vs RUPTA

Compare les performances du système d'anonymisation :
- Baseline : Système actuel (regex + NER + LLM)
- RUPTA : Optimisation itérative privacy-utility

Génère un rapport comparatif avec visualisations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_results(filepath: str) -> Dict[str, Any]:
    """Charge les résultats depuis un fichier JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_metrics(baseline: Dict[str, Any], rupta: Dict[str, Any]) -> Dict[str, Any]:
    """Compare les métriques de deux runs"""
    
    baseline_metrics = baseline['metrics']
    rupta_metrics = rupta['metrics']
    
    comparison = {
        "privacy": {
            "baseline_rank": baseline_metrics.get('avg_privacy_rank'),
            "rupta_rank": rupta_metrics.get('avg_privacy_rank'),
            "improvement": None,
            "baseline_not_identified": baseline_metrics.get('privacy_not_identified_rate'),
            "rupta_not_identified": rupta_metrics.get('privacy_not_identified_rate'),
            "improvement_rate": None
        },
        "utility": {
            "baseline_confidence": baseline_metrics.get('avg_utility_confidence'),
            "rupta_confidence": rupta_metrics.get('avg_utility_confidence'),
            "degradation": None,
            "baseline_preserved": baseline_metrics.get('utility_preserved_rate'),
            "rupta_preserved": rupta_metrics.get('utility_preserved_rate'),
            "degradation_rate": None
        }
    }
    
    # Calculer les améliorations/dégradations
    if baseline_metrics.get('avg_privacy_rank') and rupta_metrics.get('avg_privacy_rank'):
        comparison["privacy"]["improvement"] = (
            baseline_metrics['avg_privacy_rank'] - rupta_metrics['avg_privacy_rank']
        )
    
    if baseline_metrics.get('privacy_not_identified_rate') is not None and rupta_metrics.get('privacy_not_identified_rate') is not None:
        comparison["privacy"]["improvement_rate"] = (
            rupta_metrics['privacy_not_identified_rate'] - baseline_metrics['privacy_not_identified_rate']
        )
    
    if baseline_metrics.get('avg_utility_confidence') and rupta_metrics.get('avg_utility_confidence'):
        comparison["utility"]["degradation"] = (
            baseline_metrics['avg_utility_confidence'] - rupta_metrics['avg_utility_confidence']
        )
    
    if baseline_metrics.get('utility_preserved_rate') is not None and rupta_metrics.get('utility_preserved_rate') is not None:
        comparison["utility"]["degradation_rate"] = (
            baseline_metrics['utility_preserved_rate'] - rupta_metrics['utility_preserved_rate']
        )
    
    return comparison


def generate_report(comparison: Dict[str, Any], output_file: str = "comparison_report.md"):
    """Génère un rapport markdown"""
    
    report = []
    report.append("# 📊 Rapport de Comparaison : Baseline vs RUPTA\n")
    report.append("## Métriques de Privacy\n")
    
    privacy = comparison["privacy"]
    
    report.append("### Re-identification Rank\n")
    report.append(f"- **Baseline** : {privacy['baseline_rank']:.2f}" if privacy['baseline_rank'] else "- **Baseline** : N/A")
    report.append(f"- **RUPTA** : {privacy['rupta_rank']:.2f}" if privacy['rupta_rank'] else "- **RUPTA** : N/A")
    
    if privacy['improvement']:
        sign = "✅" if privacy['improvement'] > 0 else "⚠️"
        report.append(f"- **Amélioration** : {sign} {privacy['improvement']:.2f}\n")
    
    report.append("### Taux Non-Identifié\n")
    report.append(f"- **Baseline** : {privacy['baseline_not_identified']:.2%}" if privacy['baseline_not_identified'] is not None else "- **Baseline** : N/A")
    report.append(f"- **RUPTA** : {privacy['rupta_not_identified']:.2%}" if privacy['rupta_not_identified'] is not None else "- **RUPTA** : N/A")
    
    if privacy['improvement_rate'] is not None:
        sign = "✅" if privacy['improvement_rate'] > 0 else "⚠️"
        report.append(f"- **Amélioration** : {sign} {privacy['improvement_rate']:.2%}\n")
    
    report.append("\n## Métriques d'Utility\n")
    
    utility = comparison["utility"]
    
    report.append("### Confidence Score Moyen\n")
    report.append(f"- **Baseline** : {utility['baseline_confidence']:.1f}%" if utility['baseline_confidence'] else "- **Baseline** : N/A")
    report.append(f"- **RUPTA** : {utility['rupta_confidence']:.1f}%" if utility['rupta_confidence'] else "- **RUPTA** : N/A")
    
    if utility['degradation']:
        sign = "⚠️" if utility['degradation'] > 0 else "✅"
        report.append(f"- **Dégradation** : {sign} {utility['degradation']:.1f}%\n")
    
    report.append("### Taux Préservation\n")
    report.append(f"- **Baseline** : {utility['baseline_preserved']:.2%}" if utility['baseline_preserved'] is not None else "- **Baseline** : N/A")
    report.append(f"- **RUPTA** : {utility['rupta_preserved']:.2%}" if utility['rupta_preserved'] is not None else "- **RUPTA** : N/A")
    
    if utility['degradation_rate'] is not None:
        sign = "⚠️" if utility['degradation_rate'] > 0 else "✅"
        report.append(f"- **Dégradation** : {sign} {utility['degradation_rate']:.2%}\n")
    
    report.append("\n## 🎯 Conclusion\n")
    
    # Analyse du compromis
    privacy_improved = privacy.get('improvement_rate', 0) > 0 if privacy.get('improvement_rate') is not None else False
    utility_degraded = utility.get('degradation_rate', 0) > 0 if utility.get('degradation_rate') is not None else False
    
    if privacy_improved and not utility_degraded:
        report.append("✅ **RUPTA améliore la privacy sans dégrader l'utility** - Configuration optimale !\n")
    elif privacy_improved and utility_degraded:
        report.append("⚖️  **RUPTA améliore la privacy mais dégrade légèrement l'utility** - Compromis acceptable.\n")
    elif not privacy_improved and not utility_degraded:
        report.append("ℹ️  **Pas d'amélioration significative** - Ajuster les paramètres RUPTA.\n")
    else:
        report.append("❌ **RUPTA dégrade à la fois privacy et utility** - Revoir la configuration.\n")
    
    report.append("\n## 📌 Recommandations\n")
    
    if privacy_improved:
        report.append("- ✅ RUPTA efficace pour améliorer la privacy\n")
        report.append("- 💡 Considérer augmenter `max_iterations` pour plus d'amélioration\n")
    else:
        report.append("- ⚠️  Augmenter `p_threshold` pour privacy evaluation plus stricte\n")
    
    if utility_degraded:
        report.append("- ⚠️  Réduire `privacy_threshold` pour limiter la dégradation d'utility\n")
        report.append("- 💡 Ajuster le prompt de refinement pour mieux préserver l'utilité\n")
    
    # Sauvegarder le rapport
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Rapport généré : {output_file}")
    
    # Afficher aussi dans le terminal
    print("\n" + "=" * 60)
    print('\n'.join(report))
    print("=" * 60)


def compare_individual_results(baseline_results: List[Dict], rupta_results: List[Dict]):
    """Compare les résultats individuels"""
    
    improvements = []
    degradations = []
    
    for b, r in zip(baseline_results, rupta_results):
        # Privacy
        b_rank = b.get('privacy_rank')
        r_rank = r.get('privacy_rank')
        
        privacy_better = False
        if b_rank is not None and r_rank is None:
            privacy_better = True
        elif b_rank is not None and r_rank is not None and r_rank > b_rank:
            privacy_better = True
        
        # Utility
        b_util = b.get('utility_confidence', 0)
        r_util = r.get('utility_confidence', 0)
        utility_worse = r_util < b_util
        
        if privacy_better and not utility_worse:
            improvements.append({
                "text": b.get('original', '')[:100],
                "baseline_rank": b_rank,
                "rupta_rank": r_rank,
                "baseline_util": b_util,
                "rupta_util": r_util
            })
        elif privacy_better and utility_worse:
            degradations.append({
                "text": b.get('original', '')[:100],
                "baseline_rank": b_rank,
                "rupta_rank": r_rank,
                "baseline_util": b_util,
                "rupta_util": r_util
            })
    
    print(f"\n📈 Exemples d'amélioration pure (privacy + utility) : {len(improvements)}")
    for i, ex in enumerate(improvements[:3]):
        print(f"\n{i+1}. {ex['text']}...")
        print(f"   Privacy: {ex['baseline_rank']} → {ex['rupta_rank']}")
        print(f"   Utility: {ex['baseline_util']:.1f}% → {ex['rupta_util']:.1f}%")
    
    print(f"\n⚖️  Exemples de compromis (privacy vs utility) : {len(degradations)}")
    for i, ex in enumerate(degradations[:3]):
        print(f"\n{i+1}. {ex['text']}...")
        print(f"   Privacy: {ex['baseline_rank']} → {ex['rupta_rank']}")
        print(f"   Utility: {ex['baseline_util']:.1f}% → {ex['rupta_util']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Comparaison Baseline vs RUPTA")
    parser.add_argument("--baseline", type=str, default="results_dbbio_baseline.json", 
                       help="Fichier résultats baseline")
    parser.add_argument("--rupta", type=str, default="results_dbbio_rupta.json",
                       help="Fichier résultats RUPTA")
    parser.add_argument("--output", type=str, default="comparison_report.md",
                       help="Fichier de sortie du rapport")
    parser.add_argument("--detailed", action="store_true",
                       help="Afficher comparaison détaillée par exemple")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Comparaison Baseline vs RUPTA")
    print("=" * 60)
    
    # Vérifier les fichiers
    if not os.path.exists(args.baseline):
        print(f"❌ Fichier baseline non trouvé : {args.baseline}")
        print("\n💡 Exécutez d'abord :")
        print(f"   python eval_rupta_dbbio.py --use_baseline --output {args.baseline}")
        return
    
    if not os.path.exists(args.rupta):
        print(f"❌ Fichier RUPTA non trouvé : {args.rupta}")
        print("\n💡 Exécutez d'abord :")
        print(f"   python eval_rupta_dbbio.py --output {args.rupta}")
        return
    
    # Charger les résultats
    print(f"\n📂 Chargement des résultats...")
    baseline = load_results(args.baseline)
    rupta = load_results(args.rupta)
    
    print(f"   Baseline : {baseline['metrics']['n_samples']} échantillons")
    print(f"   RUPTA : {rupta['metrics']['n_samples']} échantillons")
    
    # Comparer
    comparison = compare_metrics(baseline, rupta)
    
    # Générer rapport
    generate_report(comparison, args.output)
    
    # Comparaison détaillée
    if args.detailed:
        print("\n" + "=" * 60)
        print("ANALYSE DÉTAILLÉE")
        print("=" * 60)
        compare_individual_results(baseline['results'], rupta['results'])
    
    print("\n✅ Comparaison terminée !")


if __name__ == "__main__":
    main()
