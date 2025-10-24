#!/usr/bin/env python3
"""
Test rapide de l'intégration RUPTA.

Vérifie que:
1. Le pipeline fonctionne avec L0 (baseline)
2. Le pipeline fonctionne avec L1 (RUPTA)
3. RUPTA améliore la privacy
4. L'utilité est préservée
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import anonymize_text
from src.openrouter_client import OpenRouterClient
from src.rupta.privacy_evaluator import evaluate_reidentification_risk
from src.rupta.utility_evaluator import evaluate_utility_preservation

# Textes de test
TEST_CASES = [
    {
        "text": "Marie Curie, physicienne française, a reçu deux prix Nobel pour ses travaux sur la radioactivité.",
        "people": "Marie Curie",
        "label": "physicist"
    },
    {
        "text": "Albert Einstein, born in Germany, developed the theory of relativity while working at the patent office.",
        "people": "Albert Einstein",
        "label": "physicist"
    },
    {
        "text": "Leonardo da Vinci fue un pintor, escultor e inventor italiano del Renacimiento.",
        "people": "Leonardo da Vinci",
        "label": "artist"
    }
]


def test_baseline(case):
    """Test avec L0 (baseline, sans RUPTA)."""
    print("\n--- Test BASELINE (L0) ---")
    print(f"Original: {case['text'][:80]}...")
    
    result = anonymize_text(
        value=case["text"],
        scope_id="test_baseline",
        secret_salt="test_secret",
        level="L0"  # Baseline
    )
    
    anonymized = result.get("anonymized_text", case["text"])
    print(f"Anonymisé: {anonymized[:80]}...")
    
    # Évaluer manuellement
    client = OpenRouterClient()
    
    privacy = evaluate_reidentification_risk(
        client=client,
        anonymized_text=anonymized,
        ground_truth_people=case["people"],
        p_threshold=10
    )
    
    utility = evaluate_utility_preservation(
        client=client,
        anonymized_text=anonymized,
        ground_truth_label=case["label"]
    )
    
    print(f"Privacy Rank: {privacy.get('rank', 'N/A')}")
    print(f"Utility Score: {utility.get('confidence_score', 'N/A')}")
    
    return {
        "anonymized": anonymized,
        "privacy_rank": privacy.get("rank"),
        "utility_score": utility.get("confidence_score")
    }


def test_rupta(case):
    """Test avec L1 (RUPTA activé)."""
    print("\n--- Test RUPTA (L1) ---")
    print(f"Original: {case['text'][:80]}...")
    
    result = anonymize_text(
        value=case["text"],
        scope_id="test_rupta",
        secret_salt="test_secret",
        level="L1",  # RUPTA
        overrides={
            "rupta_ground_truth_people": case["people"],
            "rupta_ground_truth_label": case["label"]
        }
    )
    
    anonymized = result.get("anonymized_text", case["text"])
    rupta_metrics = result.get("rupta_metrics", {})
    
    print(f"Anonymisé: {anonymized[:80]}...")
    print(f"Privacy Rank: {rupta_metrics.get('privacy', {}).get('rank', 'N/A')}")
    print(f"Utility Score: {rupta_metrics.get('utility', {}).get('confidence_score', 'N/A')}")
    print(f"Iterations: {rupta_metrics.get('iterations', 'N/A')}")
    print(f"Final Reward: {rupta_metrics.get('final_reward', 'N/A'):.2f}" if rupta_metrics.get('final_reward') else "Final Reward: N/A")
    
    return {
        "anonymized": anonymized,
        "privacy_rank": rupta_metrics.get("privacy", {}).get("rank"),
        "utility_score": rupta_metrics.get("utility", {}).get("confidence_score"),
        "iterations": rupta_metrics.get("iterations"),
        "reward": rupta_metrics.get("final_reward")
    }


def compare_results(baseline, rupta):
    """Compare les résultats baseline vs RUPTA."""
    print("\n" + "="*60)
    print("COMPARAISON")
    print("="*60)
    
    b_rank = baseline.get("privacy_rank") or 0
    r_rank = rupta.get("privacy_rank") or 0
    
    b_util = baseline.get("utility_score") or 0
    r_util = rupta.get("utility_score") or 0
    
    print(f"Privacy Rank:   Baseline={b_rank}, RUPTA={r_rank}")
    if r_rank > b_rank or r_rank == 999:
        print("  ✅ RUPTA améliore la privacy")
    else:
        print("  ❌ RUPTA n'améliore pas la privacy")
    
    print(f"Utility Score:  Baseline={b_util}, RUPTA={r_util}")
    if r_util >= b_util - 10:  # Tolérance de 10 points
        print("  ✅ Utilité préservée")
    else:
        print("  ❌ Perte d'utilité importante")
    
    print(f"Iterations RUPTA: {rupta.get('iterations', 'N/A')}")
    print(f"Final Reward: {rupta.get('reward', 0):.2f}" if rupta.get('reward') else "Final Reward: N/A")


def main():
    print("="*60)
    print("TEST RUPTA INTEGRATION")
    print("="*60)
    
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*60}")
        print(f"CAS DE TEST {i}/3")
        print(f"Langue: {'FR' if 'français' in case['text'] else 'EN' if 'born' in case['text'] else 'ES'}")
        print(f"{'='*60}")
        
        try:
            baseline = test_baseline(case)
            rupta = test_rupta(case)
            compare_results(baseline, rupta)
        except Exception as e:
            print(f"❌ ERREUR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("FIN DES TESTS")
    print("="*60)


if __name__ == "__main__":
    main()
