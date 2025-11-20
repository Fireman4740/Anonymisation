#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de l'intégration GPU dans l'orchestrateur

Vérifie que le pipeline GPU est correctement utilisé par l'orchestrateur
lors de l'anonymisation.
"""
import sys
import os
import time

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.orchestrator import anonymize_text, get_ner_mode, reset_ner_pipeline


def test_integration_orchestrator():
    """Test d'intégration avec l'orchestrateur."""
    print("\n" + "="*80)
    print("🧪 Test d'Intégration GPU - Orchestrateur")
    print("="*80 + "\n")
    
    # Texte de test
    text = """
    Rapport d'incident - Réf : INC-2025-001
    
    Le 24/10/2025, Jean Dupont (jean.dupont@acme.fr) a signalé un problème.
    Contact : +33 1 42 68 53 00
    Société : Acme Corporation, 123 Rue de la Paix, 75002 Paris
    
    Marie Martin et Pierre Dubois ont collaboré sur le ticket.
    """
    
    print(f"📄 Texte de test : {len(text)} caractères\n")
    
    # Vérifier le mode NER actuel
    mode = get_ner_mode()
    print(f"🔍 Mode NER détecté : {mode.upper()}\n")
    
    if mode == "gpu":
        print("✅ Le mode GPU est activé dans l'orchestrateur")
    else:
        print("⚠️  Le mode standard est utilisé")
        print("   Pour activer le GPU : config.json -> ner_gpu.enabled=true")
    
    print("\n" + "-"*80)
    print("🔄 Anonymisation en cours...")
    print("-"*80 + "\n")
    
    # Anonymisation
    start = time.time()
    
    result = anonymize_text(
        value=text,
        scope_id="test_gpu_001",
        secret_salt="test_secret",
        level="L0",  # Niveau L0 utilise le NER
    )
    
    elapsed = time.time() - start
    
    # Afficher les résultats
    print("✅ Anonymisation terminée\n")
    print(f"⏱️  Temps : {elapsed:.3f}s")
    print(f"🔍 Mode utilisé : {mode.upper()}")
    print(f"📊 Entités détectées : {len(result['audit']['entities'])}\n")
    
    print("📝 Texte anonymisé :")
    print("-"*80)
    print(result["anonymized_text"])
    print("-"*80)
    
    print("\n🏷️  Entités détectées :")
    for i, entity in enumerate(result['audit']['entities'][:10], 1):  # Limiter à 10
        print(f"  {i}. {entity.get('etype'):<15} : {entity.get('surface'):<30} → {entity.get('replacement')}")
    
    if len(result['audit']['entities']) > 10:
        print(f"  ... et {len(result['audit']['entities']) - 10} autres")
    
    print("\n" + "="*80)
    print("✅ Test terminé avec succès")
    print("="*80)
    
    return result


def test_mode_comparison():
    """Compare les performances entre mode standard et GPU."""
    print("\n" + "="*80)
    print("📊 Comparaison Mode Standard vs GPU")
    print("="*80 + "\n")
    
    text = """
    Jean Dupont (jean.dupont@acme.fr) travaille chez Acme Corporation à Paris.
    Il collabore avec Marie Martin (marie.martin@acme.fr) du Marketing et 
    Pierre Dubois (p.dubois@acme.fr) des RH.
    
    Coordonnées : +33 1 42 68 53 00
    Adresse : 123 Avenue des Champs-Élysées, 75008 Paris
    SIRET : 123 456 789 00012
    """
    
    # Test avec le mode actuel
    mode_initial = get_ner_mode()
    print(f"Mode actuel : {mode_initial.upper()}\n")
    
    print("🔄 Anonymisation...")
    start = time.time()
    result = anonymize_text(
        value=text,
        scope_id="compare_001",
        secret_salt="compare_secret",
        level="L0",
    )
    elapsed = time.time() - start
    
    print(f"✅ Terminé en {elapsed:.3f}s")
    print(f"📊 {len(result['audit']['entities'])} entités détectées")
    
    print("\n💡 Pour comparer avec l'autre mode :")
    if mode_initial == "gpu":
        print("   1. Désactiver GPU : config.json -> ner_gpu.enabled=false")
        print("   2. Relancer ce script")
    else:
        print("   1. Activer GPU : config.json -> ner_gpu.enabled=true")
        print("   2. Relancer ce script")
    
    print("\nOu utilisez :")
    print("   python scripts/benchmark_ner_gpu.py --mode both")


def test_reset_pipeline():
    """Test de réinitialisation du pipeline."""
    print("\n" + "="*80)
    print("🔄 Test de Réinitialisation du Pipeline")
    print("="*80 + "\n")
    
    mode_before = get_ner_mode()
    print(f"Mode avant reset : {mode_before}")
    
    reset_ner_pipeline()
    
    mode_after = get_ner_mode()
    print(f"Mode après reset : {mode_after}")
    
    print("\n✅ Pipeline réinitialisé")


def main():
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  🧪 Tests d'Intégration GPU - Orchestrateur                  ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    print("\nChoisissez un test :")
    print("  1. Test d'intégration basique")
    print("  2. Comparaison Standard vs GPU")
    print("  3. Test de réinitialisation")
    print("  4. Tous les tests")
    print("  0. Quitter")
    
    choice = input("\nVotre choix (0-4) : ").strip()
    
    if choice == "1":
        test_integration_orchestrator()
    elif choice == "2":
        test_mode_comparison()
    elif choice == "3":
        test_reset_pipeline()
    elif choice == "4":
        test_integration_orchestrator()
        test_mode_comparison()
        test_reset_pipeline()
    elif choice == "0":
        print("\nAu revoir ! 👋")
        return
    else:
        print("\n❌ Choix invalide")
        return
    
    print("\n📚 Documentation : docs/NER_GPU_OPTIMIZATION.md\n")


if __name__ == "__main__":
    main()
