#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple d'utilisation du mode NER GPU optimisé

Démontre comment utiliser le nouveau pipeline GPU pour l'anonymisation
avec des performances 6-10x supérieures au mode standard.
"""
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ner_gpu_optimizer import create_optimized_pipeline, load_gpu_config


def example_1_basic():
    """Exemple 1 : Utilisation basique du pipeline GPU."""
    print("\n" + "="*80)
    print("📝 Exemple 1 : Utilisation Basique")
    print("="*80 + "\n")
    
    # Texte de test
    text = """
    Jean Dupont (jean.dupont@acme.fr) travaille chez Acme Corporation à Paris.
    Il collabore avec Marie Martin (marie.martin@acme.fr) et Pierre Dubois.
    Leur adresse est 123 Avenue des Champs-Élysées, 75008 Paris.
    Téléphone : +33 1 42 68 53 00.
    """
    
    print("📄 Texte original :")
    print(text)
    
    # Créer le pipeline GPU
    print("\n🔧 Création du pipeline GPU...")
    config = load_gpu_config()
    config["enabled"] = True  # Forcer l'activation pour cet exemple
    
    pipeline = create_optimized_pipeline(config)
    
    if pipeline is None:
        print("❌ Impossible de créer le pipeline GPU (GPU non disponible)")
        return
    
    print("✅ Pipeline créé avec succès")
    
    # Prédiction
    print("\n🔍 Détection des entités...")
    entities = pipeline.predict(text)
    
    print(f"\n✅ {len(entities)} entités détectées :\n")
    for ent in entities:
        surface = text[ent["start"]:ent["end"]]
        print(f"  • {ent['entity_group']:<15} : '{surface}' (votes={ent['votes']:.1f})")


def example_2_config():
    """Exemple 2 : Configuration personnalisée."""
    print("\n" + "="*80)
    print("📝 Exemple 2 : Configuration Personnalisée")
    print("="*80 + "\n")
    
    # Configuration personnalisée
    custom_config = {
        "enabled": True,
        "vram_gb": 24,
        "batch_size": 32,  # Batch size réduit
        "max_parallel_models": 2,  # Seulement 2 modèles en parallèle
        "use_fp16": True,
        "use_torch_compile": False,
        "gliner_preset": "balanced",  # Preset plus léger
        "prefetch_models": False,  # Pas de warm-up
        "optimization_level": "medium",
    }
    
    print("⚙️  Configuration :")
    for key, value in custom_config.items():
        print(f"  - {key}: {value}")
    
    # Créer le pipeline
    print("\n🔧 Création du pipeline...")
    pipeline = create_optimized_pipeline(custom_config)
    
    if pipeline is None:
        print("❌ Pipeline non créé")
        return
    
    print("✅ Pipeline créé avec configuration personnalisée")
    
    # Test
    text = "Marie Curie, physicienne française, a reçu deux prix Nobel."
    entities = pipeline.predict(text)
    
    print(f"\n✅ {len(entities)} entités détectées")


def example_3_comparison():
    """Exemple 3 : Comparaison avec le mode standard."""
    print("\n" + "="*80)
    print("📝 Exemple 3 : Comparaison Mode Standard vs GPU")
    print("="*80 + "\n")
    
    import time
    from src.ner_ensemble import run_gliner, warm_up_models
    
    text = """
    Rapport d'incident - Réf : INC-2025-001
    
    Le 24/10/2025 à 14h30, Jean Dupont (jean.dupont@acme.fr) a signalé
    un problème de connexion depuis l'IP 192.168.1.100. 
    
    Marie Martin de l'équipe support a pris en charge le ticket.
    Contact client : +33 1 42 68 53 00
    Société : Acme Corporation SAS, 123 Rue de la Paix, 75002 Paris
    SIRET : 123 456 789 00012
    """
    
    print(f"📄 Texte de test : {len(text)} caractères\n")
    
    # Mode Standard
    print("🔵 Mode Standard (séquentiel)")
    warm_up_models(gliner_preset="balanced", load_hf=False)
    
    start = time.time()
    entities_std = run_gliner(text, preset="balanced", threshold=0.35)
    time_std = time.time() - start
    
    print(f"   ✅ Temps : {time_std:.3f}s | Entités : {len(entities_std)}")
    
    # Mode GPU
    print("\n🟢 Mode GPU (parallèle)")
    config = load_gpu_config()
    config["enabled"] = True
    config["gliner_preset"] = "balanced"
    
    pipeline = create_optimized_pipeline(config)
    
    if pipeline is None:
        print("   ❌ GPU non disponible")
        return
    
    start = time.time()
    entities_gpu = pipeline.predict(text)
    time_gpu = time.time() - start
    
    print(f"   ✅ Temps : {time_gpu:.3f}s | Entités : {len(entities_gpu)}")
    
    # Comparaison
    speedup = time_std / time_gpu if time_gpu > 0 else 0
    print(f"\n🚀 Speedup : {speedup:.2f}x plus rapide avec le mode GPU")
    
    if speedup > 1:
        improvement = ((speedup - 1) * 100)
        print(f"✅ Amélioration de {improvement:.1f}% des performances")


def example_4_integration():
    """Exemple 4 : Intégration avec l'orchestrateur."""
    print("\n" + "="*80)
    print("📝 Exemple 4 : Intégration avec l'Orchestrateur")
    print("="*80 + "\n")
    
    print("ℹ️  L'orchestrateur détecte automatiquement le mode GPU")
    print("   si `ner_gpu.enabled=true` dans config.json\n")
    
    # Simuler l'utilisation via l'orchestrateur
    print("💡 Usage recommandé :\n")
    print("```python")
    print("from src.orchestrator import anonymize_text")
    print("")
    print("# Le mode GPU est utilisé automatiquement si activé")
    print("result = anonymize_text(")
    print("    value='Jean Dupont travaille chez Acme Corp...',")
    print("    scope_id='example_001',")
    print("    secret_salt='my_secret',")
    print("    level='L0',  # Niveau L0 utilise NER")
    print(")")
    print("")
    print("print(result['anonymized_text'])")
    print("print(f\"Entités détectées : {len(result['entities'])}\")")
    print("```")


def main():
    """Lancer tous les exemples."""
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  🚀 Exemples d'Utilisation - Mode NER GPU Optimisé           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    print("\nChoisissez un exemple :")
    print("  1. Utilisation basique")
    print("  2. Configuration personnalisée")
    print("  3. Comparaison Standard vs GPU")
    print("  4. Intégration avec orchestrateur")
    print("  5. Tous les exemples")
    print("  0. Quitter")
    
    choice = input("\nVotre choix (0-5) : ").strip()
    
    if choice == "1":
        example_1_basic()
    elif choice == "2":
        example_2_config()
    elif choice == "3":
        example_3_comparison()
    elif choice == "4":
        example_4_integration()
    elif choice == "5":
        example_1_basic()
        example_2_config()
        example_3_comparison()
        example_4_integration()
    elif choice == "0":
        print("\nAu revoir ! 👋")
        return
    else:
        print("\n❌ Choix invalide")
        return
    
    print("\n" + "="*80)
    print("✅ Exemples terminés !")
    print("="*80)
    print("\n📚 Pour plus d'informations : docs/NER_GPU_OPTIMIZATION.md\n")


if __name__ == "__main__":
    main()
