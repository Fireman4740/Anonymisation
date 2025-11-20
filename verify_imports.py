#!/usr/bin/env python3
"""
Script de vérification des imports après restructuration.

Ce script vérifie que tous les imports fonctionnent correctement
avec la nouvelle structure modulaire.
"""

import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Teste tous les imports critiques."""
    
    errors = []
    success = []
    
    print("🔍 Vérification des imports...\n")
    
    # Test 1: Core imports
    print("1️⃣  Test imports core...")
    try:
        from src.core import anonymize_text, AnonymizationPolicy, load_config
        success.append("✅ Core imports (orchestrator, policy, config_loader)")
    except Exception as e:
        errors.append(f"❌ Core imports: {e}")
    
    # Test 2: Services imports
    print("2️⃣  Test imports services...")
    try:
        from src.services import DetectionService, GeneralizationService, LLMPipelineService
        success.append("✅ Services imports (detectors, generalizers, llm_pipeline)")
    except Exception as e:
        errors.append(f"❌ Services imports: {e}")
    
    # Test 3: NER imports
    print("3️⃣  Test imports NER...")
    try:
        from src.services.ner import run_gliner, merge_ner_lists, warm_up_models
        success.append("✅ NER imports (ensemble)")
    except Exception as e:
        errors.append(f"❌ NER imports: {e}")
    
    # Test 4: LLM imports
    print("4️⃣  Test imports LLM...")
    try:
        from src.llm import OpenRouterClient
        success.append("✅ LLM imports (openrouter_client, reasoner)")
    except Exception as e:
        errors.append(f"❌ LLM imports: {e}")
    
    # Test 5: Utils imports
    print("5️⃣  Test imports utils...")
    try:
        from src.utils import PseudoMapper, WHITELIST_WORDS
        success.append("✅ Utils imports (personal_info, text_sanitizer, utils_pseudo)")
    except Exception as e:
        errors.append(f"❌ Utils imports: {e}")
    
    # Test 6: RUPTA imports
    print("6️⃣  Test imports RUPTA...")
    try:
        from src.rupta import optimize_anonymization
        success.append("✅ RUPTA imports (optimizer, evaluators)")
    except Exception as e:
        errors.append(f"❌ RUPTA imports: {e}")
    
    # Test 7: API imports
    print("7️⃣  Test imports API...")
    try:
        from api import create_app
        success.append("✅ API imports")
    except Exception as e:
        errors.append(f"❌ API imports: {e}")
    
    # Résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS")
    print("="*60 + "\n")
    
    for msg in success:
        print(msg)
    
    if errors:
        print()
        for msg in errors:
            print(msg)
        print(f"\n❌ {len(errors)} erreur(s) détectée(s)")
        return False
    else:
        print(f"\n✅ Tous les imports fonctionnent correctement ({len(success)} tests)")
        return True

def main():
    """Point d'entrée principal."""
    print("🏗️  Vérification Imports - Architecture v2.0\n")
    
    success = test_imports()
    
    if success:
        print("\n✅ La restructuration est complète et fonctionnelle!")
        print("   Vous pouvez maintenant utiliser les nouveaux chemins d'import.")
        sys.exit(0)
    else:
        print("\n⚠️  Certains imports nécessitent des corrections.")
        print("   Voir RESTRUCTURATION_COMPLETE.md pour les prochaines étapes.")
        sys.exit(1)

if __name__ == "__main__":
    main()
