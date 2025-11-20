
"""
Exemple Simple d'Utilisation

Démontre les fonctionnalités de base du pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api import anonymize_text, AnonymizationPipeline


def example_basic():
    """Exemple de base."""
    print("="*60)
    print("EXEMPLE 1: Anonymisation Basique (L0)")
    print("="*60)
    
    text = """
Jean Dupont habite à Paris.
Email: jean.dupont@example.com
Téléphone: +33 6 12 34 56 78
"""
    
    result = anonymize_text(
        text,
        level="L0",
        secret_salt="demo_secret"
    )
    
    print("\nTexte Original:")
    print(text)
    
    print("\nTexte Anonymisé:")
    print(result["anonymized_text"])
    
    print("\nEntités Détectées:")
    for entity in result["audit"]["entities"][:5]:  # Limiter à 5
        print(f"  - {entity['etype']:12s} : {entity['surface']:30s} (source: {entity['source']})")
    
    print(f"\nMétriques:")
    print(f"  - Entités détectées: {result['evaluation']['metrics']['entities_detected']}")
    print(f"  - Entités remplacées: {result['evaluation']['metrics']['entities_replaced']}")
    print(f"  - Ratio de longueur: {result['evaluation']['metrics']['length_ratio']:.2f}")


def example_pipeline():
    """Exemple avec pipeline réutilisable."""
    print("\n\n" + "="*60)
    print("EXEMPLE 2: Pipeline Réutilisable")
    print("="*60)
    
    pipeline = AnonymizationPipeline(
        level="L0",
        secret_salt="production_secret",
        date_granularity="month"
    )
    
    documents = [
        "Alice Martin travaille chez Google",
        "Bob Johnson habite à Lyon",
        "Contact: charlie@example.com"
    ]
    
    print("\nTraitement de plusieurs documents:")
    for i, doc in enumerate(documents, 1):
        result = pipeline.anonymize(doc, scope_id=f"doc_{i}")
        print(f"\n{i}. Original: {doc}")
        print(f"   Anonymisé: {result['anonymized_text']}")


def example_custom_policy():
    """Exemple avec policy personnalisée."""
    print("\n\n" + "="*60)
    print("EXEMPLE 3: Policy Personnalisée")
    print("="*60)
    
    # Policy très stricte
    result = anonymize_text(
        "Alice (alice@test.com) a travaillé avec Bob en 2024.",
        level="L2",
        secret_salt="strict_secret",
        placeholder_style="generic",
        org_policy="redact"
    )
    
    print("\nTexte Original:")
    print("Alice (alice@test.com) a travaillé avec Bob en 2024.")
    
    print("\nTexte Anonymisé (L2 - Maximum):")
    print(result["anonymized_text"])
    
    print("\nPolicy Utilisée:")
    policy = result["policy"]
    print(f"  - Niveau: {policy['level']}")
    print(f"  - Style placeholders: {policy['placeholder_style']}")
    print(f"  - Granularité dates: {policy['date_granularity']}")
    print(f"  - Policy orgs: {policy['org_policy']}")


def example_validation():
    """Exemple de validation."""
    print("\n\n" + "="*60)
    print("EXEMPLE 4: Validation et Métriques")
    print("="*60)
    
    result = anonymize_text(
        "Alice Martin: alice@test.com",
        level="L0",
        secret_salt="validation_test"
    )
    
    evaluation = result["evaluation"]
    
    print(f"\n✓ Validation: {'SUCCÈS' if evaluation['is_valid'] else 'ÉCHEC'}")
    
    if evaluation["warnings"]:
        print("\n⚠️ Avertissements:")
        for warning in evaluation["warnings"]:
            print(f"  - {warning}")
    
    if evaluation["validation_errors"]:
        print("\n❌ Erreurs:")
        for error in evaluation["validation_errors"]:
            print(f"  - {error}")
    
    print("\n📊 Métriques détaillées:")
    for key, value in evaluation["metrics"].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    print("\n" + "🔒 PIPELINE D'ANONYMISATION - EXEMPLES".center(60))
    print("Version 2.0\n")
    
    try:
        example_basic()
        example_pipeline()
        example_custom_policy()
        example_validation()
        
        print("\n\n" + "="*60)
        print("✅ Exemples terminés avec succès!")
        print("="*60)
        print("\nPour plus d'informations:")
        print("  - Documentation complète: docs/")
        print("  - API Reference: docs/API_REFERENCE.md")
        print("  - Quickstart: docs/QUICKSTART.md")
        print("")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
