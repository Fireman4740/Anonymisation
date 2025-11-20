"""
Script d'exemple pour tester le module RUPTA

Ce script montre comment :
1. Charger le module RUPTA
2. Évaluer la privacy d'un texte anonymisé
3. Évaluer l'utility
4. Utiliser la boucle d'optimisation
"""

import os
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.openrouter_client import OpenRouterClient
from src.rupta.privacy_evaluator import evaluate_reidentification_risk, evaluate_confidence_score
from src.rupta.utility_evaluator import evaluate_classification_utility
from src.rupta.optimizer import optimize_anonymization


def example_privacy_evaluation(client: OpenRouterClient) -> None:
    """Exemple d'évaluation de privacy"""
    print("=" * 60)
    print("EXEMPLE 1 : Évaluation de Privacy")
    print("=" * 60)
    
    # Texte original (exemple fictif)
    original_text = """
    Marie Curie est née à Varsovie en 1867. Elle a étudié la physique à Paris 
    et a découvert le radium en 1898. Elle a reçu le Prix Nobel de Physique en 1903.
    """
    
    # Texte anonymisé (version faible - encore identifiable)
    anonymized_text = """
    Une scientifique est née en Europe de l'Est au 19ème siècle. Elle a étudié 
    la physique dans une université française et a découvert un élément radioactif 
    important à la fin du 19ème siècle. Elle a reçu un prix scientifique prestigieux 
    au début du 20ème siècle.
    """
    
    # Évaluation
    result = evaluate_reidentification_risk(
        client=client,
        anonymized_text=anonymized_text,
        ground_truth_people="Marie Curie",
        p_threshold=10
    )
    
    print(f"\nRésultats :")
    print(f"  Rang de la vraie personne : {result['rank']}")
    print(f"  Identifié : {result['confirmation']}")
    print(f"  Candidats générés : {result['candidates'][:3]}...")
    print(f"  Entités sensibles : {result['sensitive_entities']}")
    print(f"  Score de risque : {result['confidence']:.2%}")


def example_utility_evaluation(client: OpenRouterClient) -> None:
    """Exemple d'évaluation d'utility"""
    print("\n" + "=" * 60)
    print("EXEMPLE 2 : Évaluation d'Utility")
    print("=" * 60)
    
    anonymized_text = """
    Une personne travaille dans le développement de logiciels. Elle écrit du code
    en Python et Java tous les jours. Elle participe à des revues de code et 
    des réunions d'équipe agiles.
    """
    
    result = evaluate_classification_utility(
        client=client,
        anonymized_text=anonymized_text,
        ground_truth_label="Software Engineer"
    )
    
    print(f"\nRésultats :")
    print(f"  Score de confiance : {result['confidence_score']}%")
    print(f"  Utilité préservée : {result['utility_preserved']}")
    print(f"  Classification : {result['confirmation']}")
    if result.get('confused_entities'):
        print(f"  Entités confuses : {result['confused_entities']}")


def example_optimization(client: OpenRouterClient) -> None:
    """Exemple de boucle d'optimisation complète"""
    print("\n" + "=" * 60)
    print("EXEMPLE 3 : Boucle d'Optimisation RUPTA")
    print("=" * 60)
    
    original_text = """
    Albert Einstein est né à Ulm en Allemagne en 1879. Il a développé la théorie
    de la relativité en 1905 alors qu'il travaillait à l'Office des brevets de Berne.
    Il a reçu le Prix Nobel de Physique en 1921.
    """
    
    # Anonymisation initiale (simple)
    initial_anonymized = """
    Une personne est née en Europe au 19ème siècle. Elle a développé une théorie
    importante en physique au début du 20ème siècle. Elle a reçu un prix prestigieux.
    """
    
    result = optimize_anonymization(
        client=client,
        original_text=original_text,
        initial_anonymized_text=initial_anonymized,
        ground_truth_people="Albert Einstein",
        ground_truth_label="Physicist",
        max_iterations=3,  # Limite pour l'exemple
        p_threshold=10
    )
    
    print(f"\nRésultats :")
    print(f"  Itérations : {result['iterations']}")
    print(f"  Convergé : {result['converged']}")
    print(f"  Privacy rank final : {result['privacy_score'].get('rank', 'N/A')}")
    print(f"  Utility score final : {result['utility_score'].get('confidence_score', 'N/A')}%")
    print(f"\nTexte final :")
    print(f"  {result['final_text'][:200]}...")
    
    print(f"\nHistorique :")
    for h in result['history']:
        print(f"  - Itération {h['iteration']}: Privacy={h['privacy_rank']}, Utility={h['utility_confidence']}%")


def main() -> None:
    """Point d'entrée principal"""
    
    client = OpenRouterClient.from_config()
    if client.requires_api_key and not client.api_key:
        missing = client.api_key_env or "OPENROUTER_API_KEY"
        print(f"⚠️  Variable d'environnement {missing} non définie !")
        print("   Définissez-la ou configurez un fournisseur local dans config.json.")
        return
    
    print("\n🚀 Exemples d'utilisation du module RUPTA\n")
    
    try:
        # Exemple 1 : Privacy
        example_privacy_evaluation(client)

        # Exemple 2 : Utility
        example_utility_evaluation(client)

        # Exemple 3 : Optimization (commenté par défaut car coûteux)
        # example_optimization(client)

        print("\n✅ Exemples terminés avec succès !")
        print("\n💡 Pour tester l'optimisation complète, décommentez example_optimization() dans main()")
        
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
