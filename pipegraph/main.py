import sys
import os

# Ajout du dossier courant au path pour que les imports fonctionnent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph import create_pipeline_graph
from src.state import create_initial_state

def main():
    print("🚀 Démarrage du PipeGraph d'Anonymisation...")
    
    # 1. Création du graphe
    try:
        pipeline = create_pipeline_graph()
    except ImportError as e:
        print(f"❌ Erreur critique d'import (manque langgraph ?): {e}")
        print("Veuillez installer les dépendances: pip install -r requirements.txt")
        return

    # 2. Données de test
    sample_text = "Je m'appelle Jean Dupont et mon email est jean.dupont@example.com. J'habite à Paris."
    print(f"\n📄 Texte original:\n{sample_text}\n")

    # 3. Configuration (Activation modulaire)
    config = {
        "enable_deterministic": True,
        "enable_detection": True, # AI Detection
        "enable_anonymization": True,
        # "enable_llm": False # Pas encore implémenté
    }

    # 4. État initial
    initial_state = create_initial_state(sample_text, config)

    # 5. Exécution
    print("⚙️ Exécution du pipeline...")
    try:
        # invoke retourne l'état final
        final_state = pipeline.invoke(initial_state)
        
        print("\n✅ Pipeline terminé avec succès!")
        print("-" * 30)
        print(f"📝 Texte Anonymisé:\n{final_state['text']}")
        print("-" * 30)
        print(f"🔍 Entités détectées ({len(final_state['entities'])}):")
        for ent in final_state['entities']:
            etype = ent.get("type", ent.get("entity_type"))
            val = ent.get("value", ent.get("text"))
            print(f" - {etype}: {val} -> {ent['start']}:{ent['end']}")
            
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
