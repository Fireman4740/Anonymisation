from typing import Dict, Any
from src.state import PipelineState
from .detector import DeterministicDetector

class DeterministicNode:
    def __init__(self):
        # On initialise le détecteur déterministe
        # Il chargera automatiquement patterns_config.yaml
        self.detector = DeterministicDetector(config_path="config/patterns_config.yaml")

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        """
        Exécute la détection déterministe (Regex, Algorithmes) sur le texte.
        """
        print("--- Node: Deterministic Detection ---")
        
        if not state["config"].get("enable_deterministic", True):
            print("Skipping deterministic detection (disabled in config)")
            return {"entities": []}

        text = state["text"]
        
        # Exécution de la détection
        det_entities = self.detector.detect(text)
        
        # Conversion au format du State
        entities_found = []
        for ent in det_entities:
            entities_found.append({
                "text": ent.value,
                "start": ent.start,
                "end": ent.end,
                "entity_type": ent.etype,
                "score": ent.score,
                "source": ent.source
            })

        print(f"Entités déterministes trouvées: {len(entities_found)}")
        
        # Fusion avec les entités existantes dans l'état (si d'autres nodes ont déjà tourné)
        # Ici on fait une fusion simple (append), une déduplication globale serait idéale plus tard
        existing_entities = state.get("entities", [])
        all_entities = existing_entities + entities_found
        
        return {"entities": all_entities}
