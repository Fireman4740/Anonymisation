import sys
import os
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.state import PipelineState

# Import du PseudoMapper legacy
try:
    from src.legacy_services.utils.utils_pseudo import PseudoMapper
except ImportError:
    PseudoMapper = None

class AnonymizationNode:
    def __init__(self):
        # Initialisation du mapper avec un secret par défaut
        if PseudoMapper:
            self.mapper = PseudoMapper(secret="default_secret_key", scope_id="default_scope")
        else:
            self.mapper = None

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        """
        Applique l'anonymisation (remplacement) sur le texte.
        """
        print("--- Node: Anonymization ---")
        
        if not state["config"].get("enable_anonymization", True):
            return {}

        text = state["text"]
        entities = state["entities"]
        
        if not entities:
            return {"text": text}

        # Tri des entités par position inverse pour ne pas casser les index lors du remplacement
        # (On remplace de la fin vers le début)
        sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
        
        new_text = text
        
        for ent in sorted_entities:
            start = ent["start"]
            end = ent["end"]
            # Support for both new (type/value) and legacy (entity_type/text) formats
            entity_type = ent.get("type", ent.get("entity_type"))
            entity_text = ent.get("value", ent.get("text"))
            
            if not entity_type or not entity_text:
                print(f"⚠️ Entité malformée ignorée: {ent}")
                continue
            
            # Génération du placeholder
            if self.mapper:
                placeholder = self.mapper.placeholder(entity_type, entity_text)
            else:
                # Fallback simple
                placeholder = f"[{entity_type}_MASKED]"
            
            # Remplacement dans la string
            # Attention: ceci est une manipulation de string basique. 
            # Si les entités se chevauchent, ça peut casser. 
            # Le code legacy gère ça mieux, mais ici on fait simple pour la démo.
            new_text = new_text[:start] + placeholder + new_text[end:]

        return {"text": new_text}
