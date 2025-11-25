from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import operator

# Définition de l'état du graphe
# TypedDict est recommandé pour LangGraph pour le typage de l'état
class PipelineState(TypedDict):
    # Données principales
    text: str                       # Le texte qui évolue (peut être anonymisé progressivement)
    original_text: str              # Copie immuable pour référence
    
    # Résultats intermédiaires
    entities: List[Dict[str, Any]]  # Liste des entités détectées (format dict pour sérialisation facile)
    
    # Configuration (Flags d'activation)
    config: Dict[str, bool]         # Ex: {"enable_ner": True, "enable_llm": False}
    
    # Métadonnées et Logs
    metadata: Dict[str, Any]        # Stats, temps d'exécution, etc.
    errors: List[str]               # Liste d'erreurs éventuelles

def create_initial_state(text: str, config: Optional[Dict[str, bool]] = None) -> PipelineState:
    return {
        "text": text,
        "original_text": text,
        "entities": [],
        "config": config or {},
        "metadata": {},
        "errors": []
    }
