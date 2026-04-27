from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import operator


class EntityDict(TypedDict, total=False):
    """
    Format canonique d'une entité détectée par n'importe quel nœud du pipeline.

    Champs obligatoires (always present):
        start   : offset de début dans original_text (int)
        end     : offset de fin   dans original_text (int)
        type    : label normalisé (ex: "PERSON", "EMAIL", "PHONE") (str)
        value   : texte brut de l'entité (str)
        source  : origine ("regex", "gliner", "flair", "spacy", "llm") (str)
        score   : confiance / vote agrégé (float, 0.0–∞)

    Champs optionnels (présents selon le sous-détecteur):
        llm_reason       : explication du LLM
        validated        : bool — l'entité a passé un validateur algorithmique
        original_label   : label brut avant normalisation (utile debug)
    """

    start: int
    end: int
    type: str
    value: str
    source: str
    score: float
    llm_reason: str
    validated: bool
    original_label: str


# Définition de l'état du graphe
# TypedDict est recommandé pour LangGraph pour le typage de l'état
class PipelineState(TypedDict):
    # Données principales
    text: str                        # Le texte qui évolue (peut être anonymisé progressivement)
    original_text: str               # Copie immuable pour référence

    # Résultats intermédiaires
    entities: List[EntityDict]       # Liste des entités détectées (format EntityDict)

    # Configuration (Flags d'activation + paramètres d'ablation)
    config: Dict[str, Any]           # Ex: {"enable_ner": True, "gliner_preset": "best", ...}

    # Métadonnées et Logs
    metadata: Dict[str, Any]        # Stats, temps d'exécution, etc.
    errors: List[str]               # Liste d'erreurs éventuelles

    # --- Champs LLM / RUPTA ---
    privacy_score: int              # Score de re-identification 0-100 (0=anonyme, 100=identifiable)
    llm_feedback: Dict[str, Any]    # Retour de l'audit LLM (leaked_attributes, assessment, etc.)
    iteration: int                  # Compteur de boucle adversariale (max = rupta.max_iterations)


def create_initial_state(text: str, config: Optional[Dict[str, bool]] = None) -> PipelineState:
    return {
        "text": text,
        "original_text": text,
        "entities": [],
        "config": config or {},
        "metadata": {},
        "errors": [],
        # LLM defaults
        "privacy_score": 100,
        "llm_feedback": {},
        "iteration": 0,
    }
