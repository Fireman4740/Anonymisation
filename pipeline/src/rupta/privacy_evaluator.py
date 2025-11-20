"""
Privacy Evaluator pour RUPTA

Évalue le risque de ré-identification d'un texte anonymisé.
Adapté de : generators/generator_utils.py::generic_privacy_reflection
"""

from typing import Dict, Any, List, Optional
import logging

try:
    from ..llm.openrouter_client import OpenRouterClient, load_llm_settings
    from .prompts_multilang import (
        PRIVACY_REFLECTION_FR_1,
        PRIVACY_REFLECTION_FR_2,
        PRIVACY_CONFIDENCE_FR,
        GENERAL_SYSTEM_FR
    )
except ImportError:
    from src.llm.openrouter_client import OpenRouterClient, load_llm_settings
    from src.rupta.prompts_multilang import (
        PRIVACY_REFLECTION_FR_1,
        PRIVACY_REFLECTION_FR_2,
        PRIVACY_CONFIDENCE_FR,
        GENERAL_SYSTEM_FR
    )

logger = logging.getLogger(__name__)


def evaluate_reidentification_risk(
    client: OpenRouterClient,
    anonymized_text: str,
    ground_truth_people: str,
    p_threshold: int = 10,
    model: Optional[str] = None,
    language: str = "auto"
) -> Dict[str, Any]:
    """
    Évalue le risque qu'une personne puisse être ré-identifiée à partir du texte anonymisé.
    
    Processus :
    1. Génère une liste de p_threshold candidats susceptibles d'être la personne décrite
    2. Vérifie si la vraie personne (ground_truth_people) est dans cette liste
    3. Si oui, identifie les entités sensibles qui ont permis l'identification
    
    Args:
        client: Client OpenRouter pour appels LLM
        anonymized_text: Le texte après anonymisation
        ground_truth_people: Le nom réel de la personne décrite
        p_threshold: Nombre de candidats à générer (défaut: 10)
        model: Modèle LLM à utiliser
        language: Langue des prompts ('auto', 'fr', 'en', 'de', 'es', 'it', etc.)
                  'auto' utilise les prompts multilingues en anglais
    
    Returns:
        {
            'rank': int,  # Position de la vraie personne dans les candidats (999 si non trouvée)
            'confirmation': str,  # 'Yes' si identifié, 'No' sinon
            'sensitive_entities': List[str],  # Entités qui ont permis l'identification
            'candidates': List[str],  # Liste des candidats générés
            'confidence': float  # Score de confiance (si calculé)
        }
    """
    
    logger.info(f"Évaluation du risque de ré-identification pour : {ground_truth_people[:50]}...")

    if not model:
        try:
            settings = load_llm_settings()
            models = settings.get("models", {}) if isinstance(settings.get("models"), dict) else {}
            model = models.get("audit") or models.get("detect") or settings.get("fallback_model")
        except Exception:
            model = None
        if not model:
            model = "openai/gpt-4.1-mini"
    
    # Étape 1 : Générer les candidats
    prompt_1 = PRIVACY_REFLECTION_FR_1.format(
        p_threshold=p_threshold,
        anonymized_text=anonymized_text,
        format_instructions=_get_format_instructions_candidates()
    )
    
    try:
        response_1_text = client.call_json(
            system_prompt=GENERAL_SYSTEM_FR,
            user_prompt=prompt_1,
            model=model,
            temperature=0.0
        )
        
        # Parser la réponse JSON (déjà parsée par call_json)
        candidates_data = response_1_text
        candidates = candidates_data.get("candidates", [])
        
        if not candidates:
            logger.warning("Aucun candidat généré")
            return {
                'rank': 999,
                'confirmation': 'No',
                'sensitive_entities': [],
                'candidates': [],
                'confidence': 0.0
            }
        
        logger.info(f"Candidats générés : {candidates[:3]}...")
        
        # Vérifier si la vraie personne est dans les candidats
        rank = _find_person_rank(ground_truth_people, candidates)
        
        if rank <= p_threshold:
            # Étape 2 : Identifier les entités sensibles
            logger.info(f"Personne trouvée au rang {rank}, identification des entités sensibles...")
            
            # Construire un nouveau prompt avec le contexte
            import json
            candidates_str = json.dumps(candidates, ensure_ascii=False)
            
            prompt_2_full = f"""
Contexte: J'ai demandé à identifier des candidats susceptibles d'être décrits par un texte anonymisé.

Voici les candidats générés : {candidates_str}

{PRIVACY_REFLECTION_FR_2.format(
                people=ground_truth_people,
                format_instructions=_get_format_instructions_entities()
            )}
"""
            
            response_2_data = client.call_json(
                system_prompt=GENERAL_SYSTEM_FR,
                user_prompt=prompt_2_full,
                model=model,
                temperature=0.0
            )
            
            confirmation = response_2_data.get("confirmation", "No")
            sensitive_entities_str = response_2_data.get("sensitive_entities", "")
            sensitive_entities = [e.strip() for e in sensitive_entities_str.split(",") if e.strip()]
            
            return {
                'rank': rank,
                'confirmation': confirmation,
                'sensitive_entities': sensitive_entities,
                'candidates': candidates,
                'confidence': 1.0 - (rank / p_threshold)
            }
        else:
            logger.info(f"Personne non trouvée dans les {p_threshold} premiers candidats")
            return {
                'rank': rank,
                'confirmation': 'No',
                'sensitive_entities': [],
                'candidates': candidates,
                'confidence': 0.0
            }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation de ré-identification : {e}")
        return {
            'rank': 999,
            'confirmation': 'Error',
            'sensitive_entities': [],
            'candidates': [],
            'confidence': 0.0,
            'error': str(e)
        }


def evaluate_confidence_score(
    client: OpenRouterClient,
    anonymized_text: str,
    candidate_person: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Évalue le score de confiance pour associer le texte anonymisé à une personne candidate.
    
    Args:
        client: Client OpenRouter
        anonymized_text: Texte anonymisé
        candidate_person: Nom de la personne candidate
        model: Modèle LLM
    
    Returns:
        {
            'confidence_score': int,  # 0-100
            'reason': str  # Explication du score
        }
    """
    
    if not model:
        try:
            settings = load_llm_settings()
            models = settings.get("models", {}) if isinstance(settings.get("models"), dict) else {}
            model = models.get("audit") or models.get("detect") or settings.get("fallback_model")
        except Exception:
            model = None
        if not model:
            model = "openai/gpt-4.1-mini"

    prompt = PRIVACY_CONFIDENCE_FR.format(
        anonymized_text=anonymized_text,
        people=candidate_person,
        format_instructions=_get_format_instructions_confidence()
    )
    
    try:
        response_data = client.call_json(
            system_prompt=GENERAL_SYSTEM_FR,
            user_prompt=prompt,
            model=model,
            temperature=0.0
        )
        
        return {
            'confidence_score': int(response_data.get("confidence_score", 0)),
            'reason': response_data.get("reason", "")
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul du score de confiance : {e}")
        return {
            'confidence_score': 0,
            'reason': f"Erreur : {str(e)}"
        }


# ==================== Helpers ====================

def _find_person_rank(ground_truth: str, candidates: List[str]) -> int:
    """
    Trouve le rang de la vraie personne dans la liste des candidats.
    Utilise une correspondance approximative (similarity).
    """
    ground_truth_lower = ground_truth.lower().strip()
    
    for i, candidate in enumerate(candidates):
        candidate_lower = candidate.lower().strip()
        
        # Correspondance exacte
        if ground_truth_lower == candidate_lower:
            return i + 1
        
        # Correspondance partielle (nom contenu dans candidat ou inverse)
        if ground_truth_lower in candidate_lower or candidate_lower in ground_truth_lower:
            return i + 1
    
    return 999  # Non trouvé


def _get_format_instructions_candidates() -> str:
    """Format JSON pour la liste de candidats"""
    return """
Répondez au format JSON suivant :
{
    "candidates": ["Nom Candidat 1", "Nom Candidat 2", ..., "Nom Candidat N"]
}

Où N = nombre de candidats demandé.
"""


def _get_format_instructions_entities() -> str:
    """Format JSON pour les entités sensibles"""
    return """
Répondez au format JSON suivant :
{
    "confirmation": "Yes ou No",
    "sensitive_entities": "entité1, entité2, entité3, ..."
}

Si confirmation = No, sensitive_entities peut être vide.
"""


def _get_format_instructions_confidence() -> str:
    """Format JSON pour le score de confiance"""
    return """
Répondez au format JSON suivant :
{
    "confidence_score": <nombre entre 0 et 100>,
    "reason": "Explication de votre évaluation"
}
"""
