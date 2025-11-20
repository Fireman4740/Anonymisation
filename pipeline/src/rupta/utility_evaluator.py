"""
Utility Evaluator pour RUPTA

Évalue la préservation de l'utilité du texte anonymisé pour des tâches en aval.
Adapté de : generators/generator_utils.py::generic_utility_reflection
"""

from typing import Dict, Any, Optional
import logging

try:
    from ..llm.openrouter_client import OpenRouterClient, load_llm_settings
    from .prompts_multilang import (
        UTILITY_REFLECTION_FR_1,
        UTILITY_CONFUSED_ENTITIES_FR,
        GENERAL_SYSTEM_FR
    )
except ImportError:
    from src.llm.openrouter_client import OpenRouterClient, load_llm_settings
    from src.rupta.prompts_multilang import (
        UTILITY_REFLECTION_FR_1,
        UTILITY_CONFUSED_ENTITIES_FR,
        GENERAL_SYSTEM_FR
    )

logger = logging.getLogger(__name__)


def evaluate_classification_utility(
    client: OpenRouterClient,
    anonymized_text: str,
    ground_truth_label: str,
    original_text: str = "",
    model: Optional[str] = None,
    confidence_threshold: int = 80
) -> Dict[str, Any]:
    """
    Évalue si le texte anonymisé préserve suffisamment d'information 
    pour la tâche de classification.
    
    Processus :
    1. Demande au LLM d'évaluer la confiance de classification (0-100)
    2. Si confiance >= threshold : utilité préservée
    3. Si confiance < threshold : identifie les entités confuses
    
    Args:
        client: Client OpenRouter
        anonymized_text: Le texte après anonymisation
        ground_truth_label: La vraie catégorie (ex: occupation)
        original_text: Texte original (optionnel, pour comparaison)
        model: Modèle LLM à utiliser
        confidence_threshold: Seuil de confiance minimal (défaut: 80)
    
    Returns:
        {
            'confidence_score': int,  # 0-100
            'confirmation': str,  # 'Yes' si score >= threshold, 'No' sinon
            'confused_entities': List[str],  # Entités qui nuisent à la classification
            'utility_preserved': bool  # True si utilité acceptable
        }
    """
    
    logger.info(f"Évaluation de l'utilité pour la classification : {ground_truth_label}")

    if not model:
        try:
            settings = load_llm_settings()
            models = settings.get("models", {}) if isinstance(settings.get("models"), dict) else {}
            model = models.get("paraphrase") or models.get("detect") or settings.get("fallback_model")
        except Exception:
            model = None
        if not model:
            model = "openai/gpt-4.1-mini"
    
    prompt = UTILITY_REFLECTION_FR_1.format(
        anonymized_text=anonymized_text,
        label=ground_truth_label,
        format_instructions=_get_format_instructions_confidence()
    )
    
    try:
        response_data = client.call_json(
            system_prompt=GENERAL_SYSTEM_FR,
            user_prompt=prompt,
            model=model,
            temperature=0.0
        )
        
        confidence_score = int(response_data.get("confidence_score", 0))
        
        # Vérifier si l'utilité est préservée
        utility_preserved = confidence_score >= confidence_threshold
        confirmation = "Yes" if utility_preserved else "No"
        
        confused_entities = []
        
        # Si utilité insuffisante, identifier les entités problématiques
        if not utility_preserved and original_text:
            logger.info(f"Utilité insuffisante ({confidence_score}%), identification des entités confuses...")
            
            prompt_entities = f"""
Texte anonymisé : {anonymized_text}
Texte original : {original_text}
Label attendu : {ground_truth_label}

{UTILITY_CONFUSED_ENTITIES_FR.format(
    format_instructions=_get_format_instructions_entities()
)}
"""
            
            try:
                entities_data = client.call_json(
                    system_prompt=GENERAL_SYSTEM_FR,
                    user_prompt=prompt_entities,
                    model=model,
                    temperature=0.0
                )
                
                entities_str = entities_data.get("confused_entities", "")
                confused_entities = [e.strip() for e in entities_str.split(",") if e.strip()]
                
            except Exception as e:
                logger.warning(f"Erreur lors de l'identification des entités confuses : {e}")
        
        return {
            'confidence_score': confidence_score,
            'confirmation': confirmation,
            'confused_entities': confused_entities,
            'utility_preserved': utility_preserved
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation d'utilité : {e}")
        return {
            'confidence_score': 0,
            'confirmation': 'Error',
            'confused_entities': [],
            'utility_preserved': False,
            'error': str(e)
        }


def calculate_utility_metrics(
    results: list,
    ground_truth_key: str = 'label'
) -> Dict[str, Any]:
    """
    Calcule des métriques d'utilité sur un ensemble de résultats.
    
    Args:
        results: Liste de résultats d'évaluation
        ground_truth_key: Clé pour la vérité terrain
    
    Returns:
        {
            'avg_confidence': float,
            'utility_preserved_rate': float,
            'low_confidence_count': int
        }
    """
    
    if not results:
        return {
            'avg_confidence': 0.0,
            'utility_preserved_rate': 0.0,
            'low_confidence_count': 0
        }
    
    total_confidence = 0
    preserved_count = 0
    low_conf_count = 0
    
    for result in results:
        conf = result.get('confidence_score', 0)
        total_confidence += conf
        
        if result.get('utility_preserved', False):
            preserved_count += 1
        if conf < 50:
            low_conf_count += 1
    
    return {
        'avg_confidence': total_confidence / len(results),
        'utility_preserved_rate': preserved_count / len(results),
        'low_confidence_count': low_conf_count
    }


# ==================== Helpers ====================

def _get_format_instructions_confidence() -> str:
    """Format JSON pour le score de confiance"""
    return """
Répondez au format JSON suivant :
{
    "confidence_score": <nombre entre 0 et 100>
}
"""


def _get_format_instructions_entities() -> str:
    """Format JSON pour les entités confuses"""
    return """
Répondez au format JSON suivant :
{
    "confused_entities": "entité1, entité2, entité3, ..."
}
"""


# Alias pour compatibilité avec les scripts d'évaluation
def evaluate_utility_preservation(
    client: OpenRouterClient,
    anonymized_text: str,
    ground_truth_label: str,
    language: str = "auto",
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Alias vers evaluate_classification_utility pour compatibilité.
    
    Évalue la préservation de l'utilité dans le texte anonymisé.
    """
    return evaluate_classification_utility(
        client=client,
        anonymized_text=anonymized_text,
        ground_truth_label=ground_truth_label,
        model=model
    )
