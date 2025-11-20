"""
Optimizer RUPTA - Boucle de raffinement itératif

Coordonne les évaluations de privacy et utility pour améliorer progressivement
l'anonymisation.
"""

from typing import Dict, Any, Optional
import logging

try:
    from ..openrouter_client import OpenRouterClient, load_llm_settings
    from .privacy_evaluator import evaluate_reidentification_risk
    from .utility_evaluator import evaluate_classification_utility
    from .prompts_multilang import REINFORCEMENT_FR, GENERAL_SYSTEM_FR
except ImportError:
    from src.openrouter_client import OpenRouterClient, load_llm_settings
    from src.rupta.privacy_evaluator import evaluate_reidentification_risk
    from src.rupta.utility_evaluator import evaluate_classification_utility  
    from src.rupta.prompts_multilang import REINFORCEMENT_FR, GENERAL_SYSTEM_FR

logger = logging.getLogger(__name__)


def optimize_anonymization(
    client: OpenRouterClient,
    original_text: str,
    initial_anonymized_text: str,
    ground_truth_people: str,
    ground_truth_label: str,
    max_iterations: int = 5,
    p_threshold: int = 10,
    privacy_target_rank: int = 11,
    utility_min_confidence: int = 80,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Boucle d'optimisation itérative pour améliorer l'anonymisation.
    
    Processus :
    1. Évaluer privacy et utility du texte initial
    2. Si privacy insuffisant OU utility insuffisant :
       - Générer des suggestions d'amélioration
       - Demander au LLM de modifier le texte
       - Réévaluer
    3. Répéter jusqu'à convergence ou max_iterations
    
    Args:
        client: Client OpenRouter
        original_text: Texte original avant anonymisation
        initial_anonymized_text: Texte après anonymisation initiale
        ground_truth_people: Nom réel de la personne
        ground_truth_label: Label de classification (ex: occupation)
        max_iterations: Nombre max d'itérations
        p_threshold: Seuil pour privacy (rang acceptable)
        privacy_target_rank: Rang cible (défaut: p_threshold + 1)
        utility_min_confidence: Confiance minimale pour utility
        model: Modèle LLM
    
    Returns:
        {
            'final_text': str,
            'iterations': int,
            'privacy_score': dict,
            'utility_score': dict,
            'converged': bool,
            'history': List[dict]
        }
    """
    
    logger.info("Démarrage de l'optimisation RUPTA")

    resolved_model = model
    if not resolved_model:
        try:
            settings = load_llm_settings()
            models = settings.get("models", {}) if isinstance(settings.get("models"), dict) else {}
            resolved_model = models.get("paraphrase") or models.get("detect") or settings.get("fallback_model")
        except Exception:
            resolved_model = None
        if not resolved_model:
            resolved_model = "openai/gpt-4.1-mini"
    
    current_text = initial_anonymized_text
    history = []
    
    for iteration in range(max_iterations):
        logger.info(f"=== Itération {iteration + 1}/{max_iterations} ===")
        
        # Évaluation Privacy
        privacy_eval = evaluate_reidentification_risk(
            client=client,
            anonymized_text=current_text,
            ground_truth_people=ground_truth_people,
            p_threshold=p_threshold,
            model=resolved_model
        )
        
        # Évaluation Utility
        utility_eval = evaluate_classification_utility(
            client=client,
            anonymized_text=current_text,
            ground_truth_label=ground_truth_label,
            original_text=original_text,
            model=resolved_model,
            confidence_threshold=utility_min_confidence
        )
        
        # Enregistrer dans l'historique
        history.append({
            'iteration': iteration + 1,
            'text': current_text,
            'privacy_rank': privacy_eval.get('rank', 999),
            'privacy_confirmation': privacy_eval.get('confirmation', 'No'),
            'utility_confidence': utility_eval.get('confidence_score', 0),
            'utility_preserved': utility_eval.get('utility_preserved', False)
        })
        
        privacy_rank = privacy_eval.get('rank', 999)
        utility_preserved = utility_eval.get('utility_preserved', False)
        
        logger.info(f"Privacy rank: {privacy_rank}, Utility confidence: {utility_eval.get('confidence_score', 0)}%")
        
        # Critère de convergence
        if privacy_rank > privacy_target_rank and utility_preserved:
            logger.info("✓ Convergence atteinte !")
            return {
                'final_text': current_text,
                'iterations': iteration + 1,
                'privacy_score': privacy_eval,
                'utility_score': utility_eval,
                'converged': True,
                'history': history
            }
        
        # Déterminer l'action à prendre
        if privacy_rank <= p_threshold:
            # Privacy insuffisant → Généraliser
            action = "generalize"
            suggestion = _build_privacy_suggestion(privacy_eval)
        else:
            # Privacy OK mais utility insuffisant → Spécifier
            action = "specify"
            suggestion = _build_utility_suggestion(utility_eval)
        
        logger.info(f"Action: {action}")
        logger.info(f"Suggestion: {suggestion[:200]}...")
        
        # Demander au LLM de modifier le texte
        try:
            editing_history_str = _format_editing_history(history, p_threshold)
            
            prompt = REINFORCEMENT_FR.format(
                p_threshold=p_threshold,
                original_text=original_text,
                editing_history=editing_history_str,
                format_instructions=_get_format_instructions()
            )
            
            prompt_with_suggestion = f"{prompt}\n\nSuggestion d'amélioration :\n{suggestion}"
            
            response_data = client.call_json(
                system_prompt=GENERAL_SYSTEM_FR,
                user_prompt=prompt_with_suggestion,
                model=resolved_model,
                temperature=0.0
            )
            
            new_text = response_data.get("anonymized_text", current_text)
            
            if new_text == current_text:
                logger.warning("Le LLM n'a pas modifié le texte, arrêt des itérations")
                break
            
            current_text = new_text
            
        except Exception as e:
            logger.error(f"Erreur lors de la modification du texte : {e}")
            break
    
    # Fin des itérations sans convergence
    logger.warning("Max iterations atteint sans convergence complète")
    
    return {
        'final_text': current_text,
        'iterations': max_iterations,
        'privacy_score': privacy_eval,
        'utility_score': utility_eval,
        'converged': False,
        'history': history
    }


# ==================== Helpers ====================

def _build_privacy_suggestion(privacy_eval: dict) -> str:
    """Construit une suggestion pour améliorer la privacy"""
    entities = privacy_eval.get('sensitive_entities', [])
    rank = privacy_eval.get('rank', 999)
    
    if entities:
        entities_str = ", ".join(entities[:5])  # Max 5 entités
        return f"Privacy insuffisant (rang {rank}). Généraliser ces entités : {entities_str}"
    else:
        return f"Privacy insuffisant (rang {rank}). Généraliser davantage le texte."


def _build_utility_suggestion(utility_eval: dict) -> str:
    """Construit une suggestion pour améliorer l'utility"""
    conf = utility_eval.get('confidence_score', 0)
    entities = utility_eval.get('confused_entities', [])
    
    if entities:
        entities_str = ", ".join(entities[:5])
        return f"Utility insuffisant ({conf}%). Spécifier ces entités : {entities_str}"
    else:
        return f"Utility insuffisant ({conf}%). Ajouter plus de détails pertinents pour la classification."


def _format_editing_history(history: list, p_threshold: int) -> str:
    """Formate l'historique d'édition pour le prompt"""
    if not history:
        return "Aucun historique"
    
    lines = []
    for h in history:
        privacy_score = p_threshold + 1 if h['privacy_rank'] > p_threshold else h['privacy_rank']
        utility_score = h['utility_confidence']
        
        # Calcul de la récompense selon les règles RUPTA
        if privacy_score <= p_threshold:
            reward = privacy_score
        else:
            reward = utility_score
        
        lines.append(f"""
Édition {h['iteration']}:
Texte: {h['text'][:200]}...
Privacy score: {privacy_score}
Utility score: {utility_score}
Récompense: {reward}
""")
    
    return "\n".join(lines)


def _get_format_instructions() -> str:
    """Format JSON pour la réponse d'optimisation"""
    return """
Répondez au format JSON suivant :
{
    "anonymized_text": "le texte anonymisé amélioré",
    "explanation": "brève explication des changements effectués"
}
"""
