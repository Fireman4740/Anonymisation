
"""
Orchestrateur Refactorisé

Coordination légère des différentes couches avec injection de dépendances.
L'orchestrateur ne contient PAS de logique métier, seulement de la coordination.
"""

from typing import Dict, Any, Optional, List
import traceback

try:
    from ..layer1_detection import DetectionService, DetectedEntity
    from ..layer2_transformation import TransformationService
    from ..layer3_evaluation import Evaluator
    from ..api.policy import AnonymizationPolicy, preset
    from ..utils.pseudo_mapper import PseudoMapper
except Exception:
    # Pour tests directs
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from layer1_detection import DetectionService, DetectedEntity
    from layer2_transformation import TransformationService
    from layer3_evaluation import Evaluator
    from api.policy import AnonymizationPolicy, preset
    from utils.pseudo_mapper import PseudoMapper


def anonymize_text_refactored(
    value: str,
    scope_id: str,
    secret_salt: str,
    level: str = "L0",
    ner_results: Optional[List[dict]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    # Injection de dépendances (optionnel)
    detection_service: Optional[DetectionService] = None,
    transformation_service: Optional[TransformationService] = None,
    evaluator: Optional[Evaluator] = None,
) -> Dict[str, Any]:
    """
    Orchestrateur refactorisé avec injection de dépendances.
    
    Cette fonction coordonne les 3 couches sans contenir de logique métier.
    Tout est délégué aux services appropriés.
    
    Args:
        value: Texte à anonymiser
        scope_id: ID de scope pour pseudonymisation
        secret_salt: Secret HMAC
        level: Niveau de policy ("L0", "L1", "L2")
        ner_results: Résultats NER pré-calculés
        overrides: Overrides de policy
        detection_service: Service de détection (injecté ou créé)
        transformation_service: Service de transformation (injecté ou créé)
        evaluator: Évaluateur (injecté ou créé)
    
    Returns:
        Dict avec anonymized_text, audit, metrics, policy
    """
    try:
        # 1. Charger la policy et appliquer les overrides
        policy = preset(level)
        if overrides:
            for k, v in overrides.items():
                if hasattr(policy, k):
                    try:
                        setattr(policy, k, v)
                    except Exception:
                        pass
        
        # 2. Créer le mapper de pseudonymisation
        pseudo_mapper = PseudoMapper(secret=secret_salt, scope_id=scope_id)
        
        # 3. Créer les services si non fournis
        if detection_service is None:
            detection_service = _create_detection_service(policy, overrides)
        
        if transformation_service is None:
            transformation_service = _create_transformation_service(
                policy, pseudo_mapper, overrides
            )
        
        if evaluator is None:
            evaluator = Evaluator()
        
        # 4. COUCHE 1 : Détection
        skip_regex_tags = set()
        if overrides and isinstance(overrides.get("skip_regex_tags"), (list, tuple, set)):
            skip_regex_tags = {str(t).upper() for t in overrides["skip_regex_tags"]}
        
        detected_entities = detection_service.detect_all(
            text=value,
            skip_regex_tags=skip_regex_tags,
            external_ner=ner_results,
        )
        
        # 5. COUCHE 2 : Transformation
        ground_truth_people = overrides.get("rupta_ground_truth_people") if overrides else None
        ground_truth_label = overrides.get("rupta_ground_truth_label") if overrides else None
        
        transformation_result = transformation_service.transform(
            text=value,
            entities=detected_entities,
            original_text=value,
            ground_truth_people=ground_truth_people,
            ground_truth_label=ground_truth_label,
        )
        
        # 6. COUCHE 3 : Évaluation
        evaluation_result = evaluator.evaluate(
            original_text=value,
            anonymized_text=transformation_result.anonymized_text,
            entities_detected=len(detected_entities),
            entities_replaced=len(transformation_result.replacements),
        )
        
        # 7. Assembler le résultat
        result = {
            "anonymized_text": transformation_result.anonymized_text,
            "audit": {
                "entities": [e.to_dict() for e in detected_entities],
                "replacements": transformation_result.replacements,
                "generalizations": [
                    {
                        "start": g.start,
                        "end": g.end,
                        "surface": g.surface,
                        "replacement": g.replacement,
                        "etype": g.etype,
                        "policy_rule": g.policy_rule,
                    }
                    for g in transformation_result.generalizations
                ],
                "paraphrase_applied": transformation_result.paraphrase_applied,
                "rupta_applied": transformation_result.rupta_applied,
                **transformation_result.metadata,
            },
            "evaluation": {
                "is_valid": evaluation_result.is_valid,
                "metrics": evaluation_result.metrics,
                "validation_errors": evaluation_result.validation_errors,
                "warnings": evaluation_result.warnings,
            },
            "policy": policy.to_dict(),
        }
        
        # Supprimer les mappings si policy le demande
        if policy.mapping_retention == "discard" and "mappings" in result["audit"]:
            del result["audit"]["mappings"]
        
        return result
    
    except Exception as e:
        # Gestion d'erreur globale
        return {
            "anonymized_text": value,  # Retourner le texte original en cas d'erreur
            "audit": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "entities": [],
            },
            "evaluation": {
                "is_valid": False,
                "metrics": {},
                "validation_errors": [str(e)],
                "warnings": [],
            },
            "policy": preset(level).to_dict() if level else {},
        }


def _create_detection_service(
    policy: AnonymizationPolicy,
    overrides: Optional[Dict[str, Any]] = None,
) -> DetectionService:
    """
    Crée le service de détection selon la policy.
    
    Args:
        policy: Policy d'anonymisation
        overrides: Overrides optionnels
    
    Returns:
        DetectionService configuré
    """
    # Charger GPU pipeline si disponible
    gpu_pipeline = None
    try:
        from ..layer1_detection.ner.gpu_optimizer import create_optimized_pipeline, load_gpu_config
        config = load_gpu_config()
        if config.get("enabled"):
            gpu_pipeline = create_optimized_pipeline(config)
    except Exception:
        pass
    
    # Créer le service
    return DetectionService(
        gpu_pipeline=gpu_pipeline,
        use_gliner=True,
        gliner_preset=overrides.get("gliner_preset", "balanced") if overrides else "balanced",
        gliner_threshold=overrides.get("gliner_threshold", 0.35) if overrides else 0.35,
    )


def _create_transformation_service(
    policy: AnonymizationPolicy,
    pseudo_mapper: PseudoMapper,
    overrides: Optional[Dict[str, Any]] = None,
) -> TransformationService:
    """
    Crée le service de transformation selon la policy.
    
    Args:
        policy: Policy d'anonymisation
        pseudo_mapper: Mapper de pseudonymisation
        overrides: Overrides optionnels
    
    Returns:
        TransformationService configuré
    """
    # Créer le service LLM si nécessaire (L1+)
    llm_service = None
    if policy.llm_detection or policy.llm_paraphrase or policy.llm_audit or policy.rupta_enabled:
        try:
            # Importer et créer le service LLM
            # NOTE: Le code LLM complet (reasoner, openrouter_client, rupta) doit être ajouté
            pass
        except Exception:
            pass
    
    return TransformationService(
        policy=policy,
        pseudo_mapper=pseudo_mapper,
        llm_service=llm_service,
    )


__all__ = ["anonymize_text_refactored"]
