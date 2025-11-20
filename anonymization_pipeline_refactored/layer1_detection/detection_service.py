"""
Service de Détection Unifié

Ce service encapsule toute la logique de détection d'entités :
- Détection regex (patterns PII)
- Détection NER (GLiNER avec support GPU)
- Déduplication et fusion des résultats
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import re

try:
    from .regex.text_sanitizer import regexes_based_replacements
    from .ner.gliner_ensemble import run_gliner, merge_ner_lists, GLINER_ALL_LABELS
except Exception:
    from regex.text_sanitizer import regexes_based_replacements
    from ner.gliner_ensemble import run_gliner, merge_ner_lists, GLINER_ALL_LABELS


@dataclass
class DetectedEntity:
    """
    Représente une entité détectée avec métadonnées complètes.
    
    Attributes:
        start: Position de début dans le texte
        end: Position de fin dans le texte
        surface: Texte de l'entité
        etype: Type d'entité (PER, ORG, EMAIL, etc.)
        source: Source de détection ("regex", "ner", "ner-gpu", "llm")
        score: Score de confiance (0.0-1.0)
        metadata: Métadonnées additionnelles
    """
    start: int
    end: int
    surface: str
    etype: str  
    source: str 
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Assure que metadata est toujours un dict."""
        if self.metadata is None:
            self.metadata = {}
    
    def overlaps_with(self, other: "DetectedEntity") -> bool:
        """Vérifie si cette entité chevauche une autre."""
        return not (self.end <= other.start or self.start >= other.end)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "start": self.start,
            "end": self.end,
            "surface": self.surface,
            "etype": self.etype,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


class DetectionService:
    """
    Service de détection unifié combinant regex et NER.
    
    Ce service encapsule toute la logique de détection et peut optionnellement
    utiliser un pipeline NER GPU-optimisé pour de meilleures performances.
    """
    
    def __init__(
        self,
        gpu_pipeline=None,  # Pipeline GPU optionnel de ner_gpu_optimizer
        use_gliner: bool = True,
        gliner_models: Optional[List[str]] = None,
        gliner_labels: Optional[List[str]] = None,
        gliner_threshold: float = 0.35,
        gliner_preset: str = "balanced",
    ):
        """
        Initialise le service de détection.
        
        Args:
            gpu_pipeline: Pipeline NER GPU-optimisé optionnel
            use_gliner: Activer la détection GLiNER
            gliner_models: Liste des modèles GLiNER à utiliser
            gliner_labels: Liste des labels à détecter
            gliner_threshold: Seuil de confiance GLiNER
            gliner_preset: Preset GLiNER ("fast", "balanced", "accuracy", "pii", "best")
        """
        self.gpu_pipeline = gpu_pipeline
        self.use_gliner = use_gliner
        self.gliner_models = gliner_models
        self.gliner_labels = gliner_labels or GLINER_ALL_LABELS
        self.gliner_threshold = gliner_threshold
        self.gliner_preset = gliner_preset
    
    def detect_regex(
        self,
        text: str,
        skip_tags: Optional[Set[str]] = None,
    ) -> List[DetectedEntity]:
        """
        Détecte les entités en utilisant les patterns regex.
        
        Args:
            text: Texte à analyser
            skip_tags: Ensemble de tags à ignorer (ex: {"EMAIL", "PHONE"})
        
        Returns:
            Liste d'entités détectées
        """
        skip_tags = skip_tags or set()
        regex_results = regexes_based_replacements(text)
        
        entities = []
        for start, end, tag in regex_results:
            # Normaliser le tag (retirer < >)
            normalized_tag = tag.strip("<>")
            
            if normalized_tag.upper() in skip_tags:
                continue
            
            entities.append(DetectedEntity(
                start=start,
                end=end,
                surface=text[start:end],
                etype=normalized_tag,
                source="regex",
                score=1.0,
                metadata={"pattern": tag}
            ))
        
        return entities
    
    def detect_ner(
        self,
        text: str,
        external_ner: Optional[List[dict]] = None,
    ) -> List[DetectedEntity]:
        """
        Détecte les entités en utilisant NER (GLiNER ou GPU pipeline).
        
        Args:
            text: Texte à analyser
            external_ner: Résultats NER externes à fusionner
        
        Returns:
            Liste d'entités détectées
        """
        ner_results = []
        
        # Utiliser GPU pipeline si disponible
        if self.gpu_pipeline:
            try:
                gpu_ner = self.gpu_pipeline.predict(text, threshold=self.gliner_threshold)
                ner_results = merge_ner_lists(gpu_ner, external_ner or [])
                source = "ner-gpu"
            except Exception as e:
                print(f"[DetectionService] GPU pipeline error: {e}, falling back to GLiNER")
                ner_results = []
        
        # Fallback ou mode standard : utiliser GLiNER
        if not ner_results and self.use_gliner:
            try:
                gliner_ner = run_gliner(
                    text,
                    model_names=self.gliner_models,
                    labels=self.gliner_labels,
                    threshold=self.gliner_threshold,
                    preset=self.gliner_preset,
                )
                ner_results = merge_ner_lists(gliner_ner, external_ner or [])
                source = "ner"
            except Exception as e:
                print(f"[DetectionService] GLiNER error: {e}")
                ner_results = external_ner or []
                source = "ner-external"
        elif external_ner:
            ner_results = external_ner
            source = "ner-external"
        
        # Convertir en DetectedEntity
        entities = []
        for ent in ner_results:
            start = ent.get("start")
            end = ent.get("end")
            etype = ent.get("entity_group", "UNKNOWN")
            
            if start is None or end is None or end <= start:
                continue
            
            entities.append(DetectedEntity(
                start=start,
                end=end,
                surface=text[start:end],
                etype=etype.upper(),
                source=source if 'source' not in locals() else "ner",
                score=ent.get("score", ent.get("votes", 1.0)),
                metadata={k: v for k, v in ent.items() if k not in {"start", "end", "entity_group"}}
            ))
        
        return entities
    
    def detect_all(
        self,
        text: str,
        skip_regex_tags: Optional[Set[str]] = None,
        external_ner: Optional[List[dict]] = None,
    ) -> List[DetectedEntity]:
        """
        Détecte toutes les entités (regex + NER) avec déduplication.
        
        Args:
            text: Texte à analyser
            skip_regex_tags: Tags regex à ignorer
            external_ner: Résultats NER externes
        
        Returns:
            Liste fusionnée et dédupliquée d'entités
        """
        # Détecter avec regex
        regex_entities = self.detect_regex(text, skip_regex_tags)
        
        # Détecter avec NER
        ner_entities = self.detect_ner(text, external_ner)
        
        # Fusionner et dédupliquer (priorité: regex > ner-gpu > ner)
        all_entities = self._deduplicate_entities(regex_entities + ner_entities)
        
        # Trier par position
        all_entities.sort(key=lambda e: (e.start, e.end))
        
        return all_entities
    
    def _deduplicate_entities(self, entities: List[DetectedEntity]) -> List[DetectedEntity]:
        """
        Déduplique les entités en cas de chevauchement.
        
        Priorité: regex > ner-gpu > ner > llm
        En cas de même source, garde l'entité avec le meilleur score.
        
        Args:
            entities: Liste d'entités à dédupliquer
        
        Returns:
            Liste dédupliquée
        """
        if not entities:
            return []
        
        # Définir la priorité des sources
        source_priority = {
            "regex": 4,
            "ner-gpu": 3,
            "ner": 2,
            "ner-external": 2,
            "llm": 1,
        }
        
        # Trier par priorité de source et score
        sorted_entities = sorted(
            entities,
            key=lambda e: (-source_priority.get(e.source, 0), -e.score, e.start)
        )
        
        result = []
        for entity in sorted_entities:
            # Vérifier si elle chevauche une entité déjà ajoutée
            overlaps = False
            for existing in result:
                if entity.overlaps_with(existing):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
        
        return result


__all__ = ["DetectionService", "DetectedEntity"]
