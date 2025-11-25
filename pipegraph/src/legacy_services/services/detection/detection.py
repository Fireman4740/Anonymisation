"""
Service de Détection Unifié

Ce service encapsule toute la logique de détection d'entités :
- Détection regex (patterns PII)
- Détection NER (GLiNER avec support GPU)
- Déduplication et fusion des résultats
"""

from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import re

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...core.policy import AnonymizationPolicy

try:
    from ..regex.text_sanitizer import (
        regexes_based_replacements,
        get_patterns_config,
    )
    from ..ner.ensemble import run_gliner, merge_ner_lists, GLINER_ALL_LABELS
    from .advanced_anonymizer import AdvancedAnonymizer
except ImportError as e1:
    try:
        from src.services.regex.text_sanitizer import (
            regexes_based_replacements,
            get_patterns_config,
        )
        from src.services.ner.ensemble import run_gliner, merge_ner_lists, GLINER_ALL_LABELS
        from src.services.detection.advanced_anonymizer import AdvancedAnonymizer
    except ImportError as e2:
        # Fallback minimal si modules non trouvés
        import sys
        print(f"[WARNING] Detection service fallback utilisé. Erreurs d'import:", file=sys.stderr)
        print(f"  - Import relatif: {e1}", file=sys.stderr)
        print(f"  - Import absolu: {e2}", file=sys.stderr)
        def regexes_based_replacements(text: str):  # type: ignore
            return []

        def get_patterns_config():  # type: ignore
            return {"patterns": {}}
        def run_gliner(*args, **kwargs):  # type: ignore
            return []
        def merge_ner_lists(*args, **kwargs):  # type: ignore
            return []
        GLINER_ALL_LABELS = []  # type: ignore
        AdvancedAnonymizer = None  # type: ignore


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
        advanced_detector=None,
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
        self.advanced_detector = advanced_detector
    
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
        # Détecter avec regex (classique + pipeline avancé)
        advanced_regex, advanced_ner = self._detect_with_advanced(text, skip_regex_tags or set())
        regex_entities = advanced_regex + self.detect_regex(text, skip_regex_tags)
        
        # Détecter avec NER
        combined_external = self._merge_external_ner(external_ner, advanced_ner)
        ner_entities = self.detect_ner(text, combined_external)
        
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

    def _detect_with_advanced(
        self,
        text: str,
        skip_tags: Set[str],
    ) -> Tuple[List[DetectedEntity], List[Dict[str, Any]]]:
        if not getattr(self, "advanced_detector", None):
            return [], []
        try:
            detected = self.advanced_detector.detect_entities(text)  # type: ignore[attr-defined]
        except Exception:
            return [], []
        regex_like: List[DetectedEntity] = []
        ner_like: List[Dict[str, Any]] = []
        for ent in detected:
            start = ent.get("start")
            end = ent.get("end")
            etype = str(ent.get("type", "")).upper()
            if start is None or end is None or end <= start or not etype:
                continue
            if etype in skip_tags:
                continue
            source = str(ent.get("source", "advanced"))
            score = float(ent.get("score", 1.0) or 1.0)
            if source.startswith("advanced-ner"):
                ner_like.append({
                    "start": start,
                    "end": end,
                    "entity_group": etype,
                    "score": score,
                })
            else:
                regex_like.append(
                    DetectedEntity(
                        start=start,
                        end=end,
                        surface=text[start:end],
                        etype=etype,
                        source=source,
                        score=score,
                        metadata={"advanced": True},
                    )
                )
        return regex_like, ner_like

    def _merge_external_ner(
        self,
        original_external: Optional[List[dict]],
        advanced_ner: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not original_external and not advanced_ner:
            return original_external
        combined = list(original_external or [])
        if advanced_ner:
            combined.extend(advanced_ner)
        return combined


def create_detection_service(
    policy: Optional["AnonymizationPolicy"] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DetectionService:
    """Factory helper to configure ``DetectionService`` from policy/overrides."""
    overrides = overrides or {}

    gpu_pipeline = None
    use_gpu = bool(overrides.get("use_gpu_ner", True))
    
    if use_gpu:
        try:
            from ..ner.gpu_optimizer import create_optimized_pipeline, load_gpu_config

            gpu_cfg = load_gpu_config()
            user_gpu_cfg = overrides.get("ner_gpu_config")
            if isinstance(user_gpu_cfg, dict):
                gpu_cfg.update(user_gpu_cfg)
            gpu_pipeline = create_optimized_pipeline(gpu_cfg)
        except Exception:
            gpu_pipeline = None

    def _float(val: Any, default: float) -> float:
        try:
            return float(val)
        except Exception:
            return default

    advanced_detector = None
    use_advanced = bool(overrides.get("use_advanced_anonymizer", True))
    enable_advanced_ner = bool(overrides.get("advanced_anonymizer_enable_ner", True))
    patterns_path = overrides.get("patterns_config_path")
    patterns_data = overrides.get("patterns_config_data")
    if patterns_data is None and not patterns_path:
        try:
            patterns_data = get_patterns_config()
        except Exception:
            patterns_data = None
    if use_advanced and AdvancedAnonymizer is not None:
        try:
            advanced_detector = AdvancedAnonymizer(
                config_path=patterns_path,
                enable_ner=enable_advanced_ner,
                config_data=patterns_data,
            )
        except Exception as exc:
            print(f"[DetectionService] AdvancedAnonymizer disabled: {exc}")

    return DetectionService(
        gpu_pipeline=gpu_pipeline,
        use_gliner=bool(overrides.get("use_gliner", True)),
        gliner_models=overrides.get("gliner_models"),
        gliner_labels=overrides.get("gliner_labels"),
        gliner_threshold=_float(overrides.get("gliner_threshold", 0.35), 0.35),
        gliner_preset=str(overrides.get("gliner_preset", "balanced")),
        advanced_detector=advanced_detector,
    )


__all__ = ["DetectionService", "DetectedEntity", "create_detection_service"]
