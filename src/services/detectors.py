"""
Detection Service - Encapsulates regex and NER detection logic

This module provides a clean interface for entity detection combining:
- Regex-based detection (emails, phones, IPs, etc.)
- NER-based detection (GLiNER with optional GPU pipeline)
"""
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .text_sanitizer import regexes_based_replacements
from .ner_ensemble_clean import run_gliner, merge_ner_lists, GLINER_ALL_LABELS
from .utils_pseudo import PseudoMapper
from .policy import AnonymizationPolicy


@dataclass
class DetectedEntity:
    """Represents a detected entity with all metadata."""
    start: int
    end: int
    surface: str
    etype: str  # Entity type (PER, ORG, EMAIL, etc.)
    source: str  # "regex", "ner", "ner-gpu"
    score: float = 1.0  # Confidence score
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DetectionService:
    """
    Service for detecting entities using regex and NER.
    
    This service encapsulates all detection logic and can optionally
    use a GPU-optimized NER pipeline.
    """
    
    def __init__(
        self,
        gpu_pipeline = None,  # Optional GPU pipeline from ner_gpu_optimizer
        use_gliner: bool = True,
        gliner_models: Optional[List[str]] = None,
        gliner_labels: Optional[List[str]] = None,
        gliner_threshold: float = 0.35,
    ):
        """
        Initialize detection service.
        
        Args:
            gpu_pipeline: Optional GPU-optimized NER pipeline
            use_gliner: Whether to use GLiNER models
            gliner_models: List of GLiNER model names
            gliner_labels: List of labels to detect
            gliner_threshold: Confidence threshold for GLiNER
        """
        self.gpu_pipeline = gpu_pipeline
        self.use_gliner = use_gliner
        self.gliner_models = gliner_models or [
            "urchade/gliner_large-v2.1",
            "urchade/gliner_multi-v2.1",
        ]
        self.gliner_labels = gliner_labels or GLINER_ALL_LABELS
        self.gliner_threshold = gliner_threshold
    
    def detect_regex(
        self,
        text: str,
        skip_tags: Optional[set] = None,
    ) -> List[DetectedEntity]:
        """
        Detect entities using regex patterns.
        
        Args:
            text: Input text
            skip_tags: Set of tags to skip (e.g., {"EMAIL", "PHONE"})
            
        Returns:
            List of DetectedEntity objects
        """
        regex_hits = regexes_based_replacements(text)
        skip_tags = skip_tags or set()
        
        entities = []
        for s, e, tag in regex_hits:
            etype = tag.strip("<>").upper()
            if etype in skip_tags:
                continue
            
            entities.append(DetectedEntity(
                start=s,
                end=e,
                surface=text[s:e],
                etype=etype,
                source="regex",
            ))
        
        return entities
    
    def detect_ner(
        self,
        text: str,
        external_ner: Optional[List[dict]] = None,
    ) -> List[DetectedEntity]:
        """
        Detect entities using NER (GPU pipeline if available, else GLiNER).
        
        Args:
            text: Input text
            external_ner: Optional pre-computed NER results
            
        Returns:
            List of DetectedEntity objects
        """
        entities = []
        
        # Start with external NER if provided
        local_ner = list(external_ner or [])
        
        # Try GPU pipeline first
        if self.gpu_pipeline is not None:
            try:
                gpu_entities = self.gpu_pipeline.predict(text)
                local_ner = merge_ner_lists(local_ner, gpu_entities)
                source = "ner-gpu"
            except Exception as e:
                print(f"[DetectionService] GPU pipeline error: {e}, falling back to standard GLiNER")
                gpu_entities = run_gliner(
                    text,
                    model_names=self.gliner_models,
                    labels=self.gliner_labels,
                    threshold=self.gliner_threshold,
                ) if self.use_gliner else []
                local_ner = merge_ner_lists(local_ner, gpu_entities)
                source = "ner"
        else:
            # Standard GLiNER
            gliner_entities = run_gliner(
                text,
                model_names=self.gliner_models,
                labels=self.gliner_labels,
                threshold=self.gliner_threshold,
            ) if self.use_gliner else []
            local_ner = merge_ner_lists(local_ner, gliner_entities)
            source = "ner"
        
        # Convert to DetectedEntity format
        for ent in local_ner:
            etype = self._map_label(str(ent.get("entity_group", "")))
            if not etype or etype.startswith("DATE"):
                continue
            
            s = ent.get("start")
            e = ent.get("end")
            if not isinstance(s, int) or not isinstance(e, int) or not (0 <= s < e <= len(text)):
                continue
            
            entities.append(DetectedEntity(
                start=s,
                end=e,
                surface=text[s:e],
                etype=etype,
                source=source,
                score=ent.get("score", ent.get("votes", 1.0)),
                metadata={"raw_label": ent.get("entity_group")},
            ))
        
        return entities
    
    def detect_all(
        self,
        text: str,
        skip_regex_tags: Optional[set] = None,
        external_ner: Optional[List[dict]] = None,
    ) -> List[DetectedEntity]:
        """
        Detect entities using both regex and NER.
        
        Args:
            text: Input text
            skip_regex_tags: Set of regex tags to skip
            external_ner: Optional pre-computed NER results
            
        Returns:
            List of DetectedEntity objects, deduplicated
        """
        regex_entities = self.detect_regex(text, skip_tags=skip_regex_tags)
        ner_entities = self.detect_ner(text, external_ner=external_ner)
        
        # Merge and deduplicate
        all_entities = regex_entities + ner_entities
        return self._deduplicate_entities(all_entities)
    
    def _map_label(self, label: str) -> str:
        """Map NER labels to standard types."""
        L = label.upper()
        if L in {"PER", "PERSON"}:
            return "PER"
        if L in {"ORG", "ORGANIZATION"}:
            return "ORG"
        if L in {"LOC", "LOCATION", "GPE", "FACILITY", "FAC"}:
            return "LOC"
        if L in {"MAIL", "EMAIL", "EMAIL ADDRESS", "E-MAIL"}:
            return "MAIL"
        if L in {"TELEPHONE", "PHONE", "PHONE NUMBER"}:
            return "TELEPHONE"
        if L in {"IP", "IP ADDRESS", "IPV4", "IPV6"}:
            return "IP"
        if L in {"URL", "URI", "LINK"}:
            return "URL"
        if L in {"USERNAME", "USER", "HANDLE", "ACCOUNT"}:
            return "USERNAME"
        return ""
    
    def _deduplicate_entities(self, entities: List[DetectedEntity]) -> List[DetectedEntity]:
        """
        Deduplicate entities, keeping highest priority source.
        Priority: regex > ner-gpu > ner
        """
        priority_map = {"regex": 2, "ner-gpu": 1, "ner": 0}
        
        # Group by (start, end, etype)
        entity_map: Dict[Tuple[int, int, str], DetectedEntity] = {}
        
        for entity in entities:
            key = (entity.start, entity.end, entity.etype)
            
            if key not in entity_map:
                entity_map[key] = entity
            else:
                # Keep entity with higher priority
                existing = entity_map[key]
                if priority_map.get(entity.source, 0) > priority_map.get(existing.source, 0):
                    entity_map[key] = entity
        
        # Sort by start position
        result = sorted(entity_map.values(), key=lambda x: (x.start, x.end))
        return result


def create_detection_service(
    policy: AnonymizationPolicy,
    gpu_pipeline=None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DetectionService:
    """
    Factory function to create a DetectionService from policy and overrides.
    
    Args:
        policy: AnonymizationPolicy
        gpu_pipeline: Optional GPU pipeline
        overrides: Optional configuration overrides
        
    Returns:
        Configured DetectionService instance
    """
    overrides = overrides or {}
    
    use_gliner = bool(overrides.get("ner_use_gliner", True))
    gliner_models = list(overrides.get("gliner_models", [
        "urchade/gliner_large-v2.1",
        "urchade/gliner_multi-v2.1",
    ]))
    gliner_labels = list(overrides.get("gliner_labels", GLINER_ALL_LABELS))
    gliner_threshold = float(overrides.get("gliner_threshold", 0.35))
    
    return DetectionService(
        gpu_pipeline=gpu_pipeline,
        use_gliner=use_gliner,
        gliner_models=gliner_models,
        gliner_labels=gliner_labels,
        gliner_threshold=gliner_threshold,
    )


__all__ = [
    "DetectedEntity",
    "DetectionService",
    "create_detection_service",
]
