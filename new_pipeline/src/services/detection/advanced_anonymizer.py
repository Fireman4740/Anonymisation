"""Advanced anonymization detector combining regex, Schwifty, phonenumbers and Flair.

This module implements the two-phase pipeline described in the implementation guide:
- Phase 1: deterministic detection via regex/libraries (including secrets, IBAN/BIC, phones).
- Phase 2: contextual NER via Flair (with graceful fallbacks when models are missing).

It exposes a light API used by the DetectionService to enrich the entity list
before pseudonymisation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

try:  # pragma: no cover - external dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from schwifty import IBAN  # type: ignore
except Exception:  # pragma: no cover
    IBAN = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import phonenumbers  # type: ignore
except Exception:  # pragma: no cover
    phonenumbers = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from flair.models import SequenceTagger  # type: ignore
    from flair.data import Sentence  # type: ignore
except Exception:  # pragma: no cover
    SequenceTagger = None  # type: ignore
    Sentence = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

try:  # pragma: no cover - local dependency
    from ..regex import text_sanitizer  # type: ignore
except Exception:  # pragma: no cover
    text_sanitizer = None  # type: ignore


@dataclass
class AdvancedEntity:
    start: int
    end: int
    value: str
    etype: str
    source: str
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "value": self.value,
            "type": self.etype,
            "source": self.source,
            "score": self.score,
        }


class AdvancedAnonymizer:
    """High-level detector orchestrating regex/libraries and Flair."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        *,
        enable_ner: bool = True,
        config_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config_path = Path(config_path or Path.cwd() / "patterns_config.yaml")
        self.enable_ner = enable_ner
        self._explicit_config = deepcopy(config_data) if config_data is not None else None
        self.config: Dict[str, Any] = {"patterns": {}}
        self.pattern_specs: List[tuple[str, Dict[str, Any]]] = []
        self.flair_tagger = None
        self.spacy_nlp = None
        self._load_config()
        self._prepare_patterns()
        self._load_models()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        if self._explicit_config is not None:
            self.config = deepcopy(self._explicit_config)
            return
        if yaml and self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if isinstance(data, dict):
                self.config = data
                return
        if text_sanitizer and hasattr(text_sanitizer, "get_patterns_config"):
            try:
                self.config = text_sanitizer.get_patterns_config()
                return
            except Exception:
                pass

    def _prepare_patterns(self) -> None:
        patterns = self.config.get("patterns", {})
        if not isinstance(patterns, dict):
            self.pattern_specs = []
            return
        self.pattern_specs = sorted(
            patterns.items(),
            key=lambda item: item[1].get("priority", 999),
        )

    def _load_models(self) -> None:
        if not self.enable_ner:
            return
        if SequenceTagger is not None and Sentence is not None:
            try:  # pragma: no cover - heavy model
                self.flair_tagger = SequenceTagger.load("flair/ner-french")
            except Exception:
                self.flair_tagger = None
        if spacy is not None:
            try:  # pragma: no cover - heavy model
                self.spacy_nlp = spacy.load("fr_core_news_sm")
            except Exception:
                self.spacy_nlp = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        entities: List[AdvancedEntity] = []
        entities.extend(self._phase1_regex_detection(text))
        entities.extend(self._phase1_iban_detection(text))
        entities.extend(self._phase1_phone_detection(text))

        if self.enable_ner and self.flair_tagger and Sentence:
            entities.extend(self._phase2_ner_detection(text))

        merged = self._merge_overlaps(entities)
        return [ent.to_dict() for ent in merged]

    def anonymize(self, text: str) -> tuple[str, List[Dict[str, Any]]]:
        """Return a naive anonymized string plus metadata (utility helper)."""
        entities = self.detect_entities(text)
        replacements: List[Dict[str, Any]] = []
        offset = 0
        new_text = text
        for ent in entities:
            placeholder = f"[{ent['type']}_XXX]"
            start = ent["start"] + offset
            end = ent["end"] + offset
            new_text = new_text[:start] + placeholder + new_text[end:]
            delta = len(placeholder) - (ent["end"] - ent["start"])
            offset += delta
            replacements.append({
                "start": ent["start"],
                "end": ent["end"],
                "surface": ent["value"],
                "placeholder": placeholder,
                "type": ent["type"],
            })
        return new_text, replacements

    # ------------------------------------------------------------------
    # Phase 1 – Regex / deterministic patterns
    # ------------------------------------------------------------------
    def _phase1_regex_detection(self, text: str) -> List[AdvancedEntity]:
        entities: List[AdvancedEntity] = []
        for name, spec in self.pattern_specs:
            if not spec.get("enabled", True):
                continue
            if spec.get("type") == "library":
                # couvert par les méthodes dédiées
                continue
            regex = spec.get("regex")
            if not isinstance(regex, str):
                continue
            entity_type = str(spec.get("entity_type", name)).upper()
            pattern = re.compile(regex)
            for match in pattern.finditer(text):
                value = match.group()
                if not self._validate_optional(spec, value):
                    continue
                entities.append(
                    AdvancedEntity(
                        start=match.start(),
                        end=match.end(),
                        value=value,
                        etype=entity_type,
                        source=f"advanced-regex:{name}",
                    )
                )
        return entities

    def _phase1_iban_detection(self, text: str) -> List[AdvancedEntity]:
        if IBAN is None:
            return []
        pattern = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){9,30}\b")
        entities: List[AdvancedEntity] = []
        for match in pattern.finditer(text):
            raw = match.group()
            candidate = raw.replace(" ", "")
            try:
                iban = IBAN(candidate)
            except Exception:
                continue
            entities.append(
                AdvancedEntity(
                    start=match.start(),
                    end=match.end(),
                    value=raw,
                    etype="IBAN",
                    source="advanced-iban",
                )
            )
            if getattr(iban, "bic", None):
                bic = iban.bic
                idx = text.find(bic)
                if idx != -1:
                    entities.append(
                        AdvancedEntity(
                            start=idx,
                            end=idx + len(bic),
                            value=bic,
                            etype="BIC",
                            source="advanced-iban",
                        )
                    )
        return entities

    def _phase1_phone_detection(self, text: str) -> List[AdvancedEntity]:
        if phonenumbers is None:
            return []
        pattern = re.compile(r"(?:\+|00)[1-9]\d{1,14}|(?:\+33|0)[1-9]\d{8}")
        entities: List[AdvancedEntity] = []
        for match in pattern.finditer(text):
            candidate = match.group()
            try:
                parsed = phonenumbers.parse(candidate, None)
                if not phonenumbers.is_valid_number(parsed):
                    continue
                country = phonenumbers.region_code_for_number(parsed)
            except Exception:
                continue
            entities.append(
                AdvancedEntity(
                    start=match.start(),
                    end=match.end(),
                    value=match.group(),
                    etype="TELEPHONE",
                    source="advanced-phone",
                    score=1.0,
                )
            )
            entities[-1].source += f"::{country}" if country else ""
        return entities

    def _validate_optional(self, spec: Dict[str, Any], value: str) -> bool:
        validator = spec.get("validate_with")
        if validator == "luhn":
            digits = re.sub(r"\D", "", value)
            if len(digits) < 13:
                return False
            checksum = 0
            rev = digits[::-1]
            for idx, digit in enumerate(rev):
                n = int(digit)
                if idx % 2 == 1:
                    n *= 2
                    if n > 9:
                        n -= 9
                checksum += n
            return checksum % 10 == 0
        return True

    # ------------------------------------------------------------------
    # Phase 2 – Flair NER
    # ------------------------------------------------------------------
    def _phase2_ner_detection(self, text: str) -> List[AdvancedEntity]:
        if not self.flair_tagger or not Sentence:
            return []
        sentence = Sentence(text)
        try:
            self.flair_tagger.predict(sentence)
        except Exception:
            return []
        entities: List[AdvancedEntity] = []
        for span in sentence.get_spans("ner"):
            entities.append(
                AdvancedEntity(
                    start=span.start_position,
                    end=span.end_position,
                    value=span.text,
                    etype=span.tag.upper(),
                    source="advanced-ner",
                    score=float(span.score or 0.0),
                )
            )
        return entities

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _merge_overlaps(self, entities: List[AdvancedEntity]) -> List[AdvancedEntity]:
        if not entities:
            return []
        sorted_entities = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        merged: List[AdvancedEntity] = []
        for ent in sorted_entities:
            if not merged:
                merged.append(ent)
                continue
            last = merged[-1]
            if ent.start >= last.end:
                merged.append(ent)
                continue
            # overlap: keep the longer span / regex first
            last_len = last.end - last.start
            ent_len = ent.end - ent.start
            if ent_len > last_len:
                merged[-1] = ent
        return merged


__all__ = ["AdvancedAnonymizer"]
