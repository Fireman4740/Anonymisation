"""Central data model for the anonymization pipeline.

The pipeline historically passes plain dicts (``EntityDict`` in src/state.py,
canonical keys: start/end/type/value/source/score). These dataclasses are the
typed counterpart: validated, immutable, with lossless dict interop so they can
be introduced incrementally without breaking existing nodes.

Offsets ALWAYS reference the original text.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional


class IdentifierType(str, Enum):
    """Re-identification role of an entity (TAB / RAT-Bench taxonomy)."""

    DIRECT = "direct"          # name, email, SSN — identifies on its own
    QUASI = "quasi"            # job, city, date — identifies in combination
    SENSITIVE = "sensitive"    # health, finances — to protect, not identifying
    NO_MASK = "no_mask"        # explicitly safe to keep
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Default identifier classification per normalized entity type. Used when a
# detector does not set identifier_type itself. Conservative: unlisted → UNKNOWN.
DEFAULT_IDENTIFIER_TYPES: Dict[str, IdentifierType] = {
    "PERSON": IdentifierType.DIRECT,
    "PER": IdentifierType.DIRECT,
    "EMAIL": IdentifierType.DIRECT,
    "PHONE": IdentifierType.DIRECT,
    "TELEPHONE": IdentifierType.DIRECT,
    "IBAN": IdentifierType.DIRECT,
    "BANK_ACCOUNT": IdentifierType.DIRECT,
    "CREDIT_CARD": IdentifierType.DIRECT,
    "SSN": IdentifierType.DIRECT,
    "NATIONAL_ID": IdentifierType.DIRECT,
    "PASSPORT": IdentifierType.DIRECT,
    "USERNAME": IdentifierType.DIRECT,
    "IP_ADDRESS": IdentifierType.DIRECT,
    "LOC": IdentifierType.QUASI,
    "LOCATION": IdentifierType.QUASI,
    "GPE": IdentifierType.QUASI,
    "DATE": IdentifierType.QUASI,
    "AGE": IdentifierType.QUASI,
    "ORG": IdentifierType.QUASI,
    "ORGANIZATION": IdentifierType.QUASI,
    "OCCUPATION": IdentifierType.QUASI,
    "NATIONALITY": IdentifierType.QUASI,
    "HEALTH": IdentifierType.SENSITIVE,
    "MEDICAL": IdentifierType.SENSITIVE,
}


@dataclass(frozen=True)
class EntitySpan:
    """One detected entity. Offsets reference the original text."""

    start: int
    end: int
    text: str
    entity_type: str
    source: str = "unknown"
    confidence: float = 0.0
    identifier_type: IdentifierType = IdentifierType.UNKNOWN
    risk_level: Optional[RiskLevel] = None
    detector_name: Optional[str] = None
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"EntitySpan.start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"EntitySpan requires end > start, got [{self.start}, {self.end})"
            )
        if not self.entity_type:
            raise ValueError("EntitySpan.entity_type must be non-empty")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EntitySpan":
        """Build from a pipeline EntityDict (canonical or legacy key variants)."""
        entity_type = str(raw.get("type") or raw.get("entity_type") or "")
        metadata = {
            key: value
            for key, value in raw.items()
            if key
            not in (
                "start", "end", "type", "entity_type", "value", "text",
                "source", "score", "confidence", "identifier_type",
                "risk_level", "detector_name", "normalized_value",
            )
        }
        identifier_raw = raw.get("identifier_type")
        identifier = (
            IdentifierType(identifier_raw)
            if identifier_raw
            else DEFAULT_IDENTIFIER_TYPES.get(entity_type.upper(), IdentifierType.UNKNOWN)
        )
        risk_raw = raw.get("risk_level")
        return cls(
            start=int(raw["start"]),
            end=int(raw["end"]),
            text=str(raw.get("value") or raw.get("text") or ""),
            entity_type=entity_type,
            source=str(raw.get("source") or "unknown"),
            confidence=float(raw.get("score") or raw.get("confidence") or 0.0),
            identifier_type=identifier,
            risk_level=RiskLevel(risk_raw) if risk_raw else None,
            detector_name=raw.get("detector_name"),
            normalized_value=raw.get("normalized_value"),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the canonical pipeline EntityDict format."""
        out: Dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "value": self.text,
            "source": self.source,
            "score": self.confidence,
            "identifier_type": self.identifier_type.value,
        }
        if self.risk_level is not None:
            out["risk_level"] = self.risk_level.value
        if self.detector_name:
            out["detector_name"] = self.detector_name
        if self.normalized_value:
            out["normalized_value"] = self.normalized_value
        out.update(self.metadata)
        return out

    def overlaps(self, other: "EntitySpan") -> bool:
        return self.start < other.end and other.start < self.end

    def with_type(self, entity_type: str) -> "EntitySpan":
        return replace(self, entity_type=entity_type)


class Strategy(str, Enum):
    """Anonymization strategies supported by AnonymizationNode.apply_strategy."""

    MASK = "mask"
    REDACT = "redact"
    PSEUDO = "pseudo"
    GENERALIZE = "generalize"
    SENSITIVE = "sensitive"
    PARAPHRASE = "paraphrase"
    KEEP = "keep"


@dataclass(frozen=True)
class AnonymizationDecision:
    """Audit record: what was replaced, by what, why, under which policy."""

    span: EntitySpan
    strategy: Strategy
    replacement: str
    reason: str = ""
    policy_name: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span": self.span.to_dict(),
            "strategy": self.strategy.value,
            "replacement": self.replacement,
            "reason": self.reason,
            "policy_name": self.policy_name,
        }


def spans_from_dicts(entities: List[Dict[str, Any]]) -> List[EntitySpan]:
    """Convert a pipeline entity list, skipping malformed entries.

    Mirrors AnonymizationNode tolerance: malformed offsets are dropped, not
    fatal — detection noise must not crash a run.
    """
    spans: List[EntitySpan] = []
    for raw in entities:
        try:
            spans.append(EntitySpan.from_dict(raw))
        except (KeyError, ValueError, TypeError):
            continue
    return spans
