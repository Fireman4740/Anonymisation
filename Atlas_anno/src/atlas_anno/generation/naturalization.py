from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List

from atlas_anno.io import serialize
from atlas_anno.schemas import CharacterProfile, DocumentRecord, ScenarioSpec

# Labels du plan ↔ champs persona à exposer au naturaliseur. Tout le reste de
# la persona est volontairement absent du payload (anti-fuite + prompt minimal).
_LABEL_PERSONA_FIELDS = {
    "PERSON_NAME": ("full_name",),
    "EMAIL": ("email",),
    "PHONE": ("phone",),
    "USERNAME": ("username",),
    "ACCOUNT_ID": ("account_id",),
    "ROLE": ("role",),
    "TEAM": ("team",),
    "DEPARTMENT": ("department",),
    "LOCATION": ("location",),
    "NATIONALITY": ("nationality",),
    "AGE_RANGE": ("age_range",),
    "TENURE": ("tenure_years",),
    "DEGREE": ("degrees",),
    "CERTIFICATION": ("certifications",),
    "RARE_RESPONSIBILITY": ("rare_traits",),
    "EVENT_DATE": ("events",),
}


def build_style_directives(author: CharacterProfile, scenario: ScenarioSpec) -> Dict[str, Any]:
    """Bloc compact de conditionnement stylistique (données, pas roleplay)."""
    style = author.style_profile
    return {
        "register": scenario.register or style.register,
        "address_form": scenario.address_form or style.address_form,
        "francophone_variety": style.francophone_variety,
        "expertise_level": style.expertise_level,
        "verbosity": style.verbosity,
        "connectors": list(style.favorite_connectors),
        "typo_propensity": style.typo_propensity,
    }


def build_persona_block(author: CharacterProfile, document: DocumentRecord) -> Dict[str, Any]:
    """Persona réduite aux attributs réellement couverts par le plan du document."""
    allowed_labels = set(document.metadata.get("signal_values", {}))
    allowed_labels.update(entry.label for entry in document.scenario.mention_plan)

    persona: Dict[str, Any] = {"person_id": author.person_id, "seniority": author.seniority}
    for label, fields in _LABEL_PERSONA_FIELDS.items():
        if label not in allowed_labels:
            continue
        for field_name in fields:
            value = getattr(author, field_name, None)
            if value:
                persona[field_name] = value
    if any(label in allowed_labels for label in author.sensitive_attributes):
        persona["sensitive_attributes"] = [
            label for label in author.sensitive_attributes if label in allowed_labels
        ]
    implicit_labels = {
        entry.label for entry in document.scenario.mention_plan if entry.difficulty_mode == "implicit"
    }
    cues = [cue for cue in author.contextual_cues if cue.reveals_label in implicit_labels]
    if cues:
        persona["contextual_cues"] = serialize(cues)
    return persona


def variant_order(doc_id: str, count: int) -> List[int]:
    """Ordre déterministe d'essai des variantes (Verbalized Sampling)."""
    if count <= 1:
        return list(range(count))
    seed = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:12], 16)
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order
