from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from atlas_anno.schemas import CharacterProfile, ScenarioSpec, World

# Labels pour lesquels une forme de surface obfusquée (explicit_hard) existe.
HARD_CAPABLE_LABELS = {
    "PERSON_NAME",
    "EMAIL",
    "PHONE",
    "ACCOUNT_ID",
    "AGE_RANGE",
    "TENURE",
    "DEGREE",
}

# Labels qui peuvent être portés par un indice contextuel (implicit), sous
# réserve que l'auteur possède un ContextualCue avec reveals_label correspondant.
IMPLICIT_CAPABLE_LABELS = {"AGE_RANGE", "LOCATION", "NATIONALITY", "FAMILY_STATUS"}

_DIGIT_WORDS = {
    "0": "zéro",
    "1": "un",
    "2": "deux",
    "3": "trois",
    "4": "quatre",
    "5": "cinq",
    "6": "six",
    "7": "sept",
    "8": "huit",
    "9": "neuf",
}

_AGE_HARD_TEXT = {
    "20-24": "je débute à peine ma vie active",
    "25-29": "j'approche doucement de la trentaine",
    "30-34": "j'ai passé le cap de la trentaine",
    "35-39": "j'approche de la quarantaine",
    "40-44": "j'ai passé le cap de la quarantaine",
    "45-49": "j'approche de la cinquantaine",
    "50-54": "j'ai passé le cap de la cinquantaine",
    "55-59": "j'approche de la soixantaine",
    "60-64": "je suis en fin de carrière",
}


@dataclass(frozen=True)
class SurfaceOverride:
    snippet: str
    cue_type: str = ""


def _phone_hard(phone: str) -> str:
    digits = [char for char in phone if char.isdigit()]
    if len(digits) < 4:
        return phone.replace(" ", ".")
    leading = " ".join(_DIGIT_WORDS[d] for d in digits[:2])
    rest = ".".join("".join(pair) for pair in zip(digits[2::2], digits[3::2]))
    return f"{leading} {rest}"


def _email_hard(email: str) -> str:
    return (
        email.replace("@", " arobase ")
        .replace(".", " point ")
        .strip()
    )


def _person_name_hard(full_name: str) -> str:
    parts = full_name.split()
    if len(parts) < 2:
        return full_name
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def _account_id_hard(account_id: str) -> str:
    if len(account_id) <= 4:
        return account_id
    return f"{account_id[:4]}***{account_id[-2:]}"


def _tenure_hard(author: CharacterProfile, world: World) -> str:
    if world.calendar_events:
        from atlas_anno.generation.text_generator import _humanize_event

        return f"j'étais déjà là avant {_humanize_event(world.calendar_events[0])}"
    if author.tenure_years <= 1:
        return "je suis arrivé il y a peu"
    return "je suis là depuis un bon moment déjà"


def _degree_hard(degrees: list) -> str:
    level = degrees[0] if degrees else ""
    mapping = {
        "PhD": "j'ai poussé les études aussi loin que possible",
        "Master": "j'ai un bac+5 en poche",
        "MBA": "j'ai complété mon parcours par une école de commerce",
        "Licence": "j'ai un bac+3",
        "BTS": "j'ai un bac+2",
    }
    return mapping.get(level, "j'ai le diplôme qui va avec le poste")


def hard_surface_form(label: str, author: CharacterProfile, world: World) -> Optional[str]:
    """Forme de surface obfusquée (explicit_hard) : la valeur canonique ne doit
    jamais apparaître verbatim dans le snippet retourné."""
    if label == "PHONE":
        return _phone_hard(author.phone)
    if label == "EMAIL":
        return _email_hard(author.email)
    if label == "PERSON_NAME":
        return _person_name_hard(author.full_name)
    if label == "ACCOUNT_ID":
        return _account_id_hard(author.account_id)
    if label == "AGE_RANGE":
        return _AGE_HARD_TEXT.get(author.age_range)
    if label == "TENURE":
        return _tenure_hard(author, world)
    if label == "DEGREE":
        return _degree_hard(author.degrees)
    return None


def implicit_surface_form(label: str, author: CharacterProfile) -> Optional[SurfaceOverride]:
    """Indice contextuel porté par l'auteur pour ce label, s'il existe."""
    for cue in author.contextual_cues:
        if cue.reveals_label == label:
            return SurfaceOverride(snippet=cue.cue_text, cue_type=cue.cue_type)
    return None


def build_surface_overrides(
    scenario: ScenarioSpec,
    author: CharacterProfile,
    world: World,
) -> Dict[str, SurfaceOverride]:
    """Calcule, par label du mention_plan, la forme de surface imposée par le
    mode de difficulté. Les labels explicit_easy gardent leur forme humanisée
    habituelle (absents du résultat)."""
    overrides: Dict[str, SurfaceOverride] = {}
    for entry in scenario.mention_plan:
        if entry.difficulty_mode == "implicit":
            override = implicit_surface_form(entry.label, author)
            if override is not None:
                overrides[entry.label] = override
                continue
            # Pas d'indice disponible : retombe sur la forme hard si possible.
            hard = hard_surface_form(entry.label, author, world)
            if hard:
                overrides[entry.label] = SurfaceOverride(snippet=hard)
        elif entry.difficulty_mode == "explicit_hard":
            hard = hard_surface_form(entry.label, author, world)
            if hard:
                overrides[entry.label] = SurfaceOverride(snippet=hard)
    return overrides
