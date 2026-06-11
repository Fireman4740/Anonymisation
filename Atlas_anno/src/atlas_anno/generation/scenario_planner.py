from __future__ import annotations

import random
from typing import Dict, List, Optional

from atlas_anno.constants import DIFFICULTIES, DOMAIN_EMAIL, DOMAIN_SUPPORT
from atlas_anno.generation.surface_forms import HARD_CAPABLE_LABELS, IMPLICIT_CAPABLE_LABELS
from atlas_anno.schemas import CandidatePools, CharacterProfile, MentionPlanEntry, ScenarioSpec


GOALS = {
    DOMAIN_SUPPORT: [
        "request_help",
        "incident_report",
        "access_issue",
        "integration_issue",
        "billing_mismatch",
        "permission_review",
        "data_gap",
        "sync_delay",
    ],
    DOMAIN_EMAIL: [
        "coordination",
        "project_followup",
        "urgent_incident",
        "context_share",
        "handover",
        "vendor_followup",
        "capacity_alert",
        "delivery_risk",
    ],
}

# Mix de modes de difficulté par mention, par difficulté document (RAT-Bench).
DEFAULT_MENTION_MODE_MIX = {
    "easy": {"explicit_easy": 0.80, "explicit_hard": 0.15, "implicit": 0.05},
    "medium": {"explicit_easy": 0.45, "explicit_hard": 0.35, "implicit": 0.20},
    "hard": {"explicit_easy": 0.10, "explicit_hard": 0.45, "implicit": 0.45},
}
DIRECT_LABELS = ("PERSON_NAME", "EMAIL", "PHONE", "USERNAME", "ACCOUNT_ID")

SUPPORT_INTERNAL_RECIPIENTS = ("service_desk", "identity_support", "data_ops")
SUPPORT_EXTERNAL_RECIPIENTS = ("customer_support", "billing_support", "partner_support")
EMAIL_INTERNAL_RECIPIENTS = ("project_team", "ops_manager", "platform_team")
EMAIL_EXTERNAL_RECIPIENTS = ("client_success", "vendor_contact")


def _difficulty_for(index: int, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.40:
        return "easy"
    if roll < 0.75:
        return "medium"
    return "hard"


def _author_split_map(characters: List[CharacterProfile]) -> Dict[str, str]:
    total = len(characters)
    train_cutoff = int(total * 0.60)
    dev_cutoff = int(total * 0.80)
    mapping: Dict[str, str] = {}
    for index, character in enumerate(characters):
        if index < train_cutoff:
            mapping[character.person_id] = "train"
        elif index < dev_cutoff:
            mapping[character.person_id] = "dev"
        else:
            mapping[character.person_id] = "test"
    return mapping


def _recipient_role_for(domain: str, rng: random.Random) -> str:
    if domain == DOMAIN_SUPPORT:
        pool = SUPPORT_EXTERNAL_RECIPIENTS if rng.random() < 0.40 else SUPPORT_INTERNAL_RECIPIENTS
        return pool[int(rng.random() * len(pool)) % len(pool)]
    pool = EMAIL_EXTERNAL_RECIPIENTS if rng.random() < 0.20 else EMAIL_INTERNAL_RECIPIENTS
    return pool[int(rng.random() * len(pool)) % len(pool)]


def _goal_for(domain: str, index: int, rng: random.Random) -> str:
    pool = GOALS[domain]
    return pool[(index + rng.randrange(len(pool))) % len(pool)]


EXTERNAL_RECIPIENTS = set(SUPPORT_EXTERNAL_RECIPIENTS) | set(EMAIL_EXTERNAL_RECIPIENTS)


def effective_register(author: CharacterProfile, recipient_role: str) -> tuple[str, str]:
    """Registre et tutoiement effectifs du document après règles de cohérence :
    le familier est rétrogradé en courant face à un destinataire externe, et le
    soutenu impose le vouvoiement."""
    register = author.style_profile.register or "courant"
    address_form = author.style_profile.address_form or "vous"
    if register == "familier" and recipient_role in EXTERNAL_RECIPIENTS:
        register = "courant"
        address_form = "vous"
    if register == "soutenu":
        address_form = "vous"
    return register, address_form


def _draw_mode(rng: random.Random, mix: Dict[str, float], allowed: List[str]) -> str:
    weights = [float(mix.get(mode, 0.0)) for mode in allowed]
    if not any(weights):
        return "explicit_easy"
    return rng.choices(allowed, weights=weights, k=1)[0]


def build_mention_plan(
    rng: random.Random,
    difficulty: str,
    author: CharacterProfile,
    required_signals: List[str],
    implicit_signals: List[str],
    include_direct_identifiers: bool,
    include_sensitive: bool,
    mention_mode_mix: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[MentionPlanEntry]:
    """Assigne un mode de difficulté à chaque label attendu du document.

    Contraintes : les identifiants directs ne sont jamais implicit ; un label
    n'est implicit que si l'auteur porte un indice contextuel correspondant et
    explicit_hard que si une forme obfusquée existe ; les labels sensibles
    préfèrent implicit quand un indice existe.
    """
    mix = (mention_mode_mix or DEFAULT_MENTION_MODE_MIX).get(
        difficulty, DEFAULT_MENTION_MODE_MIX["medium"]
    )
    cue_labels = {cue.reveals_label for cue in author.contextual_cues}
    plan: List[MentionPlanEntry] = []

    for label in required_signals:
        allowed = ["explicit_easy"]
        if label in HARD_CAPABLE_LABELS:
            allowed.append("explicit_hard")
        if label in IMPLICIT_CAPABLE_LABELS and label in cue_labels:
            allowed.append("implicit")
        plan.append(MentionPlanEntry(label=label, difficulty_mode=_draw_mode(rng, mix, allowed)))

    # Signaux de style : portés tels quels (le pattern est déjà implicite par nature).
    for label in implicit_signals:
        plan.append(MentionPlanEntry(label=label, difficulty_mode="explicit_easy"))

    if include_direct_identifiers:
        for label in DIRECT_LABELS:
            allowed = ["explicit_easy"]
            if label in HARD_CAPABLE_LABELS:
                allowed.append("explicit_hard")
            plan.append(MentionPlanEntry(label=label, difficulty_mode=_draw_mode(rng, mix, allowed)))

    if include_sensitive:
        for label in author.sensitive_attributes:
            if label in IMPLICIT_CAPABLE_LABELS and label in cue_labels:
                mode = "implicit" if rng.random() < 0.7 else "explicit_easy"
            else:
                mode = "explicit_easy"
            plan.append(MentionPlanEntry(label=label, difficulty_mode=mode))

    return plan


def build_candidate_pools(author: CharacterProfile, characters: List[CharacterProfile]) -> CandidatePools:
    same_org = [candidate.person_id for candidate in characters if candidate.organization_id == author.organization_id]
    same_team = [candidate.person_id for candidate in characters if candidate.team == author.team]
    insider = list(dict.fromkeys([author.person_id] + same_team[:3]))
    org_internal = list(dict.fromkeys([author.person_id] + same_org[:8]))
    public = org_internal[:]
    return CandidatePools(public=public, org_internal=org_internal, insider=insider)


def build_scenarios(
    characters: List[CharacterProfile],
    documents: int,
    seed: int = 41,
    domain_schedule: Optional[List[str]] = None,
    difficulty_schedule: Optional[List[str]] = None,
    mention_mode_mix: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[ScenarioSpec]:
    rng = random.Random(seed)
    author_split = _author_split_map(characters)
    scenarios: List[ScenarioSpec] = []
    for index in range(documents):
        author = characters[index % len(characters)]
        domain = domain_schedule[index] if domain_schedule else (DOMAIN_SUPPORT if index % 2 == 0 else DOMAIN_EMAIL)
        difficulty = difficulty_schedule[index] if difficulty_schedule else _difficulty_for(index, rng)
        split = author_split[author.person_id]
        if difficulty_schedule is None and split == "test" and not any(item.split == "test_hard" for item in scenarios):
            difficulty = "hard"
        final_split = "test_hard" if split == "test" and difficulty == "hard" else ("test_standard" if split == "test" else split)

        required_signals = ["ROLE", "TEAM"]
        if difficulty in {"medium", "hard"}:
            required_signals.extend(["AGE_RANGE", "TENURE"])
        if author.degrees and difficulty == "hard":
            required_signals.append("DEGREE")
        if author.certifications and difficulty == "hard":
            required_signals.append("CERTIFICATION")
        if author.rare_traits:
            required_signals.append("RARE_RESPONSIBILITY")

        implicit_signals = ["SIGNATURE_PATTERN"] if domain == DOMAIN_EMAIL else ["JARGON_PATTERN"]
        include_direct = rng.random() < {"easy": 0.75, "medium": 0.35, "hard": 0.10}[difficulty]
        include_sensitive = bool(author.sensitive_attributes) and rng.random() < 0.60

        recipient_role = _recipient_role_for(domain, rng)
        register, address_form = effective_register(author, recipient_role)

        deduped_signals = list(dict.fromkeys(required_signals))
        mention_plan = build_mention_plan(
            rng,
            difficulty,
            author,
            deduped_signals,
            implicit_signals,
            include_direct,
            include_sensitive,
            mention_mode_mix=mention_mode_mix,
        )

        scenarios.append(
            ScenarioSpec(
                scenario_id=f"scenario_{index + 1:06d}",
                domain=domain,
                unit_type="thread_short" if domain == DOMAIN_EMAIL and rng.random() < 0.35 else "single_message",
                language="fr",
                author_id=author.person_id,
                recipient_role=recipient_role,
                document_goal=_goal_for(domain, index, rng),
                difficulty=difficulty,
                required_signals=deduped_signals,
                implicit_signals=implicit_signals,
                mention_plan=mention_plan,
                include_signature=domain == DOMAIN_EMAIL,
                include_direct_identifiers=include_direct,
                include_sensitive=include_sensitive,
                urgency=["low", "medium", "high"][rng.randrange(3)],
                noise_level=["low", "medium", "high"][rng.randrange(3)],
                split=final_split,
                register=register,
                address_form=address_form,
            )
        )
    return scenarios
