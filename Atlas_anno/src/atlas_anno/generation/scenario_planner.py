from __future__ import annotations

import random
from typing import Dict, List, Optional

from atlas_anno.constants import DIFFICULTIES, DOMAIN_EMAIL, DOMAIN_SUPPORT
from atlas_anno.schemas import CandidatePools, CharacterProfile, ScenarioSpec


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
) -> List[ScenarioSpec]:
    rng = random.Random(seed)
    author_split = _author_split_map(characters)
    scenarios: List[ScenarioSpec] = []
    for index in range(documents):
        author = characters[index % len(characters)]
        domain = domain_schedule[index] if domain_schedule else (DOMAIN_SUPPORT if index % 2 == 0 else DOMAIN_EMAIL)
        difficulty = difficulty_schedule[index] if difficulty_schedule else _difficulty_for(index, rng)
        split = author_split[author.person_id]
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

        scenarios.append(
            ScenarioSpec(
                scenario_id=f"scenario_{index + 1:06d}",
                domain=domain,
                unit_type="thread_short" if domain == DOMAIN_EMAIL and rng.random() < 0.35 else "single_message",
                language="fr",
                author_id=author.person_id,
                recipient_role=_recipient_role_for(domain, rng),
                document_goal=_goal_for(domain, index, rng),
                difficulty=difficulty,
                required_signals=list(dict.fromkeys(required_signals)),
                implicit_signals=implicit_signals,
                include_signature=domain == DOMAIN_EMAIL,
                include_direct_identifiers=include_direct,
                include_sensitive=include_sensitive,
                urgency=["low", "medium", "high"][rng.randrange(3)],
                noise_level=["low", "medium", "high"][rng.randrange(3)],
                split=final_split,
            )
        )
    return scenarios
