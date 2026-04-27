from __future__ import annotations

import random
from typing import List

from atlas_anno.schemas import CharacterProfile, StyleProfile, World


FIRST_NAMES = [
    "Nadia", "Karim", "Sophie", "Thomas", "Leila", "Mina", "Yassine", "Camille", "Romain",
    "Salome", "Claire", "Amina", "Leo", "Ines", "Noemie", "Mehdi", "Pauline", "Hugo",
]
LAST_NAMES = [
    "Mercier", "Diallo", "Bernard", "Garcia", "Petit", "Fontaine", "Roussel", "Marchand",
    "Benali", "Dupuis", "Meyer", "Carlier", "Robin", "Lopez", "Boyer", "Gautier", "Rolland",
]
COUNTRIES = ["France", "Belgique", "Suisse", "Maroc", "Canada"]
LOCATIONS = ["Paris", "Lyon", "Lille", "Nantes", "Toulouse", "Bordeaux", "Bruxelles", "Montreal"]
SENIORITIES = ["junior", "mid", "senior", "lead"]
DEGREES = [["Licence"], ["Master"], ["Master", "MBA"], ["PhD"]]
CERTIFICATIONS = [
    ["azure-ai-architect"],
    ["aws-cloud-practitioner"],
    ["itil-v4"],
    ["okta-admin"],
    ["google-professional-data-engineer"],
]
SKILLS = [
    ["python", "rag", "prompting"],
    ["support enterprise", "oauth", "sso"],
    ["incident response", "ticket triage", "runbooks"],
    ["hris", "workflows", "reporting"],
    ["vector db", "observability", "api design"],
]
SENSITIVE_VALUES = [
    "HEALTH",
    "ETHNICITY",
    "RELIGION",
    "DISABILITY",
    "FAMILY_STATUS",
    "SEXUAL_ORIENTATION",
    "LEGAL",
    "FINANCIAL",
]


def _style_profile(index: int) -> StyleProfile:
    connectors = [
        ["donc", "par contre"],
        ["du coup", "à ce stade"],
        ["en revanche", "merci d'avance"],
        ["dans ce contexte", "à noter"],
    ]
    return StyleProfile(
        formality="medium" if index % 2 == 0 else "high",
        signature_pattern="thanks_name" if index % 3 else "full_signature",
        verbosity="short" if index % 2 else "medium",
        emoji_usage="none",
        favorite_connectors=connectors[index % len(connectors)],
        jargon_pattern="support_ops" if index % 2 else "ai_ops",
    )


def build_characters(worlds: List[World], per_world: int, seed: int = 27) -> List[CharacterProfile]:
    rng = random.Random(seed)
    characters: List[CharacterProfile] = []
    counter = 0

    for world_index, world in enumerate(worlds):
        for local_index in range(per_world):
            counter += 1
            full_name = f"{FIRST_NAMES[counter % len(FIRST_NAMES)]} {LAST_NAMES[(counter + world_index) % len(LAST_NAMES)]}"
            department = world.departments[local_index % len(world.departments)]
            team = world.teams[local_index % len(world.teams)]
            degree = DEGREES[local_index % len(DEGREES)]
            certifications = CERTIFICATIONS[(local_index + 1) % len(CERTIFICATIONS)]
            skills = SKILLS[(local_index + 2) % len(SKILLS)]
            rare_traits = []
            if local_index % 11 == 0:
                rare_traits.append("only_phd_under_30_in_team")
            if local_index % 13 == 0:
                rare_traits.append("owner_incident_unique")
            if local_index % 17 == 0:
                rare_traits.append("only_certified_on_connector")

            sensitive_attributes = []
            if local_index % 7 == 0:
                sensitive_attributes.append(SENSITIVE_VALUES[(local_index + world_index) % len(SENSITIVE_VALUES)])

            role = {
                "AI Solutions": "AI Support Engineer",
                "Customer Care": "Customer Success Specialist",
                "IT Operations": "Identity Administrator",
                "HR Services": "HR Operations Partner",
                "Platform Engineering": "Platform Reliability Engineer",
                "Billing Operations": "Billing Operations Analyst",
                "Identity Programs": "Access Governance Specialist",
                "Analytics Delivery": "Analytics Delivery Lead",
                "Partner Success": "Partner Success Manager",
            }.get(department, "Operations Analyst")

            age_range = ["25-29", "30-34", "35-39", "40-49"][local_index % 4]
            phone_suffix = f"{counter:04d}"
            username = f"{full_name.split()[0].lower()}.{full_name.split()[1].lower()}"
            characters.append(
                CharacterProfile(
                    person_id=f"p_{counter:04d}",
                    full_name=full_name,
                    email=f"{username}@{world.organization_name.lower().replace(' ', '')}.example",
                    phone=f"+33 6 00 {phone_suffix[:2]} {phone_suffix[2:]}",
                    username=username,
                    account_id=f"ACC-{world_index + 1:02d}-{counter:04d}",
                    language=world.language,
                    country=COUNTRIES[counter % len(COUNTRIES)],
                    location=LOCATIONS[(counter + 3) % len(LOCATIONS)],
                    age_range=age_range,
                    gender="female" if counter % 2 == 0 else "male",
                    nationality=COUNTRIES[(counter + 1) % len(COUNTRIES)],
                    organization_id=world.organization_id,
                    department=department,
                    team=team,
                    role=role,
                    seniority=SENIORITIES[counter % len(SENIORITIES)],
                    tenure_years=(counter + world_index) % 8,
                    degrees=degree,
                    skills=skills,
                    certifications=certifications,
                    rare_traits=rare_traits,
                    events=rng.sample(world.incidents + world.calendar_events, k=2),
                    sensitive_attributes=sensitive_attributes,
                    style_profile=_style_profile(counter),
                )
            )
    return characters
