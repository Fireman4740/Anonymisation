from __future__ import annotations

import random
from typing import Dict, List

from atlas_anno.generation.attribute_sampler import (
    IdentitySampler,
    MarginalSampler,
    load_contextual_cues_catalog,
    load_insee_marginals,
    sample_contextual_cues,
)
from atlas_anno.generation.style_sampler import StyleFactorSampler
from atlas_anno.schemas import CharacterProfile, World


SENIORITIES = ["junior", "mid", "senior", "lead"]
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

# Quasi-identifiants soumis au contrôle d'unicité (critère SynthPAI : au plus
# un attribut partagé entre deux profils d'un même monde).
UNIQUENESS_FIELDS = ("age_range", "location", "role", "tenure_years")
MAX_UNIQUENESS_ATTEMPTS = 50


def _shared_quasi_count(left: CharacterProfile, right: CharacterProfile) -> int:
    return sum(1 for field in UNIQUENESS_FIELDS if getattr(left, field) == getattr(right, field))


def _resample_collision(
    profile: CharacterProfile, other: CharacterProfile, sampler: MarginalSampler
) -> bool:
    """Resample un attribut en collision (hors rôle, lié au département)."""
    if profile.location == other.location:
        region, city = sampler.sample_location(profile.country)
        profile.region = region
        profile.location = city
        return True
    if profile.tenure_years == other.tenure_years:
        profile.tenure_years = sampler.sample_tenure_years()
        return True
    if profile.age_range == other.age_range:
        profile.age_range = sampler.sample_age_range(profile.gender)
        profile.birth_year = sampler.sample_birth_year(profile.age_range)
        return True
    return False


def _enforce_world_uniqueness(profiles: List[CharacterProfile], sampler: MarginalSampler) -> int:
    """Valide les profils un à un contre les précédents (jamais re-mutés) : converge."""
    resample_count = 0
    for index in range(1, len(profiles)):
        profile = profiles[index]
        attempts = 0
        while attempts < MAX_UNIQUENESS_ATTEMPTS:
            conflict = next(
                (other for other in profiles[:index] if _shared_quasi_count(profile, other) > 1),
                None,
            )
            if conflict is None or not _resample_collision(profile, conflict, sampler):
                break
            resample_count += 1
            attempts += 1
    return resample_count


def build_characters(
    worlds: List[World],
    per_world: int,
    seed: int = 27,
    stats: Dict[str, int] | None = None,
) -> List[CharacterProfile]:
    rng = random.Random(seed)
    marginal_sampler = MarginalSampler(rng, load_insee_marginals())
    identity_sampler = IdentitySampler(seed)
    style_sampler = StyleFactorSampler(seed)
    cues_catalog = load_contextual_cues_catalog()
    characters: List[CharacterProfile] = []
    uniqueness_resample_count = 0
    counter = 0

    for world_index, world in enumerate(worlds):
        world_profiles: List[CharacterProfile] = []
        email_domain = world.email_domain or (
            world.organization_name.lower().replace(" ", "") + ".example"
        )
        for local_index in range(per_world):
            counter += 1
            department = world.departments[local_index % len(world.departments)]
            team = world.teams[local_index % len(world.teams)]

            gender = marginal_sampler.sample_gender()
            age_range = marginal_sampler.sample_age_range(gender)
            birth_year = marginal_sampler.sample_birth_year(age_range)
            country = marginal_sampler.sample_country()
            region, location = marginal_sampler.sample_location(country)
            nationality = marginal_sampler.sample_nationality()
            role, occupation_code = marginal_sampler.sample_occupation(department)
            degrees = marginal_sampler.sample_degrees()
            tenure_years = marginal_sampler.sample_tenure_years()
            identity = identity_sampler.sample_identity(counter, gender, birth_year, region)
            seniority = SENIORITIES[counter % len(SENIORITIES)]
            style_profile = style_sampler.sample(counter, seniority, nationality)

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
                sensitive_attributes.append(
                    SENSITIVE_VALUES[(local_index + world_index) % len(SENSITIVE_VALUES)]
                )

            world_profiles.append(
                CharacterProfile(
                    person_id=f"p_{counter:04d}",
                    full_name=identity.full_name,
                    email=f"{identity.username}@{email_domain}",
                    phone=identity.phone,
                    username=identity.username,
                    account_id=f"ACC-{world_index + 1:02d}-{counter:04d}",
                    language=world.language,
                    country=country,
                    location=location,
                    age_range=age_range,
                    gender=gender,
                    nationality=nationality,
                    organization_id=world.organization_id,
                    department=department,
                    team=team,
                    role=role,
                    seniority=seniority,
                    tenure_years=tenure_years,
                    degrees=degrees,
                    skills=skills,
                    certifications=certifications,
                    rare_traits=rare_traits,
                    events=rng.sample(world.incidents + world.calendar_events, k=2),
                    sensitive_attributes=sensitive_attributes,
                    style_profile=style_profile,
                    region=region,
                    birth_year=birth_year,
                    nir_like=identity.nir_like,
                    address=identity.address,
                    occupation_code=occupation_code,
                )
            )
        uniqueness_resample_count += _enforce_world_uniqueness(world_profiles, marginal_sampler)
        # Les indices contextuels sont tirés après le contrôle d'unicité pour
        # rester cohérents avec les attributs éventuellement resamplés.
        for profile in world_profiles:
            profile.contextual_cues = sample_contextual_cues(
                rng,
                cues_catalog,
                region=profile.region,
                role=profile.role,
                occupation_code=profile.occupation_code,
                age_range=profile.age_range,
                birth_year=profile.birth_year,
                nationality=profile.nationality,
                sensitive_attributes=profile.sensitive_attributes,
            )
        characters.extend(world_profiles)

    if stats is not None:
        stats["uniqueness_resample_count"] = uniqueness_resample_count
        stats["marginals_version"] = marginal_sampler.version  # type: ignore[assignment]
        stats["style_factors_version"] = style_sampler.version  # type: ignore[assignment]
        register_counts: Dict[str, int] = {}
        for character in characters:
            register = character.style_profile.register
            register_counts[register] = register_counts.get(register, 0) + 1
        stats["registers"] = register_counts  # type: ignore[assignment]
    return characters
