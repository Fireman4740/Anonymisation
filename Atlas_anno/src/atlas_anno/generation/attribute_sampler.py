from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from faker import Faker

from atlas_anno.io import load_yaml, project_file
from atlas_anno.schemas import ContextualCue

DISTRIBUTION_INSEE_PATH = "configs/distributions/fr_insee_marginals.yaml"
DISTRIBUTION_PHONE_PATH = "configs/distributions/phone_prefixes_fr.yaml"
DISTRIBUTION_CUES_PATH = "configs/distributions/contextual_cues_fr.yaml"

REFERENCE_YEAR = 2026


def load_insee_marginals() -> Dict[str, Any]:
    return load_yaml(project_file(DISTRIBUTION_INSEE_PATH))


def load_phone_prefixes() -> Dict[str, Any]:
    return load_yaml(project_file(DISTRIBUTION_PHONE_PATH))


def load_contextual_cues_catalog() -> Dict[str, Any]:
    return load_yaml(project_file(DISTRIBUTION_CUES_PATH))


def sample_contextual_cues(
    rng: random.Random,
    catalog: Dict[str, Any],
    *,
    region: str,
    role: str,
    occupation_code: str,
    age_range: str,
    birth_year: int,
    nationality: str,
    sensitive_attributes: List[str],
    min_cues: int = 2,
    max_cues: int = 4,
) -> List[ContextualCue]:
    """Tire 2–4 indices contextuels cohérents avec le profil, chacun déclarant
    l'attribut qu'il fuite (reveals_label/reveals_value)."""
    candidates: List[ContextualCue] = []

    regional = catalog.get("regional_references", {}).get(region)
    if regional:
        candidates.append(
            ContextualCue(
                cue_type="regional_reference",
                cue_text=rng.choice(regional),
                reveals_label="LOCATION",
                reveals_value=region,
            )
        )

    jargon_catalog = catalog.get("metier_jargon", {})
    jargon = jargon_catalog.get(occupation_code) or jargon_catalog.get("default")
    if jargon:
        candidates.append(
            ContextualCue(
                cue_type="metier_jargon",
                cue_text=rng.choice(jargon),
                reveals_label="ROLE",
                reveals_value=role,
            )
        )

    era_templates = catalog.get("education_era", [])
    if era_templates and birth_year:
        template = rng.choice(era_templates)
        cue_text = template.format(
            bac_year=birth_year + 18, career_start_year=birth_year + 23
        )
        candidates.append(
            ContextualCue(
                cue_type="education_era",
                cue_text=cue_text,
                reveals_label="AGE_RANGE",
                reveals_value=age_range,
            )
        )

    if "FAMILY_STATUS" in sensitive_attributes:
        habits = catalog.get("schedule_habits", [])
        if habits:
            candidates.append(
                ContextualCue(
                    cue_type="schedule_habit",
                    cue_text=rng.choice(habits),
                    reveals_label="FAMILY_STATUS",
                    reveals_value="FAMILY_STATUS",
                )
            )

    dialect = catalog.get("dialect_markers", {}).get(nationality)
    if dialect:
        candidates.append(
            ContextualCue(
                cue_type="dialect_marker",
                cue_text=rng.choice(dialect),
                reveals_label="NATIONALITY",
                reveals_value=nationality,
            )
        )

    if len(candidates) <= min_cues:
        return candidates
    target = rng.randint(min_cues, min(max_cues, len(candidates)))
    indexes = sorted(rng.sample(range(len(candidates)), k=target))
    return [candidates[index] for index in indexes]


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(char for char in ascii_value.lower() if char.isalnum() or char == "-")


def _weighted_choice(rng: random.Random, mapping: Dict[str, float]) -> str:
    keys = list(mapping.keys())
    weights = [float(mapping[key]) for key in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


@dataclass(frozen=True)
class Identity:
    first_name: str
    last_name: str
    full_name: str
    username: str
    phone: str
    nir_like: str
    address: str


class MarginalSampler:
    """Tirages conditionnels de quasi-identifiants depuis les marginales INSEE-like.

    Tous les tirages consomment le `rng` fourni : le déterminisme est garanti
    par l'appelant (random.Random(seed) partagé du character_builder).
    """

    def __init__(self, rng: random.Random, marginals: Dict[str, Any]) -> None:
        self._rng = rng
        self._marginals = marginals

    @property
    def version(self) -> str:
        return str(self._marginals.get("version", "unknown"))

    def sample_gender(self) -> str:
        return "female" if self._rng.random() < 0.5 else "male"

    def sample_age_range(self, gender: str) -> str:
        by_sex = self._marginals["age_by_sex"]
        distribution = by_sex.get(gender) or next(iter(by_sex.values()))
        return _weighted_choice(self._rng, distribution)

    def sample_birth_year(self, age_range: str) -> int:
        low, high = (int(part) for part in age_range.split("-"))
        age = self._rng.randint(low, high)
        return REFERENCE_YEAR - age

    def sample_country(self) -> str:
        return _weighted_choice(self._rng, self._marginals["countries"])

    def sample_nationality(self) -> str:
        return _weighted_choice(self._rng, self._marginals["nationalities"])

    def sample_location(self, country: str) -> Tuple[str, str]:
        """Retourne (région, ville). Région vide hors France."""
        if country == "France":
            regions = self._marginals["regions"]
            region = _weighted_choice(
                self._rng, {name: spec["weight"] for name, spec in regions.items()}
            )
            city = _weighted_choice(self._rng, regions[region]["cities"])
            return region, city
        foreign = self._marginals["foreign_cities"].get(country)
        if not foreign:
            return "", "Paris"
        return "", _weighted_choice(self._rng, foreign)

    def sample_occupation(self, department: str) -> Tuple[str, str]:
        """Retourne (rôle, code PCS-like) conditionné au département."""
        occupations = self._marginals["occupations"]
        entries = occupations.get(department) or occupations["default"]
        weights = [float(entry["weight"]) for entry in entries]
        entry = self._rng.choices(entries, weights=weights, k=1)[0]
        return str(entry["role"]), str(entry["code"])

    def sample_degrees(self) -> List[str]:
        entries = self._marginals["degrees"]
        weights = [float(entry["weight"]) for entry in entries]
        entry = self._rng.choices(entries, weights=weights, k=1)[0]
        return list(entry["values"])

    def sample_tenure_years(self) -> int:
        return int(_weighted_choice(self._rng, self._marginals["tenure_years"]))


class IdentitySampler:
    """Identifiants directs réalistes via Faker fr_FR, déterministes par index.

    Le seed est dérivé par personnage (`base_seed * 10_000 + index`) : insérer
    ou retirer un personnage ne décale pas les identités des autres.
    """

    def __init__(self, seed: int, locale: str = "fr_FR", phone_prefixes: Dict[str, Any] | None = None) -> None:
        self._base_seed = seed
        self._faker = Faker(locale)
        self._phone_prefixes = phone_prefixes or load_phone_prefixes()

    def _derived_seed(self, index: int) -> int:
        return self._base_seed * 10_000 + index

    def sample_identity(self, index: int, gender: str, birth_year: int, region: str) -> Identity:
        derived = self._derived_seed(index)
        self._faker.seed_instance(derived)
        rng = random.Random(derived)

        if gender == "female":
            first_name = self._faker.first_name_female()
        else:
            first_name = self._faker.first_name_male()
        last_name = self._faker.last_name()
        full_name = f"{first_name} {last_name}"
        username = f"{_slugify(first_name)}.{_slugify(last_name)}"

        phone = self._sample_phone(rng, region)
        nir_like = self._sample_nir_like(rng, gender, birth_year)
        address = self._faker.street_address()
        return Identity(
            first_name=first_name,
            last_name=last_name,
            full_name=full_name,
            username=username,
            phone=phone,
            nir_like=nir_like,
            address=address,
        )

    def _sample_phone(self, rng: random.Random, region: str) -> str:
        config = self._phone_prefixes
        if rng.random() < float(config.get("mobile_share", 0.8)):
            prefix = rng.choice(config["mobile_prefixes"])
        else:
            by_region = config["landline_by_region"]
            prefix = by_region.get(region, by_region.get("default", "01"))
        groups = [f"{rng.randint(0, 99):02d}" for _ in range(4)]
        return " ".join([prefix] + groups)

    def _sample_nir_like(self, rng: random.Random, gender: str, birth_year: int) -> str:
        """NIR de forme valide mais clé volontairement fausse (donnée fictive)."""
        sex_digit = 2 if gender == "female" else 1
        year = birth_year % 100
        month = rng.randint(1, 12)
        dept = rng.randint(1, 95)
        commune = rng.randint(1, 990)
        order = rng.randint(1, 999)
        body = int(f"{sex_digit}{year:02d}{month:02d}{dept:02d}{commune:03d}{order:03d}")
        valid_key = 97 - (body % 97)
        invalid_key = valid_key % 97 + 1
        return f"{sex_digit} {year:02d} {month:02d} {dept:02d} {commune:03d} {order:03d} {invalid_key:02d}"
