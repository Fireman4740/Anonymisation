from __future__ import annotations

import random
from typing import Any, Dict

from atlas_anno.io import load_yaml, project_file
from atlas_anno.schemas import StyleProfile

DISTRIBUTION_STYLE_PATH = "configs/distributions/style_factors_fr.yaml"

# Champs v1 conservés pour le code aval (connecteurs, signature, verbosité).
# Connecteurs conditionnés au registre pour rester cohérents avec les
# register_taboos du catalogue (« du coup » interdit en soutenu, etc.).
_CONNECTOR_SETS_BY_REGISTER = {
    "familier": [
        ["du coup", "au passage"],
        ["bref", "franchement"],
        ["donc", "par contre"],
    ],
    "courant": [
        ["donc", "par contre"],
        ["en revanche", "merci d'avance"],
        ["dans ce contexte", "à noter"],
        ["concrètement", "pour info"],
    ],
    "soutenu": [
        ["en revanche", "par conséquent"],
        ["dans ce contexte", "en outre"],
        ["par ailleurs", "à toutes fins utiles"],
    ],
}
_SIGNATURE_PATTERNS = ["thanks_name", "full_signature", "thanks_name"]
_VERBOSITIES = ["short", "medium", "long"]
_VERBOSITY_WEIGHTS = [0.40, 0.45, 0.15]


def load_style_factors() -> Dict[str, Any]:
    return load_yaml(project_file(DISTRIBUTION_STYLE_PATH))


def _weighted_choice(rng: random.Random, mapping: Dict[str, float]) -> str:
    keys = list(mapping.keys())
    weights = [float(mapping[key]) for key in keys]
    if not any(weights):
        return keys[0]
    return rng.choices(keys, weights=weights, k=1)[0]


class StyleFactorSampler:
    """Tire les facteurs de diversité stylistique par personnage.

    Seed dérivé par personnage (`(base_seed + 1) * 20_011 + counter`) sur le
    modèle d'IdentitySampler : insérer ou retirer un personnage ne décale pas
    le style des autres.
    """

    def __init__(self, seed: int, factors: Dict[str, Any] | None = None) -> None:
        self._base_seed = seed
        self._factors = factors or load_style_factors()

    @property
    def version(self) -> str:
        return str(self._factors.get("version", "unknown"))

    def sample(self, counter: int, seniority: str, nationality: str) -> StyleProfile:
        rng = random.Random((self._base_seed + 1) * 20_011 + counter)
        factors = self._factors

        register = _weighted_choice(rng, factors["registers"])
        address_form = _weighted_choice(rng, factors["address_form_by_register"][register])
        expertise_table = factors["expertise_by_seniority"]
        expertise_level = _weighted_choice(
            rng, expertise_table.get(seniority) or expertise_table["default"]
        )
        variety_table = factors["variety_by_nationality"]
        francophone_variety = variety_table.get(nationality, variety_table.get("default", "metropole"))
        typo_propensity = _weighted_choice(rng, factors["typo_by_register"][register])

        formality = {"familier": "low", "courant": "medium", "soutenu": "high"}[register]
        return StyleProfile(
            formality=formality,
            signature_pattern=rng.choice(_SIGNATURE_PATTERNS),
            verbosity=rng.choices(_VERBOSITIES, weights=_VERBOSITY_WEIGHTS, k=1)[0],
            emoji_usage="rare" if register == "familier" and rng.random() < 0.4 else "none",
            favorite_connectors=rng.choice(_CONNECTOR_SETS_BY_REGISTER[register]),
            jargon_pattern="support_ops" if counter % 2 else "ai_ops",
            register=register,
            address_form=address_form,
            expertise_level=expertise_level,
            francophone_variety=francophone_variety,
            typo_propensity=typo_propensity,
        )
