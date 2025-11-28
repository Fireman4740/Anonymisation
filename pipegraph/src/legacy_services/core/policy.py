"""Définition de la politique d'anonymisation (version simplifiée).

Objectif demandé: n'avoir qu'un choix binaire "utilisation LLM ou non".

Implémentation:
    - level "L0"  => aucun usage de LLM (détection / paraphrase / audit désactivés)
    - tout autre level (L1, L2, L3, L4, on, yes, true, llm, etc.) => configuration LLM unique (level normalisé à "L1").

Les anciens niveaux (L2..L4) sont mappés sur L1 pour conserver la compatibilité
avec le code existant qui peut encore envoyer ces valeurs.

Les champs supplémentaires (granularités, etc.) sont conservés pour ne pas casser
le reste du pipeline mais ne varient plus entre plusieurs niveaux avancés.
"""
from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any

Gran = Literal["none", "day", "week", "month", "quarter", "year", "redact"]
LocGran = Literal["address", "city", "region", "country", "redact"]
IPGran = Literal["exact", "public_private", "cidr24", "redact"]
PlaceholderStyle = Literal["typed", "generic", "redacted"]


@dataclass
class AnonymizationPolicy:
    level: Literal["L0", "L1"]  # on conserve les labels historiques
    placeholder_style: PlaceholderStyle = "typed"
    scope: Literal["ticket", "batch", "none"] = "ticket"
    mapping_retention: Literal["session", "discard"] = "session"

    # Les granularités ne changent plus selon un niveau avancé; laissées neutres
    date_granularity: Gran = "none"
    time_granularity: Gran = "none"
    location_granularity: LocGran = "city"
    ip_policy: IPGran = "public_private"
    amount_binning: Literal["none", "round2", "range10", "range100", "redact"] = "none"
    org_policy: Literal["replace", "categorize", "generalize", "redact"] = "categorize"

    # Composants LLM
    llm_detection: bool = False
    llm_paraphrase: bool = False
    llm_audit: bool = False

    # RUPTA - Privacy-Utility optimization
    rupta_enabled: bool = False
    rupta_p_threshold: int = 10
    rupta_max_iterations: int = 3
    rupta_privacy_threshold: int | None = None
    rupta_utility_threshold: int = 80

    # Seuil de risque (si audit actif) et tours de durcissement (désactivés ici)
    risk_threshold: int = 100
    max_hardening_rounds: int = 0

    paraphrase_intensity: int = 0
    paraphrase_preserve_multiplicity: bool = False
    rupta_preserve_entity_counts: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def preset(level: str) -> AnonymizationPolicy:
    """Retourne une politique binaire.

    Paramètres acceptés pour "sans LLM": "L0", "0", "NO_LLM", "OFF", "FALSE".
    Tout autre valeur => mode LLM unique normalisé à "L1".
    """
    norm = (level or "").strip().upper()
    no_llm_aliases = {"L0", "0", "NO_LLM", "OFF", "FALSE"}
    if norm in no_llm_aliases:
        # L0: toutes les anonymisations "classiques" (regex, NER, généralisations déterministes)
        # MAIS aucun composant LLM (détection raisonnée, paraphrase, audit).
        # On applique donc la même granularité de dates qu'en mode LLM par défaut (month)
        # pour offrir un bénéfice de privacy sans altérations stylométriques LLM.
        return AnonymizationPolicy(
            level="L0",
            placeholder_style="typed",
            scope="ticket",
            mapping_retention="session",
            date_granularity="month",  # active la généralisation déterministe
            time_granularity="none",
            location_granularity="city",
            ip_policy="public_private",
            amount_binning="none",
            org_policy="categorize",  # cohérent avec L1 mais sans LLM
            llm_detection=False,
            llm_paraphrase=False,
            llm_audit=False,
            risk_threshold=100,
            max_hardening_rounds=0,
            paraphrase_intensity=0,  # pas de paraphrase (LLM off)
            paraphrase_preserve_multiplicity=False,
            rupta_preserve_entity_counts=False,
        )
    # Mode LLM (unique)
    return AnonymizationPolicy(
        level="L1",
        placeholder_style="typed",
        scope="ticket",
        mapping_retention="session",
        date_granularity="month",
        time_granularity="none",
        location_granularity="city",
        ip_policy="public_private",
        amount_binning="none",
        org_policy="categorize",
        llm_detection=True,   # seule fonctionnalité LLM active
        llm_paraphrase=True,
        llm_audit=True,
        # RUPTA activation for privacy-utility optimization
        rupta_enabled=True,
        rupta_p_threshold=10,  # Nombre de candidats pour privacy evaluation
        rupta_max_iterations=3,  # Optimisation rapide (3 itérations max)
        rupta_privacy_threshold=None,  # Pas de seuil strict (cherche non-identifié)
        rupta_utility_threshold=80,  # Maintenir au moins 80% d'utilité
        paraphrase_intensity=1,
        paraphrase_preserve_multiplicity=True,
        rupta_preserve_entity_counts=True,
        risk_threshold=45,
        max_hardening_rounds=0,
    )