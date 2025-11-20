
"""
Policy de Configuration pour l'Anonymisation

Définit les niveaux d'anonymisation (L0, L1, L2) et les paramètres configurables.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class AnonymizationPolicy:
    """
    Politique d'anonymisation définissant les stratégies de détection et transformation.
    
    Attributs:
        level: Niveau d'anonymisation ("L0", "L1", "L2")
        placeholder_style: Style des placeholders ("typed", "generic")
        date_granularity: Granularité des dates ("none", "week", "month", "quarter", "year")
        org_policy: Politique pour les organisations ("keep", "generalize", "redact")
        ip_policy: Politique pour les IPs ("keep", "cidr24", "redact")
        
        # LLM Features (L1 uniquement)
        llm_detection: Activer détection LLM avancée
        llm_paraphrase: Activer paraphrase stylométrique
        llm_audit: Activer audit de risque
        paraphrase_intensity: Intensité de paraphrase (0-3)
        risk_threshold: Seuil de risque pour hardening (0-100)
        max_hardening_rounds: Nombre max de tours de durcissement
        
        # RUPTA Features (L1 uniquement)
        rupta_enabled: Activer optimisation RUPTA
        rupta_max_iterations: Nombre max d'itérations RUPTA
        rupta_p_threshold: Seuil p-value pour privacy
        rupta_privacy_threshold: Seuil privacy rank minimum
        rupta_utility_threshold: Seuil utility score minimum
        
        # Autres
        mapping_retention: Conserver ou supprimer mappings ("keep", "discard")
    """
    
    # Niveau et style
    level: str = "L0"
    placeholder_style: str = "typed"  # "typed" ou "generic"
    
    # Policies de généralisation
    date_granularity: str = "none"  # "none", "week", "month", "quarter", "year"
    org_policy: str = "keep"  # "keep", "generalize", "redact"
    ip_policy: str = "keep"  # "keep", "cidr24", "redact"
    
    # LLM features (L1 uniquement)
    llm_detection: bool = False
    llm_paraphrase: bool = False
    llm_audit: bool = False
    paraphrase_intensity: int = 1  # 0-3
    risk_threshold: int = 60  # 0-100
    max_hardening_rounds: int = 2
    
    # RUPTA features (L1 uniquement)
    rupta_enabled: bool = False
    rupta_max_iterations: int = 3
    rupta_p_threshold: float = 0.05
    rupta_privacy_threshold: Optional[int] = None
    rupta_utility_threshold: float = 0.7
    
    # Autres
    mapping_retention: str = "keep"  # "keep", "discard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la policy en dictionnaire."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnonymizationPolicy":
        """Crée une policy depuis un dictionnaire."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


def preset(level: str) -> AnonymizationPolicy:
    """
    Retourne une policy pré-configurée selon le niveau.
    
    Args:
        level: "L0" (regex+NER uniquement), "L1" (+ LLM + RUPTA), "L2" (L1 + généralisation agressive)
    
    Returns:
        AnonymizationPolicy configurée
    """
    level = level.upper()
    
    if level == "L0":
        # Mode de base : regex + NER, pas de LLM
        return AnonymizationPolicy(
            level="L0",
            placeholder_style="typed",
            date_granularity="none",
            org_policy="keep",
            ip_policy="keep",
            llm_detection=False,
            llm_paraphrase=False,
            llm_audit=False,
            rupta_enabled=False,
            mapping_retention="keep",
        )
    
    elif level == "L1":
        # Mode avancé : L0 + LLM detection + paraphrase + audit + RUPTA
        return AnonymizationPolicy(
            level="L1",
            placeholder_style="typed",
            date_granularity="month",
            org_policy="generalize",
            ip_policy="cidr24",
            llm_detection=True,
            llm_paraphrase=True,
            llm_audit=True,
            paraphrase_intensity=2,
            risk_threshold=60,
            max_hardening_rounds=2,
            rupta_enabled=True,
            rupta_max_iterations=3,
            rupta_p_threshold=0.05,
            rupta_privacy_threshold=None,
            rupta_utility_threshold=0.7,
            mapping_retention="discard",
        )
    
    elif level == "L2":
        # Mode maximal : L1 avec généralisation très agressive
        return AnonymizationPolicy(
            level="L2",
            placeholder_style="generic",
            date_granularity="year",
            org_policy="redact",
            ip_policy="redact",
            llm_detection=True,
            llm_paraphrase=True,
            llm_audit=True,
            paraphrase_intensity=3,
            risk_threshold=40,
            max_hardening_rounds=3,
            rupta_enabled=True,
            rupta_max_iterations=5,
            rupta_p_threshold=0.01,
            rupta_privacy_threshold=None,
            rupta_utility_threshold=0.6,
            mapping_retention="discard",
        )
    
    else:
        raise ValueError(f"Niveau de policy inconnu: {level}. Utilisez L0, L1 ou L2.")


__all__ = ["AnonymizationPolicy", "preset"]
