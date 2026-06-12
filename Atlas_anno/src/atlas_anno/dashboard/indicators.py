"""Référentiel unique des indicateurs du tableau de bord.

Centralise le libellé, le format d'affichage, le sens de lecture (plus haut /
plus bas = mieux) et le texte d'aide de chaque métrique. L'app Streamlit s'en
sert pour les info-bulles, les KPI et le guide de lecture, afin que la
sémantique d'un indicateur ne soit définie qu'à un seul endroit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    fmt: str  # "pct" | "num"
    higher_is_better: Optional[bool]  # None => purement descriptif
    help: str
    reading: str  # résumé court "sens de lecture" pour le guide

    @property
    def direction_icon(self) -> str:
        if self.higher_is_better is True:
            return "↑"
        if self.higher_is_better is False:
            return "↓"
        return "—"


# Ordre = ordre d'affichage dans le guide de lecture.
INDICATORS: Dict[str, Indicator] = {
    "privacy_score": Indicator(
        key="privacy_score",
        label="Confidentialité",
        fmt="pct",
        higher_is_better=True,
        help=(
            "Part de l'information identifiante neutralisée par l'anonymisation. "
            "Plus haut = mieux : ~100 % indique une protection forte."
        ),
        reading="↑ plus haut = mieux",
    ),
    "utility_score": Indicator(
        key="utility_score",
        label="Utilité",
        fmt="pct",
        higher_is_better=True,
        help=(
            "Utilité / lisibilité du texte préservée après anonymisation (contenu "
            "non identifiant conservé). Plus haut = mieux. Un bon système maximise "
            "confidentialité ET utilité."
        ),
        reading="↑ plus haut = mieux",
    ),
    "reid_top1": Indicator(
        key="reid_top1",
        label="ReID top-1",
        fmt="pct",
        higher_is_better=False,
        help=(
            "Taux de ré-identification réussie au 1er rang par l'attaquant. "
            "C'est un RISQUE : plus bas = mieux. 0 % = aucun document ré-identifié."
        ),
        reading="↓ plus bas = mieux (risque)",
    ),
    "span_f1": Indicator(
        key="span_f1",
        label="Span F1",
        fmt="pct",
        higher_is_better=True,
        help=(
            "F1 de détection des entités à anonymiser (vs annotations de référence). "
            "Combine précision et rappel des spans PII. Plus haut = mieux."
        ),
        reading="↑ plus haut = mieux",
    ),
    "self_bleu": Indicator(
        key="self_bleu",
        label="self-BLEU",
        fmt="num",
        higher_is_better=False,
        help=(
            "Similarité moyenne entre documents (chevauchement de n-grammes). "
            "Plus bas = plus divers. Alerte « collapse » de diversité si > 0.90."
        ),
        reading="↓ plus bas = plus divers (alerte > 0.90)",
    ),
    "distinct_2": Indicator(
        key="distinct_2",
        label="distinct-2",
        fmt="num",
        higher_is_better=True,
        help=(
            "Proportion de bigrammes uniques dans le corpus. Plus haut = plus "
            "divers lexicalement. Diversité faible si < 0.15."
        ),
        reading="↑ plus haut = plus divers (faible < 0.15)",
    ),
    "duplicate_rate": Indicator(
        key="duplicate_rate",
        label="duplicate_rate",
        fmt="num",
        higher_is_better=False,
        help=(
            "Part de documents quasi-dupliqués (MinHash, Jaccard ≥ 0.80). "
            "Plus bas = mieux. Seuil d'échec : > 2 %."
        ),
        reading="↓ plus bas = mieux (échec > 2 %)",
    ),
    "cell_coverage": Indicator(
        key="cell_coverage",
        label="cell_coverage",
        fmt="num",
        higher_is_better=True,
        help=(
            "Part des cellules factorielles (domaine × difficulté × registre × "
            "objectif) effectivement remplies. Plus haut = mieux. Faible si < 0.60."
        ),
        reading="↑ plus haut = mieux (faible < 0.60)",
    ),
    "cell_entropy": Indicator(
        key="cell_entropy",
        label="cell_entropy",
        fmt="num",
        higher_is_better=True,
        help=(
            "Équilibre de la répartition entre cellules factorielles (entropie "
            "normalisée 0–1). Plus haut = répartition plus équilibrée."
        ),
        reading="↑ plus haut = plus équilibré",
    ),
}


def format_value(indicator: Indicator, value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if indicator.fmt == "pct":
        return f"{number:.1%}"
    return f"{number:.4f}"


def guide_rows() -> list[Dict[str, str]]:
    """Lignes du tableau « guide de lecture » affiché dans l'interface."""
    return [
        {
            "Indicateur": indicator.label,
            "Signification": indicator.help.split(". ")[0] + ".",
            "Sens de lecture": indicator.reading,
        }
        for indicator in INDICATORS.values()
    ]
