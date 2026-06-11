from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from atlas_anno.generation.style_sampler import load_style_factors
from atlas_anno.schemas import DocumentRecord

# Patterns conservateurs : « vous » seul est exclu (ambigu pluriel) ; on ne
# flague que des marqueurs univoques de tutoiement/vouvoiement.
_TU_MARKERS = [
    re.compile(r"\b(tu|ton|ta|tes)\b", re.IGNORECASE),
    re.compile(r"\bpeux-tu\b", re.IGNORECASE),
    re.compile(r"\bdis-moi\b", re.IGNORECASE),
]
_VOUS_MARKERS = [
    re.compile(r"\bpouvez-vous\b", re.IGNORECASE),
    re.compile(r"\bveuillez\b", re.IGNORECASE),
    re.compile(r"\bdites-moi\b", re.IGNORECASE),
]


@lru_cache(maxsize=1)
def _register_taboos() -> Dict[str, List[str]]:
    return load_style_factors().get("register_taboos", {}) or {}


def check_document(document: DocumentRecord) -> Dict[str, Any]:
    """Conformité tu/vous + marqueurs tabous du registre. Catalogue partagé avec
    la génération (style_factors_fr.yaml) pour éviter toute dérive."""
    flags: List[str] = []
    text = document.text
    address_form = document.scenario.address_form or str(document.metadata.get("address_form", ""))
    if address_form == "vous":
        if any(pattern.search(text) for pattern in _TU_MARKERS):
            flags.append("address_form:tutoiement_dans_document_vous")
    elif address_form == "tu":
        if any(pattern.search(text) for pattern in _VOUS_MARKERS):
            flags.append("address_form:vouvoiement_dans_document_tu")

    register = document.scenario.register or str(document.metadata.get("register", ""))
    lowered = text.lower()
    for taboo in _register_taboos().get(register, []):
        if taboo.lower() in lowered:
            flags.append(f"register_taboo:{taboo}")
    return {"passed": not flags, "flags": flags}


def apply_register_conformance(documents: Sequence[DocumentRecord]) -> Dict[str, Any]:
    """Stocke le résultat par document (metadata) et route les échecs vers la
    revue humaine. N'alimente PAS audit_document : un écart de registre est un
    signal de qualité, pas une erreur bloquante de génération."""
    checked = 0
    failed = 0
    for document in documents:
        register = document.scenario.register or document.metadata.get("register")
        if not register:
            continue
        checked += 1
        result = check_document(document)
        document.metadata["register_conformance"] = result
        if not result["passed"]:
            failed += 1
            document.metadata["human_review_required"] = True
            reasons = document.metadata.setdefault("review_reasons", [])
            if "register_conformance" not in reasons:
                reasons.append("register_conformance")
    return {
        "checked": checked,
        "failed": failed,
        "rate": round(failed / checked, 4) if checked else 0.0,
    }
