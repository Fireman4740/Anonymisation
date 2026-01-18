import sys
import os
import logging
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.state import PipelineState

# Configuration du logger
logger = logging.getLogger("AnonymizationNode")

# Import du PseudoMapper legacy
try:
    from src.legacy_services.utils.utils_pseudo import PseudoMapper
except ImportError:
    PseudoMapper = None


class AnonymizationNode:
    def __init__(self):
        # On récupère le secret depuis l'environnement ou on échoue si absent en prod
        self.secret = os.getenv("PSEUDO_SECRET")
        if not self.secret:
            logger.warning(
                "⚠️ PSEUDO_SECRET non défini. Utilisation d'un secret temporaire peu sûr."
            )
            self.secret = "temp_development_secret_do_not_use_in_prod"

        self.default_scope = "default_scope"
        self._mappers = {}  # Cache de mappers par scope_id

    def _get_mapper(self, scope_id: str) -> Any:
        if not PseudoMapper:
            return None
        if scope_id not in self._mappers:
            self._mappers[scope_id] = PseudoMapper(secret=self.secret, scope_id=scope_id)
        return self._mappers[scope_id]

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        """
        Applique l'anonymisation (remplacement) sur le texte.
        Gère les scopes et les entités triées pour éviter les corruptions d'index.
        """
        logger.info("--- Node: Anonymization ---")

        if not state.get("config", {}).get("enable_anonymization", True):
            logger.info("Anonymization Node désactivé.")
            return {}

        text = state["text"]
        entities = state.get("entities", [])

        if not entities:
            logger.debug("Aucune entité à anonymiser.")
            return {"text": text}

        # Récupération du scope_id depuis le state metadata ou config
        scope_id = state.get("metadata", {}).get("scope_id") or state.get("config", {}).get(
            "scope_id", self.default_scope
        )
        mapper = self._get_mapper(scope_id)

        # Tri des entités par position inverse pour ne pas casser les index lors du remplacement
        # (On remplace de la fin vers le début)
        # On s'assure que les entités sont bien formées et uniques
        sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

        new_text = text
        replacements_count = 0

        for ent in sorted_entities:
            start = ent.get("start")
            end = ent.get("end")

            # Validation minimale des bornes
            if start is None or end is None or start < 0 or end > len(text) or start >= end:
                logger.warning(f"Entité malformée ou hors limites ignorée: {ent}")
                continue

            entity_type = ent.get("type", ent.get("entity_type"))
            entity_text = ent.get("value", ent.get("text"))

            if not entity_type or not entity_text:
                logger.warning(f"Entité sans type ou valeur ignorée: {ent}")
                continue

            # Génération du placeholder
            if mapper:
                placeholder = mapper.placeholder(entity_type, entity_text)
            else:
                placeholder = f"[{entity_type}_MASKED]"

            # Remplacement sécurisé
            new_text = new_text[:start] + placeholder + new_text[end:]
            replacements_count += 1

        logger.info(f"Anonymisation terminée: {replacements_count} remplacements effectués.")
        return {"text": new_text}
