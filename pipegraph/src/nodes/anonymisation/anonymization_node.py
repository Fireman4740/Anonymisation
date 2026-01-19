import sys
import os
import logging
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.state import PipelineState
from src.config import settings

# Configuration du logger
logger = logging.getLogger("AnonymizationNode")

# Import du PseudoMapper nouveau
try:
    from src.utils.pseudo import PseudoMapper
except ImportError:
    # Fallback critique : on ne devrait pas arriver ici avec la nouvelle structure
    logger.error("❌ Impossible d'importer PseudoMapper depuis src.utils.pseudo")
    PseudoMapper = None


class AnonymizationNode:
    def __init__(self):
        # Utilisation des settings centralisés et sécurisés
        self.secret = settings.security.PSEUDO_SECRET.get_secret_value()

        # Validation explicite en production
        settings.security.validate_secrets()

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
        # Typage explicite pour satisfaire le linter si nécessaire
        raw_scope = state.get("metadata", {}).get("scope_id") or state.get("config", {}).get(
            "scope_id", self.default_scope
        )
        scope_id = str(raw_scope) if raw_scope is not None else self.default_scope
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
