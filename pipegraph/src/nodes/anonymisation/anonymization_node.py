import sys
import os
import re
import logging
from typing import Dict, Any, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import yaml

from src.state import PipelineState
from src.config import settings

# Configuration du logger
logger = logging.getLogger("AnonymizationNode")

# Import du PseudoMapper nouveau
try:
    from src.utils.pseudo import PseudoMapper
except ImportError:
    logger.error("❌ Impossible d'importer PseudoMapper depuis src.utils.pseudo")
    PseudoMapper = None


# ---------------------------------------------------------------------------
# Stratégies d'anonymisation par type
# ---------------------------------------------------------------------------

def _mask_email(value: str) -> str:
    """jean.dupont@email.com → j*******@email.com"""
    if "@" not in value:
        return "[EMAIL]"
    local, domain = value.split("@", 1)
    masked_local = local[0] + "*" * max(1, len(local) - 1) if local else "*"
    return f"{masked_local}@{domain}"


def _mask_phone(value: str) -> str:
    """06 12 34 56 78 → 06 ** ** ** 78"""
    digits_only = re.sub(r"\D", "", value)
    if len(digits_only) < 6:
        return "[PHONE]"
    # Garde les 2 premiers et 2 derniers chiffres, masque le reste
    return digits_only[:2] + " ** ** ** " + digits_only[-2:]


def _mask_iban(value: str) -> str:
    """FR76 3000 6000 0112 3456 7890 189 → FR76 **** **** **** **** **** ***"""
    clean = value.replace(" ", "")
    if len(clean) < 4:
        return "[IBAN]"
    return clean[:4] + " " + " ".join("****" for _ in range((len(clean) - 4 + 3) // 4))


def _mask_generic(value: str, label: str) -> str:
    """Masque générique : garde les 2 premiers caractères."""
    if len(value) <= 3:
        return f"[{label}]"
    return value[:2] + "*" * (len(value) - 2)


def apply_strategy(
    entity_type: str,
    entity_text: str,
    strategy: str,
    mapper: Any,
    policy_overrides: Optional[Dict[str, str]] = None,
) -> str:
    """
    Applique la stratégie d'anonymisation pour une entité.

    Stratégies :
      - ``pseudo``     : pseudonyme cohérent via PseudoMapper
      - ``mask``       : masquage partiel adapté au type
      - ``generalize`` : remplacement par ``[TYPE]``
      - ``redact``     : remplacement complet par ``[TYPE_REDACTED]``

    L'ordre de priorité :
    1. policy_overrides (runtime, depuis state.config)
    2. strategy (global override depuis state.config)
    3. Valeur par défaut: "pseudo"
    """
    # Résolution de l'action effective
    if policy_overrides and entity_type in policy_overrides:
        action = policy_overrides[entity_type]
    else:
        action = strategy

    if action == "pseudo":
        if mapper:
            return mapper.placeholder(entity_type, entity_text)
        return f"[{entity_type}_PSEUDO]"

    if action == "mask":
        upper = entity_type.upper()
        if upper in ("EMAIL",):
            return _mask_email(entity_text)
        if upper in ("PHONE", "TELEPHONE", "MOBILE", "TEL"):
            return _mask_phone(entity_text)
        if upper in ("IBAN", "BANK_ACCOUNT"):
            return _mask_iban(entity_text)
        return _mask_generic(entity_text, entity_type)

    if action == "generalize":
        return f"[{entity_type.upper()}]"

    if action == "redact":
        return f"[{entity_type.upper()}_REDACTED]"

    # fallback
    if mapper:
        return mapper.placeholder(entity_type, entity_text)
    return f"[{entity_type}_MASKED]"


class AnonymizationNode:
    def __init__(self):
        # Utilisation des settings centralisés et sécurisés
        self.secret = settings.security.PSEUDO_SECRET.get_secret_value()
        settings.security.validate_secrets()

        self.default_scope = "default_scope"
        self._mappers: Dict[str, Any] = {}

        # --- Chargement de la politique d'anonymisation depuis pipeline_config.yaml ---
        self._yaml_policy: Dict[str, str] = {}
        self._yaml_global_strategy: str = "pseudo"
        _config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../config/pipeline_config.yaml")
        )
        try:
            if os.path.exists(_config_path):
                with open(_config_path, "r", encoding="utf-8") as f:
                    raw_yaml = yaml.safe_load(f) or {}
                anon_node = (
                    raw_yaml.get("pipeline", {})
                    .get("nodes", {})
                    .get("anonymization", {})
                )
                self._yaml_global_strategy = anon_node.get("strategy", "pseudo")
                raw_policy = anon_node.get("policy") or {}
                # Normaliser : {"PER": {"action": "pseudo"}} → {"PER": "pseudo"}
                for ent_type, cfg in raw_policy.items():
                    if isinstance(cfg, dict):
                        self._yaml_policy[ent_type.upper()] = cfg.get("action", "pseudo")
                    elif isinstance(cfg, str):
                        self._yaml_policy[ent_type.upper()] = cfg
                logger.info(
                    f"Politique d'anonymisation chargée: {len(self._yaml_policy)} règles, "
                    f"stratégie globale='{self._yaml_global_strategy}'"
                )
        except Exception as e:
            logger.warning(f"Impossible de charger la politique depuis le YAML: {e}")

    def _get_mapper(self, scope_id: str) -> Any:
        if not PseudoMapper:
            return None
        if scope_id not in self._mappers:
            self._mappers[scope_id] = PseudoMapper(secret=self.secret, scope_id=scope_id)
        return self._mappers[scope_id]

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        """
        Applique l'anonymisation sur le texte selon la politique configurée.

        Paramètres runtime supportés dans state.config :
          - ``enable_anonymization`` : bool — activer/désactiver le nœud
          - ``anon_strategy``        : str  — stratégie globale (pseudo|mask|generalize|redact)
          - ``anon_policy``          : Dict[str, str] — overrides par type (ex: {"PERSON":"pseudo", "LOC":"generalize"})
          - ``anon_clear_yaml_policy`` : bool — ignore la politique YAML et n'utilise que les overrides runtime
          - ``scope_id``             : str  — scope du pseudonyme (pour cohérence inter-documents)
        """
        logger.info("--- Node: Anonymization ---")
        state_config = state.get("config", {})

        if not state_config.get("enable_anonymization", True):
            logger.info("Anonymization Node désactivé.")
            return {}

        # Il faut TOUJOURS partir du texte original car les offsets des entités 
        # (start, end) sont calibrés sur l'original_text. 
        # Si on faisait l'anonymisation sur un texte déjà modifié, cela provoquerait des erreurs de type "out of bounds".
        base_text = state.get("original_text") or state["text"]
        
        entities = state.get("entities", [])

        if not entities:
            logger.debug("Aucune entité à anonymiser.")
            return {"text": base_text}

        # Résolution de la stratégie effective (runtime > YAML > défaut)
        global_strategy: str = state_config.get("anon_strategy", self._yaml_global_strategy)

        # Politique par type : fusion YAML + overrides runtime.
        # ``anon_clear_yaml_policy`` permet de forcer une stratégie globale pure
        # sans héritage de la politique mixte définie dans le YAML.
        runtime_policy: Dict[str, str] = {
            k.upper(): v
            for k, v in (state_config.get("anon_policy") or {}).items()
        }
        if state_config.get("anon_clear_yaml_policy", False):
            effective_policy = dict(runtime_policy)
        else:
            effective_policy = {**self._yaml_policy, **runtime_policy}

        # Scope_id pour la cohérence des pseudonymes
        raw_scope = (
            state.get("metadata", {}).get("scope_id")
            or state_config.get("scope_id", self.default_scope)
        )
        scope_id = str(raw_scope) if raw_scope is not None else self.default_scope
        mapper = self._get_mapper(scope_id)

        # Tri de la fin vers le début pour ne pas casser les offsets
        sorted_entities = sorted(entities, key=lambda x: x.get("start", 0), reverse=True)

        new_text = base_text
        replacements_count = 0

        for ent in sorted_entities:
            start = ent.get("start")
            end = ent.get("end")

            if start is None or end is None or start < 0 or end > len(base_text) or start >= end:
                logger.warning(f"Entité malformée ou hors limites ignorée: {ent}")
                continue

            # Support du double format de clés (type/entity_type, value/text)
            entity_type = str(ent.get("type") or ent.get("entity_type") or "")
            entity_text = str(ent.get("value") or ent.get("text") or "")

            if not entity_type or not entity_text:
                logger.warning(f"Entité sans type ou valeur ignorée: {ent}")
                continue

            placeholder = apply_strategy(
                entity_type=entity_type.upper(),
                entity_text=entity_text,
                strategy=global_strategy,
                mapper=mapper,
                policy_overrides=effective_policy if effective_policy else None,
            )

            new_text = new_text[:start] + placeholder + new_text[end:]
            replacements_count += 1

        logger.info(f"Anonymisation terminée: {replacements_count} remplacements effectués.")
        return {"text": new_text}
