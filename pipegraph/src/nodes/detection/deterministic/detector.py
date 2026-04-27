import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path

from .validators import Validators
from .utils import is_whitelisted

# Configuration du logger
logger = logging.getLogger("DeterministicDetector")

_TITLE_TOKEN = r"(?:[A-Z][A-Za-z]+(?:[-'][A-Za-z]+)*|[A-Z]{2,}(?:[.-][A-Z]{1,})*)"
_CONNECTOR_TOKEN = r"(?:of|and|the|for|de|del|della|dello|des|du|da|van|von|la|le|los|las)"
_ENTITY_TOKEN = rf"(?:{_TITLE_TOKEN}|{_CONNECTOR_TOKEN})"
_ENTITY_SPAN = rf"{_TITLE_TOKEN}(?:\s+{_ENTITY_TOKEN}){{0,5}}"

_FIXTURE_RE = re.compile(
    rf"(?m)(?P<left>{_ENTITY_SPAN})(?:\s+\(\s*\d+\s*\))?\s+[vV]\s+"
    rf"(?P<right>{_ENTITY_SPAN})(?:\s+\(\s*\d+\s*\))?(?=\s|$)"
)
_STANDINGS_ROW_RE = re.compile(
    rf"(?m)^(?P<team>{_ENTITY_SPAN})\s+\d+(?:\s+\d+){{3,}}\s*$"
)
_HEAD_TO_HEAD_SCORE_RE = re.compile(
    rf"(?m)^(?P<team1>{_ENTITY_SPAN})\s+\d+\s+(?P<team2>{_ENTITY_SPAN})\s+\d+(?:\s*\.)?\s*$"
)
_ORG_CONTEXT_RE = re.compile(
    rf"(?P<org>{_ENTITY_SPAN}\s+(?:news agency|newspaper|Party|Council|Union|Administration))"
)
_ORG_ALIAS_RE = re.compile(
    rf"(?P<long>{_ENTITY_SPAN})\s+\(\s*(?P<acro>[A-Z](?:[A-Z.]{{1,}}))\s*\)"
)
_BARE_AGENCY_RE = re.compile(
    r"\b(?:told|quoted|according to)\s+(?P<org>Reuters|Interfax|Itar-Tass|TASR)\b"
)


@dataclass
class DeterministicEntity:
    start: int
    end: int
    value: str
    etype: str
    source: str
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "value": self.value,
            "type": self.etype,
            "source": self.source,
            "score": self.score,
        }


class DeterministicDetector:
    """
    Détecteur d'entités basé uniquement sur des règles déterministes (Regex, Algorithmes).
    Ne dépend d'aucun modèle d'IA.
    """

    # Regex par défaut pour les types "library" si non fournis dans la config
    LIBRARY_REGEXES = {
        "IBAN": r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){9,30}\b",
        # Pour le téléphone, on utilise souvent la regex générique si la lib est utilisée pour valider
        "TELEPHONE": r"(?:\+|00)[1-9]\d{1,14}|(?:\+33|0)[1-9]\d{8}",
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/patterns_config.yaml"
        self.patterns = []
        self.forbidden_defaults = []
        self._load_config()

    def _load_config(self):
        """Charge la configuration des patterns depuis le fichier YAML."""
        try:
            path = Path(self.config_path)
            if not path.exists():
                # Fallback: chercher dans le dossier parent ou config/
                alt_path = Path("pipegraph") / self.config_path
                if alt_path.exists():
                    path = alt_path
                else:
                    # Dernier recours: chercher dans le dossier courant (si exécuté depuis pipegraph/)
                    if Path(self.config_path).exists():
                        path = Path(self.config_path)
                    else:
                        logger.warning(f"Config file not found: {self.config_path}")
                        return

            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not config:
                return

            # Chargement des patterns
            patterns_dict = config.get("patterns", {})
            # Tri par priorité (plus petit = plus prioritaire)
            sorted_patterns = sorted(
                patterns_dict.items(), key=lambda item: item[1].get("priority", 999)
            )

            self.patterns = []
            for name, spec in sorted_patterns:
                if not spec.get("enabled", True):
                    continue

                regex_str = spec.get("regex")
                ptype = spec.get("type")  # "regex" (default) or "library"
                entity_type = spec.get("entity_type", name.upper())

                # Gestion des types "library" qui n'ont pas forcément de regex dans la config
                if ptype == "library" and not regex_str:
                    regex_str = self.LIBRARY_REGEXES.get(entity_type)
                    if not regex_str:
                        logger.warning(f"No default regex for library type {entity_type} ({name})")
                        continue

                if regex_str:
                    try:
                        compiled_regex = re.compile(regex_str)
                        self.patterns.append(
                            {
                                "name": name,
                                "regex": compiled_regex,
                                "type": entity_type,
                                "validate_with": spec.get("validate_with"),
                                "group_index": spec.get(
                                    "group_index"
                                ),  # Pour extraire un groupe spécifique (ex: uid=123(user))
                                "spec": spec,
                            }
                        )
                    except re.error as e:
                        logger.error(f"Invalid regex for {name}: {e}")

            self.forbidden_defaults = set(config.get("forbidden_defaults", []))

            logger.info(f"Loaded {len(self.patterns)} deterministic patterns.")

        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def detect(self, text: str) -> List[DeterministicEntity]:
        """Exécute toutes les détections déterministes."""
        logger.debug(f"Début de l'analyse déterministe ({len(self.patterns)} patterns chargés)...")
        entities = []

        # Détection par Regex
        entities.extend(self._detect_regex_patterns(text))
        entities.extend(self._detect_news_sports_heuristics(text))

        deduplicated = self._deduplicate(entities)
        logger.debug(
            f"Fin de l'analyse déterministe. {len(entities)} brutes -> {len(deduplicated)} uniques."
        )
        return deduplicated

    def _detect_news_sports_heuristics(self, text: str) -> List[DeterministicEntity]:
        entities: List[DeterministicEntity] = []

        def _append_entity(start: int, end: int, etype: str = "ORG") -> None:
            if end <= start:
                return
            value = text[start:end].strip()
            if len(value) < 2 or is_whitelisted(value):
                return
            entities.append(
                DeterministicEntity(
                    start=start,
                    end=end,
                    value=value,
                    etype=etype,
                    source="heuristic",
                    score=0.92,
                )
            )

        for match in _FIXTURE_RE.finditer(text):
            _append_entity(match.start("left"), match.end("left"))
            _append_entity(match.start("right"), match.end("right"))

        for match in _STANDINGS_ROW_RE.finditer(text):
            _append_entity(match.start("team"), match.end("team"))

        for match in _HEAD_TO_HEAD_SCORE_RE.finditer(text):
            _append_entity(match.start("team1"), match.end("team1"))
            _append_entity(match.start("team2"), match.end("team2"))

        for match in _ORG_CONTEXT_RE.finditer(text):
            _append_entity(match.start("org"), match.end("org"))

        for match in _ORG_ALIAS_RE.finditer(text):
            _append_entity(match.start("long"), match.end("long"))
            _append_entity(match.start("acro"), match.end("acro"))

        for match in _BARE_AGENCY_RE.finditer(text):
            _append_entity(match.start("org"), match.end("org"))

        return entities

    def _detect_regex_patterns(self, text: str) -> List[DeterministicEntity]:
        entities = []
        for pattern in self.patterns:
            regex = pattern["regex"]
            etype = pattern["type"]
            name = pattern["name"]
            validator_name = pattern["validate_with"]
            group_index = pattern.get("group_index")

            if not validator_name:
                if etype == "IBAN":
                    validator_name = "iban"
                elif etype == "TELEPHONE":
                    validator_name = "phone"
                elif etype == "BIC":
                    validator_name = "bic"

            validator_func = Validators.get_validator(validator_name)

            matches_for_pattern = 0
            for match in regex.finditer(text):
                if group_index is not None and isinstance(group_index, int):
                    try:
                        value = match.group(group_index)
                        start = match.start(group_index)
                        end = match.end(group_index)
                    except IndexError:
                        continue
                else:
                    value = match.group()
                    start = match.start()
                    end = match.end()

                if is_whitelisted(value):
                    continue

                if value in self.forbidden_defaults:
                    continue

                if validator_func:
                    if not validator_func(value):
                        continue

                entities.append(
                    DeterministicEntity(
                        start=start, end=end, value=value, etype=etype, source="regex", score=1.0
                    )
                )
                matches_for_pattern += 1

            if matches_for_pattern > 0:
                logger.debug(f"Pattern '{name}' ({etype}): {matches_for_pattern} trouvés")

        return entities

    def _deduplicate(self, entities: List[DeterministicEntity]) -> List[DeterministicEntity]:
        """Supprime les chevauchements en gardant le plus long ou le premier."""
        if not entities:
            return []

        # Tri par position de début, puis longueur décroissante
        sorted_ents = sorted(entities, key=lambda x: (x.start, -(x.end - x.start)))

        merged = []
        for ent in sorted_ents:
            if not merged:
                merged.append(ent)
                continue

            last = merged[-1]

            # Si chevauchement
            if ent.start < last.end:
                # Si l'entité courante est incluse dans la précédente, on l'ignore
                # (la précédente est plus longue ou égale grâce au tri)
                continue
            else:
                merged.append(ent)

        return merged
