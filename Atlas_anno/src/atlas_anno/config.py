from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from atlas_anno.constants import DEFAULT_CONFIG_PATH, DEFAULT_EVALUATION_PATH, DEFAULT_ONTOLOGY_PATH
from atlas_anno.io import load_yaml, project_file


@dataclass(frozen=True)
class AtlasConfig:
    defaults: Dict[str, Any]
    ontology: Dict[str, Any]
    evaluation: Dict[str, Any]


def load_config() -> AtlasConfig:
    return AtlasConfig(
        defaults=load_yaml(project_file(DEFAULT_CONFIG_PATH)),
        ontology=load_yaml(project_file(DEFAULT_ONTOLOGY_PATH)),
        evaluation=load_yaml(project_file(DEFAULT_EVALUATION_PATH)),
    )

