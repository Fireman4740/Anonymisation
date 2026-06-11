"""Central dataset registry for the evaluation engine.

The registry is the single source of truth for dataset names, aliases,
capabilities and dataset-specific evaluation behavior.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from eval.core.datasets import DATASET_ALIASES, normalize_dataset_key
from eval.core.dataset_adapters import (
    AnonymizationDatasetAdapter,
    Conll2003DatasetAdapter,
    DatasetAdapter,
    DbBioDatasetAdapter,
    PersonalRedditDatasetAdapter,
    RatbenchDatasetAdapter,
    TabDatasetAdapter,
)


class DatasetRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, DatasetAdapter] = {}

    def register(self, adapter: DatasetAdapter, overwrite: bool = False) -> None:
        key = normalize_dataset_key(adapter.name)
        if key in self._adapters and not overwrite:
            raise ValueError(f"Dataset already registered: {key!r}")
        self._adapters[key] = adapter

    def get(self, name: str) -> DatasetAdapter:
        key = normalize_dataset_key(name)
        if key not in self._adapters:
            raise KeyError(f"Unknown dataset: {name!r}. Available: {', '.join(self.list())}")
        return self._adapters[key]

    def list(self) -> List[str]:
        return sorted(self._adapters)

    def describe_all(self) -> List[dict]:
        return [self._adapters[name].describe() for name in self.list()]

    def __contains__(self, name: str) -> bool:
        return normalize_dataset_key(name) in self._adapters


def _aliases_for(name: str) -> tuple[str, ...]:
    return tuple(sorted(alias for alias, target in DATASET_ALIASES.items() if target == name))


def _build_default_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    full = {"span_metrics": True, "leakage_metrics": True, "risk_metrics": False, "utility_metrics": True}

    registry.register(TabDatasetAdapter(
        name="tab",
        description="Text Anonymization Benchmark - 1268 ECHR court decisions, span+masking annotations",
        supports=dict(full),
        aliases=_aliases_for("tab"),
    ))
    registry.register(DbBioDatasetAdapter(
        name="dbbio",
        description="DB-bio - biographical texts, person names annotated (utility/occupation use case)",
        supports=dict(full),
        aliases=_aliases_for("dbbio"),
    ))
    registry.register(AnonymizationDatasetAdapter(
        name="anonymization",
        description="Local synthetic anonymization dataset (Atlas_anno generated)",
        supports=dict(full),
        aliases=_aliases_for("anonymization"),
    ))
    registry.register(RatbenchDatasetAdapter(
        name="ratbench",
        description="RAT-Bench - re-identification risk benchmark, direct/indirect identifiers, levels 1-3",
        supports={**full, "risk_metrics": True},
        aliases=_aliases_for("ratbench"),
    ))
    registry.register(Conll2003DatasetAdapter(
        name="conll2003",
        description="CleanCoNLL 2003 - strict NER benchmark (PER/ORG/LOC/MISC)",
        supports={"span_metrics": True, "leakage_metrics": False, "risk_metrics": False, "utility_metrics": False},
        aliases=_aliases_for("conll2003"),
    ))
    registry.register(PersonalRedditDatasetAdapter(
        name="personalreddit",
        description="PersonalReddit (synthetic) - informal first-person texts, privacy-utility trade-off",
        supports=dict(full),
        aliases=_aliases_for("personalreddit"),
    ))
    return registry


_REGISTRY: Optional[DatasetRegistry] = None


def get_registry() -> DatasetRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_default_registry()
    return _REGISTRY
