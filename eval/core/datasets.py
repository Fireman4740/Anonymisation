from __future__ import annotations

import os
from typing import Any, FrozenSet, List, Optional, Set, Tuple

from eval.core.pipeline import (
    build_docs_from_anonymization_dataset,
    build_docs_from_db_bio,
    build_docs_from_personalreddit,
    build_docs_from_tab,
)
from eval.core.loaders.ratbench import build_docs_from_ratbench
from eval.core.loaders.conll2003 import build_docs_from_conll2003, get_conll2003_dataset_name
from eval.core.profiles import resolve_eval_profile

Span = Tuple[int, int, str]
DatasetDoc = Tuple[str, str, List[Span]]

DATASET_ALIASES: dict[str, str] = {
    "tab": "tab",
    "db-bio": "dbbio",
    "dbbio": "dbbio",
    "json": "anonymization",
    "anonymization": "anonymization",
    "rat-bench": "ratbench",
    "ratbench": "ratbench",
    "conll2003": "conll2003",
    "cleanconll2003": "conll2003",
    "personalreddit": "personalreddit",
    "personal-reddit": "personalreddit",
    "reddit": "personalreddit",
}

# ---------------------------------------------------------------------------
# Label scope per dataset
# ---------------------------------------------------------------------------
# Maps each benchmark dataset to the set of entity labels it annotates.
# ``None`` means "no filtering" — all predicted labels are valid.
# When a scope is defined, only predictions matching these labels are kept
# for the precision/FP calculation, avoiding unfair penalisation when the
# pipeline detects valid PII that the dataset simply doesn't annotate.
#
# The scope is applied to *predictions only* — ground-truth is never
# filtered.
# ---------------------------------------------------------------------------

DATASET_LABEL_SCOPE: dict[str, Optional[FrozenSet[str]]] = {
    # CoNLL2003: strict NER benchmark — only 4 entity types
    "conll2003": frozenset({"PER", "ORG", "LOC", "MISC"}),
    # DB-bio: biographical texts — only person names annotated
    "dbbio": frozenset({"PER", "PERSON"}),
    # TAB / anonymization / ratbench / personalreddit: broad PII — no filtering
    "tab": None,
    "anonymization": None,
    "ratbench": None,
    "personalreddit": None,
}


def normalize_dataset_key(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset.strip().lower(), dataset.strip().lower())


def uses_news_ner_profile(dataset: str) -> bool:
    return normalize_dataset_key(dataset) == "conll2003"


def get_allowed_labels(dataset: str, profile: str = "auto") -> Optional[FrozenSet[str]]:
    """Return the set of labels valid for *dataset*, or ``None`` if all
    labels should be kept (no filtering)."""
    try:
        return resolve_eval_profile(profile, dataset_key=dataset).allowed_labels
    except Exception:
        return DATASET_LABEL_SCOPE.get(normalize_dataset_key(dataset))


def _infer_conll_split(dataset_path: str) -> str:
    base = os.path.basename(dataset_path).strip().lower()
    if base in {"train", "train.txt", "cleanconll.train"} or base.endswith(".train"):
        return "train"
    if base in {"dev", "dev.txt", "valid.txt", "cleanconll.dev"} or base.endswith(".dev"):
        return "dev"
    if base in {"test", "test.txt", "cleanconll.test"} or base.endswith(".test"):
        return "test"
    return "test"


def load_local_dataset_docs(
    *,
    dataset_kind: str,
    dataset_path: str,
    limit: Optional[int],
    split: Optional[str] = None,
) -> List[DatasetDoc]:
    normalized_kind = normalize_dataset_key(dataset_kind)

    if normalized_kind == "tab":
        return build_docs_from_tab(dataset_path, limit=limit)
    if normalized_kind == "dbbio":
        return build_docs_from_db_bio(dataset_path, limit=limit)
    if normalized_kind == "conll2003":
        return build_docs_from_conll2003(
            limit=limit,
            split=(split or _infer_conll_split(dataset_path)),
            variant="clean",
        )
    return build_docs_from_anonymization_dataset(dataset_path, limit=limit)


def load_benchmark_docs(
    *,
    dataset: str,
    project_root: str,
    limit: int,
    level: Optional[int] = None,
    language: str = "english",
    split: str = "test",
) -> Tuple[List[Any], str]:
    if dataset == "tab":
        path = os.path.join(project_root, "eval", "datasets", "TAB", f"{split}.jsonl")
        return build_docs_from_tab(path, limit=limit), f"TAB/{split}"

    if dataset == "dbbio":
        path = os.path.join(project_root, "eval", "datasets", "DB-bio", "test.jsonl")
        return build_docs_from_db_bio(path, limit=limit), "DB-bio/test"

    if dataset == "anonymization":
        path = os.path.join(project_root, "eval", "datasets", "data", "anonymization_dataset.json")
        return build_docs_from_anonymization_dataset(path, limit=limit), "anonymization_dataset"

    if dataset == "ratbench":
        docs = build_docs_from_ratbench(language=language, level=level, limit=limit)
        level_str = f"L{level}" if level else "all"
        return docs, f"RAT-Bench/{language}/{level_str}"

    if dataset in {"conll2003", "cleanconll2003"}:
        return build_docs_from_conll2003(limit=limit, split=split), get_conll2003_dataset_name(split)

    if dataset == "personalreddit":
        path = os.path.join(project_root, "eval", "datasets", "PersonalReddit", "Reddit_synthetic", f"{split}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(project_root, "eval", "datasets", "PersonalReddit", "Reddit_synthetic", "test.jsonl")
        return build_docs_from_personalreddit(path, limit=limit), f"PersonalReddit/{split}"

    raise ValueError(
        f"Dataset inconnu: {dataset!r}. Choix: tab, dbbio, anonymization, ratbench, conll2003, personalreddit"
    )
