from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

from atlas_anno.paths import project_root


ARTIFACTS = {
    "worlds": "worlds/worlds.jsonl",
    "characters": "characters/characters.jsonl",
    "scenarios": "scenarios/scenarios.jsonl",
    "raw_docs": "raw_docs/raw_docs.jsonl",
    "annotations": "annotations/preannotations.jsonl",
    "anonymized": "anonymized/{strategy}.jsonl",
    "attack_pairs": "attacks/pairs.jsonl",
    "attacks_structured": "attacks/{strategy}_structured.jsonl",
    "attacks_llm": "attacks/{strategy}_llm.jsonl",
    "llm_runs": "logs/llm_runs.jsonl",
}


@dataclass(frozen=True)
class DashboardData:
    data_dir: Path
    batch: str
    strategy: str
    worlds: List[Dict[str, Any]]
    characters: List[Dict[str, Any]]
    scenarios: List[Dict[str, Any]]
    raw_docs: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    anonymized: List[Dict[str, Any]]
    attack_pairs: List[Dict[str, Any]]
    attacks_structured: List[Dict[str, Any]]
    attacks_llm: List[Dict[str, Any]]
    llm_runs: List[Dict[str, Any]]
    reports: Dict[str, Dict[str, Any]]
    parquet_files: Dict[str, Path]
    missing_files: List[Path] = field(default_factory=list)


@dataclass(frozen=True)
class DashboardSummary:
    coverage: Dict[str, int]
    duplicates: Dict[str, int]
    missing_files: List[str]
    report_metrics: Dict[str, Any]
    factor_counts: Dict[str, Dict[str, int]]
    linguistic_metrics: Dict[str, Any]
    linguistic_flags: Dict[str, bool]
    quality_tables: Dict[str, List[Dict[str, Any]]]
    anonymization_tables: Dict[str, List[Dict[str, Any]]]
    reid_tables: Dict[str, List[Dict[str, Any]]]
    document_rows: List[Dict[str, Any]]
    llm_tables: Dict[str, List[Dict[str, Any]]]


def _resolve_data_dir(data_dir: Path) -> Path:
    if data_dir.exists():
        return data_dir
    atlas_relative = project_root() / data_dir
    if atlas_relative.exists():
        return atlas_relative
    return data_dir


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_dashboard_data(data_dir: Path | str, *, batch: str, strategy: str) -> DashboardData:
    base = _resolve_data_dir(Path(data_dir))
    rows: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[Path] = []
    for name, pattern in ARTIFACTS.items():
        path = base / pattern.format(strategy=strategy)
        rows[name] = _read_jsonl(path)
        if not path.exists():
            missing.append(path)

    reports_dir = base / "reports"
    reports = {path.stem: _read_json(path) for path in sorted(reports_dir.glob("*.json"))}
    parquet_dir = base / "parquet" / batch
    parquet_files = {path.stem: path for path in sorted(parquet_dir.glob("*.parquet"))}

    return DashboardData(
        data_dir=base,
        batch=batch,
        strategy=strategy,
        worlds=rows["worlds"],
        characters=rows["characters"],
        scenarios=rows["scenarios"],
        raw_docs=rows["raw_docs"],
        annotations=rows["annotations"],
        anonymized=rows["anonymized"],
        attack_pairs=rows["attack_pairs"],
        attacks_structured=rows["attacks_structured"],
        attacks_llm=rows["attacks_llm"],
        llm_runs=rows["llm_runs"],
        reports=reports,
        parquet_files=parquet_files,
        missing_files=missing,
    )


def summarize_dashboard_data(data: DashboardData) -> DashboardSummary:
    docs_by_id = {str(row.get("doc_id")): row for row in data.raw_docs}
    annotations_by_id = {str(row.get("doc_id")): row for row in data.annotations}
    anonymized_by_id = {str(row.get("doc_id")): row for row in data.anonymized}
    pairs_by_id = {str(row.get("pair_id")): row for row in data.attack_pairs}
    pair_targets = {str(row.get("pair_id")): row.get("target_person_id") for row in data.attack_pairs}

    coverage = {
        "worlds": len(data.worlds),
        "characters": len(data.characters),
        "scenarios": len(data.scenarios),
        "raw_docs": len(data.raw_docs),
        "annotations": len(data.annotations),
        "anonymized": len(data.anonymized),
        "attack_pairs": len(data.attack_pairs),
        "attacks_structured": len(data.attacks_structured),
        "attacks_llm": len(data.attacks_llm),
        "llm_runs": len(data.llm_runs),
        "parquet_files": len(data.parquet_files),
    }

    duplicates = {
        "raw_docs": _duplicate_count(data.raw_docs, "doc_id"),
        "annotations": _duplicate_count(data.annotations, "doc_id"),
        "anonymized": _duplicate_count(data.anonymized, "doc_id"),
        "attack_pairs": _duplicate_count(data.attack_pairs, "pair_id"),
        "attacks_structured_pair": _attack_duplicate_count(data.attacks_structured),
        "attacks_llm_pair": _attack_duplicate_count(data.attacks_llm),
    }

    factor_counts = _factor_counts(data.raw_docs, data.characters)
    linguistic_metrics = _linguistic_metrics(data)
    linguistic_flags = {
        "self_bleu_collapse": float(linguistic_metrics.get("self_bleu") or 0.0) > 0.90,
        "distinct_2_low": float(linguistic_metrics.get("distinct_2") or 0.0) < 0.15,
        "cell_coverage_low": float(linguistic_metrics.get("cell_coverage") or 0.0) < 0.60,
    }

    report_metrics = _report_metrics(data.reports, data.strategy)
    quality_tables = _quality_tables(data, factor_counts)
    anonymization_tables = _anonymization_tables(data)
    reid_tables = _reid_tables(data, pairs_by_id, pair_targets)
    document_rows = _document_rows(data, docs_by_id, annotations_by_id, anonymized_by_id, pairs_by_id)
    llm_tables = _llm_tables(data)

    return DashboardSummary(
        coverage=coverage,
        duplicates=duplicates,
        missing_files=[str(path) for path in data.missing_files],
        report_metrics=report_metrics,
        factor_counts=factor_counts,
        linguistic_metrics=linguistic_metrics,
        linguistic_flags=linguistic_flags,
        quality_tables=quality_tables,
        anonymization_tables=anonymization_tables,
        reid_tables=reid_tables,
        document_rows=document_rows,
        llm_tables=llm_tables,
    )


def _duplicate_count(rows: List[Dict[str, Any]], key: str) -> int:
    values = [row.get(key) for row in rows if row.get(key) not in (None, "")]
    return len(values) - len(set(values))


def _attack_duplicate_count(rows: List[Dict[str, Any]]) -> int:
    keys = []
    for row in rows:
        metadata = row.get("metadata") or {}
        pair_id = metadata.get("pair_id")
        if pair_id:
            keys.append(pair_id)
        else:
            keys.append((row.get("doc_id"), metadata.get("aux_level"), metadata.get("difficulty")))
    return len(keys) - len(set(keys))


def _counter_table(counter: Counter[Any], *, key_name: str = "value", count_name: str = "count") -> List[Dict[str, Any]]:
    total = sum(counter.values()) or 1
    return [
        {key_name: str(key), count_name: count, "share": round(count / total, 4)}
        for key, count in counter.most_common()
    ]


def _factor_counts(raw_docs: List[Dict[str, Any]], characters: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = defaultdict(Counter)
    chars_by_id = {str(row.get("person_id")): row for row in characters}
    for doc in raw_docs:
        scenario = doc.get("scenario") or {}
        author = chars_by_id.get(str(doc.get("author_id")), {})
        values = {
            "domain": doc.get("domain") or scenario.get("domain"),
            "difficulty": scenario.get("difficulty") or doc.get("metadata", {}).get("difficulty"),
            "split": doc.get("split") or scenario.get("split"),
            "register": scenario.get("register") or doc.get("metadata", {}).get("register") or "unknown",
            "address_form": scenario.get("address_form") or "unknown",
            "document_goal": scenario.get("document_goal") or "unknown",
            "urgency": scenario.get("urgency") or "unknown",
            "noise_level": scenario.get("noise_level") or "unknown",
            "expertise_level": (author.get("style_profile") or {}).get("expertise_level", "unknown"),
            "francophone_variety": (author.get("style_profile") or {}).get("francophone_variety", "unknown"),
            "typo_propensity": (author.get("style_profile") or {}).get("typo_propensity", "unknown"),
        }
        for key, value in values.items():
            counters[key][str(value or "unknown")] += 1
    return {key: dict(counter) for key, counter in counters.items()}


def _report_metrics(reports: Dict[str, Dict[str, Any]], strategy: str) -> Dict[str, Any]:
    consolidated = reports.get(f"{strategy}_report", {})
    summary = dict(consolidated.get("summary") or {})
    spans = reports.get(f"{strategy}_spans", {}).get("summary", {})
    privacy = reports.get(f"{strategy}_privacy", {}).get("summary", {})
    reid = reports.get(f"{strategy}_reid", {}).get("summary", {})
    utility = reports.get(f"{strategy}_utility", {}).get("summary", {})
    diversity = reports.get("raw_diversity", {}).get("summary", {})
    calibration = reports.get("raw_calibration", {})
    return {
        "privacy_score": _first(summary.get("privacy_score"), privacy.get("privacy_score")),
        "utility_score": _first(summary.get("utility_score"), utility.get("utility_score")),
        "reid_top1": _first(summary.get("reid_top1"), reid.get("top1")),
        "span_f1": _first(summary.get("span_f1"), spans.get("f1")),
        "self_bleu": _first(summary.get("self_bleu"), diversity.get("self_bleu")),
        "distinct_2": _first(summary.get("distinct_2"), diversity.get("distinct_2")),
        "duplicate_rate": _first(summary.get("duplicate_rate"), diversity.get("duplicate_rate")),
        "cell_coverage": diversity.get("cell_coverage"),
        "cell_entropy": diversity.get("cell_entropy"),
        "calibration_passed": _first(summary.get("calibration_passed"), calibration.get("passed")),
        "diversity_passed": _first(summary.get("diversity_passed"), diversity.get("passed")),
    }


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _linguistic_metrics(data: DashboardData) -> Dict[str, Any]:
    diversity = data.reports.get("raw_diversity", {}).get("summary", {})
    lengths = [len(str(row.get("text") or "")) for row in data.raw_docs]
    tokens = [len(str(row.get("text") or "").split()) for row in data.raw_docs]
    return {
        "documents": len(data.raw_docs),
        "distinct_1": diversity.get("distinct_1"),
        "distinct_2": diversity.get("distinct_2"),
        "distinct_3": diversity.get("distinct_3"),
        "self_bleu": diversity.get("self_bleu"),
        "duplicate_rate": diversity.get("duplicate_rate"),
        "cell_coverage": diversity.get("cell_coverage"),
        "cell_entropy": diversity.get("cell_entropy"),
        "text_length_min": min(lengths) if lengths else 0,
        "text_length_median": median(lengths) if lengths else 0,
        "text_length_mean": round(mean(lengths), 2) if lengths else 0,
        "text_length_max": max(lengths) if lengths else 0,
        "token_count_mean": round(mean(tokens), 2) if tokens else 0,
    }


def _quality_tables(data: DashboardData, factor_counts: Dict[str, Dict[str, int]]) -> Dict[str, List[Dict[str, Any]]]:
    missing_required = []
    for name, rows in {
        "raw_docs": data.raw_docs,
        "annotations": data.annotations,
        "anonymized": data.anonymized,
        "attack_pairs": data.attack_pairs,
    }.items():
        for field_name in _required_fields(name):
            missing = sum(1 for row in rows if row.get(field_name) in (None, "", []))
            if missing:
                missing_required.append({"artifact": name, "field": field_name, "missing": missing})

    factor_tables = {
        f"factor_{key}": _counter_table(Counter(values), key_name=key)
        for key, values in factor_counts.items()
    }
    tables = {"missing_required": missing_required}
    tables.update(factor_tables)
    return tables


def _required_fields(name: str) -> List[str]:
    return {
        "raw_docs": ["doc_id", "text", "domain", "author_id", "split"],
        "annotations": ["doc_id", "text", "annotations"],
        "anonymized": ["doc_id", "anonymized_text", "strategy"],
        "attack_pairs": ["pair_id", "doc_id", "target_person_id", "candidate_pool"],
    }.get(name, [])


def _anonymization_tables(data: DashboardData) -> Dict[str, List[Dict[str, Any]]]:
    actions = Counter()
    rows = []
    for row in data.anonymized:
        for action in row.get("actions_performed") or []:
            actions[str(action)] += 1
        rows.append(
            {
                "doc_id": row.get("doc_id"),
                "estimated_privacy_gain": _to_float(row.get("estimated_privacy_gain")),
                "estimated_utility_loss": _to_float(row.get("estimated_utility_loss")),
                "actions_count": len(row.get("actions_performed") or []),
            }
        )
    rows.sort(key=lambda item: item["estimated_utility_loss"], reverse=True)
    return {"actions": _counter_table(actions, key_name="action"), "documents": rows}


def _reid_tables(
    data: DashboardData,
    pairs_by_id: Dict[str, Dict[str, Any]],
    pair_targets: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    rows = []
    for attack_type, attacks in (("structured", data.attacks_structured), ("llm", data.attacks_llm)):
        for row in attacks:
            metadata = row.get("metadata") or {}
            pair_id = str(metadata.get("pair_id") or "")
            pair = pairs_by_id.get(pair_id, {})
            target = pair_targets.get(pair_id)
            ranked_ids = [candidate.get("person_id") for candidate in row.get("top_k") or []]
            best_person_id = row.get("best_person_id")
            # Sans cible connue, aucune attaque ne peut être comptée comme réussie
            # (sinon None == None gonflerait faussement le risque de ré-identification).
            has_target = target is not None
            rows.append(
                {
                    "attacker_type": attack_type,
                    "pair_id": pair_id,
                    "doc_id": row.get("doc_id"),
                    "difficulty": metadata.get("difficulty") or pair.get("difficulty"),
                    "aux_level": metadata.get("aux_level") or (pair.get("aux_knowledge") or {}).get("level"),
                    "candidate_pool_size": row.get("candidate_pool_size"),
                    "best_person_id": best_person_id,
                    "target_person_id": target,
                    "top1_success": has_target and best_person_id == target,
                    "target_in_top3": has_target and target in ranked_ids[:3],
                    "confidence": _to_float(row.get("confidence")),
                }
            )
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["attacker_type"]), str(row["difficulty"]), str(row["aux_level"]))].append(row)
    by_segment = []
    for (attacker_type, difficulty, aux_level), items in sorted(grouped.items()):
        n = len(items) or 1
        by_segment.append(
            {
                "attacker_type": attacker_type,
                "difficulty": difficulty,
                "aux_level": aux_level,
                "attacks": len(items),
                "top1": round(sum(1 for item in items if item["top1_success"]) / n, 4),
                "top3": round(sum(1 for item in items if item["target_in_top3"]) / n, 4),
            }
        )
    risky = sorted(rows, key=lambda item: (not item["top1_success"], -item["confidence"]))[:50]
    return {"attacks": rows, "by_segment": by_segment, "risky": risky}


def _document_rows(
    data: DashboardData,
    docs_by_id: Dict[str, Dict[str, Any]],
    annotations_by_id: Dict[str, Dict[str, Any]],
    anonymized_by_id: Dict[str, Dict[str, Any]],
    pairs_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pair_by_doc_aux = {
        (str(row.get("doc_id")), str((row.get("aux_knowledge") or {}).get("level"))): row
        for row in data.attack_pairs
    }
    attack_success_by_doc = defaultdict(int)
    attack_count_by_doc = defaultdict(int)
    for attack in data.attacks_structured:
        pair_id = str((attack.get("metadata") or {}).get("pair_id") or "")
        target = pairs_by_id.get(pair_id, {}).get("target_person_id")
        doc_id = str(attack.get("doc_id"))
        attack_count_by_doc[doc_id] += 1
        attack_success_by_doc[doc_id] += int(target is not None and attack.get("best_person_id") == target)

    rows = []
    for doc_id, doc in docs_by_id.items():
        scenario = doc.get("scenario") or {}
        annotation = annotations_by_id.get(doc_id, {})
        anonymized = anonymized_by_id.get(doc_id, {})
        spans = (annotation.get("annotations") or {}).get("spans") or []
        pair_none = pair_by_doc_aux.get((doc_id, "none"), {})
        rows.append(
            {
                "doc_id": doc_id,
                "domain": doc.get("domain") or scenario.get("domain"),
                "split": doc.get("split") or scenario.get("split"),
                "difficulty": scenario.get("difficulty") or doc.get("metadata", {}).get("difficulty"),
                "register": scenario.get("register") or "unknown",
                "address_form": scenario.get("address_form") or "unknown",
                "document_goal": scenario.get("document_goal") or "unknown",
                "text_length": len(str(doc.get("text") or "")),
                "text": doc.get("text") or "",
                "anonymized_text": anonymized.get("anonymized_text") or "",
                "span_count": len(spans),
                "actions_count": len(anonymized.get("actions_performed") or []),
                "estimated_utility_loss": _to_float(anonymized.get("estimated_utility_loss")),
                "attack_top1_success": attack_success_by_doc[doc_id],
                "attack_count": attack_count_by_doc[doc_id],
                "candidate_pool_size": len(pair_none.get("candidate_pool") or []),
            }
        )
    return rows


def _llm_tables(data: DashboardData) -> Dict[str, List[Dict[str, Any]]]:
    step_counts = Counter(str(row.get("step_name") or row.get("prompt_name") or "unknown") for row in data.llm_runs)
    fallback_counts = Counter(str(bool(row.get("fallback_used"))) for row in data.llm_runs)
    error_counts = Counter(str(bool(row.get("error"))) for row in data.llm_runs)
    return {
        "steps": _counter_table(step_counts, key_name="step"),
        "fallback": _counter_table(fallback_counts, key_name="fallback_used"),
        "errors": _counter_table(error_counts, key_name="error"),
    }


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

