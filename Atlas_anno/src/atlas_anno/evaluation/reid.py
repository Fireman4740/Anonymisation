from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from atlas_anno.schemas import AttackResult, DocumentRecord


def _empty_counts() -> Dict[str, float]:
    return {"n": 0, "top1": 0, "top3": 0, "top5": 0, "mrr_total": 0.0, "singleton": 0, "pool_total": 0}


def _add_counts(counts: Dict[str, float], attack: AttackResult, target: str) -> float:
    ranked_ids = [entry.person_id for entry in attack.top_k]
    counts["n"] += 1
    counts["top1"] += int(ranked_ids[:1] == [target])
    counts["top3"] += int(target in ranked_ids[:3])
    counts["top5"] += int(target in ranked_ids[:5])
    counts["singleton"] += int(attack.candidate_pool_size == 1)
    counts["pool_total"] += attack.candidate_pool_size
    rank = ranked_ids.index(target) + 1 if target in ranked_ids else 0
    reciprocal_rank = 1.0 / rank if rank else 0.0
    counts["mrr_total"] += reciprocal_rank
    return reciprocal_rank


def _summarize(counts: Dict[str, float]) -> Dict[str, object]:
    n = int(counts["n"]) or 1
    return {
        "attacks": int(counts["n"]),
        "top1": round(counts["top1"] / n, 4),
        "top3": round(counts["top3"] / n, 4),
        "top5": round(counts["top5"] / n, 4),
        "mrr": round(counts["mrr_total"] / n, 4),
        "singleton_rate": round(counts["singleton"] / n, 4),
        "avg_candidate_pool_size": round(counts["pool_total"] / n, 4),
    }


def evaluate_reidentification(documents: List[DocumentRecord], attacks: Dict[str, AttackResult] | Sequence[AttackResult]) -> Dict[str, object]:
    if isinstance(attacks, dict):
        attack_rows = list(attacks.values())
    else:
        attack_rows = list(attacks)
    documents_by_id = {document.doc_id: document for document in documents}
    top1 = top3 = top5 = 0
    mrr_total = 0.0
    singleton = 0
    details = []
    grouped: Dict[str, Dict[str, float]] = defaultdict(_empty_counts)

    for attack in attack_rows:
        document = documents_by_id.get(attack.doc_id)
        if document is None:
            continue
        target = document.author_id
        ranked_ids = [entry.person_id for entry in attack.top_k]
        if ranked_ids[:1] == [target]:
            top1 += 1
        if target in ranked_ids[:3]:
            top3 += 1
        if target in ranked_ids[:5]:
            top5 += 1
        if attack.candidate_pool_size == 1:
            singleton += 1
        rank = ranked_ids.index(target) + 1 if target in ranked_ids else 0
        reciprocal_rank = 1.0 / rank if rank else 0.0
        mrr_total += reciprocal_rank
        aux_level = str(attack.metadata.get("aux_level", "legacy"))
        difficulty = str(attack.metadata.get("difficulty", document.metadata.get("difficulty", document.scenario.difficulty)))
        _add_counts(grouped[f"{difficulty}|{aux_level}"], attack, target)
        details.append(
            {
                "doc_id": document.doc_id,
                "pair_id": attack.metadata.get("pair_id", ""),
                "aux_level": aux_level,
                "difficulty": difficulty,
                "best_person_id": attack.best_person_id,
                "target_person_id": target,
                "candidate_pool_size": attack.candidate_pool_size,
                "reciprocal_rank": round(reciprocal_rank, 4),
            }
        )

    n_documents = len(details) or 1
    return {
        "summary": {
            "documents": len(documents),
            "attacks": len(details),
            "top1": round(top1 / n_documents, 4),
            "top3": round(top3 / n_documents, 4),
            "top5": round(top5 / n_documents, 4),
            "mrr": round(mrr_total / n_documents, 4),
            "singleton_rate": round(singleton / n_documents, 4),
            "avg_candidate_pool_size": round(sum(item.candidate_pool_size for item in attack_rows) / n_documents, 4),
            "by_difficulty_aux_level": {
                key: _summarize(value) for key, value in sorted(grouped.items())
            },
        },
        "details": details,
    }

