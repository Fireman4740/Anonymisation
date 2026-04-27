from __future__ import annotations

from typing import Dict, List

from atlas_anno.schemas import AttackResult, DocumentRecord


def evaluate_reidentification(documents: List[DocumentRecord], attacks: Dict[str, AttackResult]) -> Dict[str, object]:
    top1 = top3 = top5 = 0
    mrr_total = 0.0
    singleton = 0
    details = []

    for document in documents:
        attack = attacks[document.doc_id]
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
        details.append(
            {
                "doc_id": document.doc_id,
                "best_person_id": attack.best_person_id,
                "target_person_id": target,
                "candidate_pool_size": attack.candidate_pool_size,
                "reciprocal_rank": round(reciprocal_rank, 4),
            }
        )

    n_documents = len(documents) or 1
    return {
        "summary": {
            "documents": len(documents),
            "top1": round(top1 / n_documents, 4),
            "top3": round(top3 / n_documents, 4),
            "top5": round(top5 / n_documents, 4),
            "mrr": round(mrr_total / n_documents, 4),
            "singleton_rate": round(singleton / n_documents, 4),
            "avg_candidate_pool_size": round(sum(item.candidate_pool_size for item in attacks.values()) / n_documents, 4),
        },
        "details": details,
    }

