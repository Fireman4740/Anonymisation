from __future__ import annotations

from typing import Any, Dict, List

from atlas_anno.records import attack_result_from_dict
from atlas_anno.schemas import AttackCandidateScore, AttackResult, CharacterProfile, DocumentRecord
from atlas_anno.storage import (
    load_anonymization_results,
    load_attack_pairs,
    load_characters,
    load_documents,
    save_attack_results,
)


WEIGHTS = {
    "full_name": 8.0,
    "email": 7.0,
    "phone": 7.0,
    "username": 6.0,
    "account_id": 6.0,
    "role": 3.0,
    "team": 3.0,
    "department": 2.0,
    "location": 2.0,
    "nationality": 1.0,
    "age_range": 2.0,
    "degrees": 2.0,
    "certifications": 2.5,
    "rare_traits": 4.0,
    "events": 3.0,
    "sensitive_attributes": 2.0,
}


def _score_candidate(text: str, candidate: CharacterProfile, aux_knowledge: Dict[str, Any] | None = None) -> AttackCandidateScore:
    matched: List[str] = []
    score = 0.0
    for field_name, weight in WEIGHTS.items():
        value = getattr(candidate, field_name)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not item:
                continue
            if item in text:
                score += weight
                matched.append(f"{field_name}:{item}")
    if candidate.style_profile.signature_pattern in text:
        score += 1.0
        matched.append(f"signature:{candidate.style_profile.signature_pattern}")
    if candidate.style_profile.jargon_pattern in text:
        score += 1.0
        matched.append(f"jargon:{candidate.style_profile.jargon_pattern}")
    for field_name, known_value in (aux_knowledge or {}).items():
        candidate_value = getattr(candidate, field_name, None)
        if candidate_value is None and field_name == "rare_trait":
            if known_value in candidate.rare_traits:
                score += 2.0
                matched.append(f"aux:{field_name}:{known_value}")
            continue
        if candidate_value == known_value:
            score += 2.0
            matched.append(f"aux:{field_name}:{known_value}")
    return AttackCandidateScore(person_id=candidate.person_id, score=round(score, 4), matched_signals=matched)


def _attack_text_by_doc(documents: List[DocumentRecord], strategy: str) -> Dict[str, str]:
    if strategy == "raw":
        return {document.doc_id: document.text for document in documents}
    return {item.doc_id: item.anonymized_text for item in load_anonymization_results(strategy)}


def run_attack(documents: List[DocumentRecord], characters: Dict[str, CharacterProfile], strategy: str) -> List[AttackResult]:
    attack_texts = _attack_text_by_doc(documents, strategy)
    results: List[AttackResult] = []
    documents_by_id = {document.doc_id: document for document in documents}
    pairs = [
        pair for pair in load_attack_pairs()
        if pair.doc_id in documents_by_id
        and pair.target_person_id in characters
        and any(candidate_id in characters for candidate_id in pair.candidate_pool)
    ]
    if pairs:
        for pair in pairs:
            attack_text = attack_texts[pair.doc_id]
            candidate_ids = [candidate_id for candidate_id in pair.candidate_pool if candidate_id in characters]
            candidates = [
                _score_candidate(attack_text, characters[candidate_id], pair.aux_knowledge.known_attributes)
                for candidate_id in candidate_ids
            ]
            ranked = sorted(candidates, key=lambda item: (-item.score, item.person_id))
            best = ranked[0] if ranked else AttackCandidateScore(person_id="", score=0.0, matched_signals=[])
            total = sum(item.score for item in ranked) or 1.0
            confidence = round(best.score / total, 4)
            results.append(
                AttackResult(
                    doc_id=pair.doc_id,
                    attacker_type="structured",
                    top_k=ranked[:5],
                    best_person_id=best.person_id,
                    confidence=confidence,
                    candidate_pool_size=len(candidate_ids),
                    matched_signals=list(best.matched_signals),
                    metadata={
                        "strategy": strategy,
                        "pair_id": pair.pair_id,
                        "aux_level": pair.aux_knowledge.level,
                        "difficulty": pair.difficulty,
                    },
                )
            )
        return results

    for document in documents:
        attack_text = attack_texts[document.doc_id]
        candidate_ids = document.candidate_pools.insider if document.split == "test_hard" else document.candidate_pools.org_internal
        candidates = [_score_candidate(attack_text, characters[candidate_id]) for candidate_id in candidate_ids if candidate_id in characters]
        ranked = sorted(candidates, key=lambda item: (-item.score, item.person_id))
        best = ranked[0] if ranked else AttackCandidateScore(person_id="", score=0.0, matched_signals=[])
        total = sum(item.score for item in ranked) or 1.0
        confidence = round(best.score / total, 4)
        results.append(
            AttackResult(
                doc_id=document.doc_id,
                attacker_type="structured",
                top_k=ranked[:5],
                best_person_id=best.person_id,
                confidence=confidence,
                candidate_pool_size=len(candidate_ids),
                matched_signals=list(best.matched_signals),
                metadata={"strategy": strategy},
            )
        )
    return results


def run_structured_attack_command(strategy: str) -> None:
    documents = load_documents(annotated=True)
    characters = {character.person_id: character for character in load_characters()}
    results = run_attack(documents, characters, strategy)
    save_attack_results(strategy, "structured", results)

