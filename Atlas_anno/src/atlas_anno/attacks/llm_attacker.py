from __future__ import annotations

from typing import Any, Dict, List

from atlas_anno.attacks.structured import _score_candidate, run_attack
from atlas_anno.llm import LLMError, OpenRouterClient
from atlas_anno.schemas import AttackCandidateScore, AttackResult
from atlas_anno.settings import load_settings
from atlas_anno.storage import (
    load_anonymization_results,
    load_attack_pairs,
    load_characters,
    load_documents,
    save_attack_results,
)


def _candidate_payload(character: Any) -> Dict[str, Any]:
    return {
        "person_id": character.person_id,
        "full_name": character.full_name,
        "role": character.role,
        "team": character.team,
        "department": character.department,
        "age_range": character.age_range,
        "certifications": character.certifications,
        "rare_traits": character.rare_traits,
        "events": character.events,
        "sensitive_attributes": character.sensitive_attributes,
    }


def _llm_attack(strategy: str) -> List[AttackResult]:
    settings = load_settings()
    client = OpenRouterClient(settings)
    documents = load_documents(annotated=True)
    if strategy == "raw":
        attack_texts = {document.doc_id: document.text for document in documents}
    else:
        attack_texts = {item.doc_id: item.anonymized_text for item in load_anonymization_results(strategy)}
    characters = {character.person_id: character for character in load_characters()}
    results: List[AttackResult] = []
    pairs = [
        pair for pair in load_attack_pairs()
        if pair.doc_id in {document.doc_id for document in documents}
        and pair.target_person_id in characters
        and any(candidate_id in characters for candidate_id in pair.candidate_pool)
    ]
    if pairs:
        for pair in pairs:
            candidate_ids = [candidate_id for candidate_id in pair.candidate_pool if candidate_id in characters]
            candidates = [_candidate_payload(characters[candidate_id]) for candidate_id in candidate_ids]
            try:
                response = client.score_candidates(attack_texts[pair.doc_id], candidates, pair.aux_knowledge.known_attributes)
                if not isinstance(response, dict):
                    raise LLMError(
                        f"llm attacker expected a JSON object, got {type(response).__name__}"
                    )
                top_k = []
                for item in response.get("top_k", [])[:5]:
                    if not isinstance(item, dict):
                        continue
                    top_k.append(
                        AttackCandidateScore(
                            person_id=str(item.get("person_id", "")),
                            score=float(item.get("score", 0.0)),
                            matched_signals=list(item.get("matched_signals", [])),
                        )
                    )
                if not top_k:
                    raise LLMError("llm attacker returned no candidates")
                best = top_k[0]
                confidence = round(best.score / max(1.0, sum(entry.score for entry in top_k)), 4)
                results.append(
                    AttackResult(
                        doc_id=pair.doc_id,
                        attacker_type="llm",
                        top_k=top_k,
                        best_person_id=best.person_id,
                        confidence=confidence,
                        candidate_pool_size=len(candidate_ids),
                        matched_signals=list(best.matched_signals),
                        metadata={
                            "strategy": strategy,
                            "pair_id": pair.pair_id,
                            "aux_level": pair.aux_knowledge.level,
                            "difficulty": pair.difficulty,
                            "rationale": response.get("rationale", ""),
                        },
                    )
                )
            except (LLMError, ValueError, TypeError, AttributeError):
                candidate_scores = [
                    _score_candidate(attack_texts[pair.doc_id], characters[candidate_id], pair.aux_knowledge.known_attributes)
                    for candidate_id in candidate_ids
                ]
                ranked = sorted(candidate_scores, key=lambda item: (-item.score, item.person_id))
                best = ranked[0] if ranked else AttackCandidateScore(person_id="", score=0.0, matched_signals=[])
                total = sum(item.score for item in ranked) or 1.0
                results.append(
                    AttackResult(
                        doc_id=pair.doc_id,
                        attacker_type="llm_fallback",
                        top_k=ranked[:5],
                        best_person_id=best.person_id,
                        confidence=round(best.score / total, 4),
                        candidate_pool_size=len(candidate_ids),
                        matched_signals=list(best.matched_signals),
                        metadata={
                            "strategy": strategy,
                            "pair_id": pair.pair_id,
                            "aux_level": pair.aux_knowledge.level,
                            "difficulty": pair.difficulty,
                            "fallback": True,
                        },
                    )
                )
        return results

    for document in documents:
        candidate_ids = document.candidate_pools.insider if document.split == "test_hard" else document.candidate_pools.org_internal
        candidates = [_candidate_payload(characters[candidate_id]) for candidate_id in candidate_ids if candidate_id in characters]
        try:
            response = client.score_candidates(attack_texts[document.doc_id], candidates)
            top_k = []
            for item in response.get("top_k", [])[:5]:
                top_k.append(
                    AttackCandidateScore(
                        person_id=str(item.get("person_id", "")),
                        score=float(item.get("score", 0.0)),
                        matched_signals=list(item.get("matched_signals", [])),
                    )
                )
            if not top_k:
                raise LLMError("llm attacker returned no candidates")
            best = top_k[0]
            confidence = round(best.score / max(1.0, sum(entry.score for entry in top_k)), 4)
            results.append(
                AttackResult(
                    doc_id=document.doc_id,
                    attacker_type="llm",
                    top_k=top_k,
                    best_person_id=best.person_id,
                    confidence=confidence,
                    candidate_pool_size=len(candidate_ids),
                    matched_signals=list(best.matched_signals),
                    metadata={"strategy": strategy, "rationale": response.get("rationale", "")},
                )
            )
        except (LLMError, ValueError, TypeError):
            structured = run_attack([document], characters, strategy)[0]
            structured.attacker_type = "llm_fallback"
            structured.metadata["fallback"] = True
            results.append(structured)
    return results


def run_llm_attack_command(strategy: str) -> None:
    results = _llm_attack(strategy)
    save_attack_results(strategy, "llm", results)

