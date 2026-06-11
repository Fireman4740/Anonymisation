from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Sequence

from atlas_anno.config import load_config
from atlas_anno.schemas import AttackPair, AuxiliaryKnowledge, CharacterProfile, DocumentRecord
from atlas_anno.storage import load_characters, load_documents, save_attack_pairs


AUX_LEVELS = ("none", "partial", "strong")


def _known_attributes(author: CharacterProfile, level: str) -> Dict[str, Any]:
    if level == "none":
        return {}
    known: Dict[str, Any] = {
        "organization_id": author.organization_id,
        "department": author.department,
    }
    if level == "strong":
        known.update(
            {
                "team": author.team,
                "role": author.role,
            }
        )
        if author.rare_traits:
            known["rare_trait"] = author.rare_traits[0]
    return known


def _target_attributes(author: CharacterProfile) -> Dict[str, Any]:
    return {
        "person_id": author.person_id,
        "full_name": author.full_name,
        "role": author.role,
        "team": author.team,
        "department": author.department,
        "location": author.location,
        "age_range": author.age_range,
        "nationality": author.nationality,
    }


def _dedupe(ids: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(item for item in ids if item))


def _candidate_pool(
    author: CharacterProfile,
    characters: List[CharacterProfile],
    *,
    size: int,
    rng: random.Random,
) -> List[str]:
    same_team = [candidate.person_id for candidate in characters if candidate.person_id != author.person_id and candidate.team == author.team]
    same_role = [candidate.person_id for candidate in characters if candidate.person_id != author.person_id and candidate.role == author.role]
    same_org = [candidate.person_id for candidate in characters if candidate.person_id != author.person_id and candidate.organization_id == author.organization_id]
    others = [candidate.person_id for candidate in characters if candidate.person_id != author.person_id]
    for bucket in (same_team, same_role, same_org, others):
        rng.shuffle(bucket)
    pool = _dedupe([author.person_id] + same_team + same_role + same_org + others)
    return pool[: max(1, min(size, len(pool)))]


def _dominant_difficulty_mode(document: DocumentRecord) -> str:
    counts = Counter(document.metadata.get("mention_difficulty", {}))
    if not counts:
        for item in document.metadata.get("surface_grounding", []):
            if isinstance(item, dict) and item.get("difficulty_mode"):
                counts[str(item["difficulty_mode"])] += 1
    if not counts:
        return "explicit_easy"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_attack_pairs(
    documents: List[DocumentRecord],
    characters: Dict[str, CharacterProfile] | List[CharacterProfile],
    seed: int = 53,
) -> List[AttackPair]:
    character_list = list(characters.values()) if isinstance(characters, dict) else list(characters)
    characters_by_id = {character.person_id: character for character in character_list}
    pool_sizes = load_config().defaults.get("generation", {}).get("attack_pools", {})
    pairs: List[AttackPair] = []
    for document_index, document in enumerate(documents):
        author = characters_by_id.get(document.author_id)
        if author is None:
            continue
        difficulty = str(document.metadata.get("difficulty") or document.scenario.difficulty)
        difficulty_mode = _dominant_difficulty_mode(document)
        pool_size = int(pool_sizes.get(difficulty, pool_sizes.get("medium", 15)))
        for level_index, level in enumerate(AUX_LEVELS):
            rng = random.Random(seed + (document_index * 100) + level_index)
            pair_id = f"{document.doc_id}_{level}"
            pairs.append(
                AttackPair(
                    pair_id=pair_id,
                    doc_id=document.doc_id,
                    target_person_id=author.person_id,
                    target_attributes=_target_attributes(author),
                    aux_knowledge=AuxiliaryKnowledge(level=level, known_attributes=_known_attributes(author, level)),
                    candidate_pool=_candidate_pool(author, character_list, size=pool_size, rng=rng),
                    difficulty=difficulty,
                    metadata={
                        "split": document.split,
                        "difficulty": difficulty,
                        "difficulty_mode": difficulty_mode,
                        "aux_level": level,
                        "source": "attack_pair_builder",
                    },
                )
            )
    return pairs


def run_build_attack_pairs_command() -> None:
    documents = load_documents(annotated=False)
    if not documents:
        documents = load_documents(annotated=True)
    characters = {character.person_id: character for character in load_characters()}
    save_attack_pairs(build_attack_pairs(documents, characters))
