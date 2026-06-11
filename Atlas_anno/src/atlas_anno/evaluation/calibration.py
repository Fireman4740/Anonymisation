from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from atlas_anno.attacks.structured import _score_candidate
from atlas_anno.config import load_config
from atlas_anno.schemas import AttackPair, CharacterProfile, DocumentRecord
from atlas_anno.storage import load_attack_pairs, load_characters, load_documents, save_report


DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_MODES = ("explicit_easy", "explicit_hard", "implicit")


def _dominant_mode(document: DocumentRecord) -> str:
    counts = Counter(document.metadata.get("mention_difficulty", {}))
    if not counts:
        for item in document.metadata.get("surface_grounding", []):
            mode = item.get("difficulty_mode") if isinstance(item, dict) else None
            if mode:
                counts[mode] += 1
    if not counts:
        return "explicit_easy"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _top1_for_pair(document: DocumentRecord, pair: AttackPair, characters: Dict[str, CharacterProfile]) -> bool:
    candidates = [
        _score_candidate(document.text, characters[candidate_id], pair.aux_knowledge.known_attributes)
        for candidate_id in pair.candidate_pool
        if candidate_id in characters
    ]
    ranked = sorted(candidates, key=lambda item: (-item.score, item.person_id))
    return bool(ranked and ranked[0].person_id == pair.target_person_id)


def _rates(counts: Dict[str, List[int]], labels: tuple[str, ...]) -> Dict[str, Dict[str, float]]:
    rates: Dict[str, Dict[str, float]] = {}
    for label in labels:
        values = counts.get(label, [])
        rates[label] = {
            "n": len(values),
            "top1": round(sum(values) / len(values), 4) if values else 0.0,
        }
    return rates


def _monotone(values: Dict[str, Dict[str, float]], order: tuple[str, ...], min_gap: float, strict: bool) -> Dict[str, object]:
    observed = [label for label in order if values.get(label, {}).get("n", 0)]
    if len(observed) < 2:
        return {"passed": True, "skipped": True, "reason": "not enough observed groups"}
    comparisons = []
    passed = True
    for left, right in zip(observed, observed[1:]):
        left_value = float(values[left]["top1"])
        right_value = float(values[right]["top1"])
        ok = left_value >= right_value + min_gap if strict else left_value >= right_value
        comparisons.append({"left": left, "right": right, "left_top1": left_value, "right_top1": right_value, "passed": ok})
        passed = passed and ok
    return {"passed": passed, "skipped": False, "comparisons": comparisons}


def run_difficulty_calibration(
    documents: List[DocumentRecord],
    characters: Dict[str, CharacterProfile],
    pairs: List[AttackPair],
) -> Dict[str, object]:
    config = load_config()
    min_gap = float(config.defaults.get("calibration", {}).get("min_gap", 0.05))
    min_easy_top1 = float(config.defaults.get("calibration", {}).get("min_easy_top1", 0.85))
    documents_by_id = {document.doc_id: document for document in documents}
    by_difficulty: Dict[str, List[int]] = defaultdict(list)
    by_mode: Dict[str, List[int]] = defaultdict(list)
    details = []

    for pair in pairs:
        document = documents_by_id.get(pair.doc_id)
        if document is None:
            continue
        success = int(_top1_for_pair(document, pair, characters))
        difficulty = pair.difficulty or document.scenario.difficulty
        mode = str(pair.metadata.get("difficulty_mode") or _dominant_mode(document))
        by_difficulty[difficulty].append(success)
        by_mode[mode].append(success)
        details.append(
            {
                "pair_id": pair.pair_id,
                "doc_id": pair.doc_id,
                "difficulty": difficulty,
                "difficulty_mode": mode,
                "aux_level": pair.aux_knowledge.level,
                "top1": bool(success),
            }
        )

    difficulty_rates = _rates(by_difficulty, DIFFICULTIES)
    mode_rates = _rates(by_mode, DIFFICULTY_MODES)
    difficulty_gate = _monotone(difficulty_rates, DIFFICULTIES, min_gap, strict=True)
    mode_gate = _monotone(mode_rates, DIFFICULTY_MODES, 0.0, strict=False)
    easy_gate = {
        "passed": difficulty_rates["easy"]["n"] == 0 or difficulty_rates["easy"]["top1"] >= min_easy_top1,
        "top1": difficulty_rates["easy"]["top1"],
        "min_top1": min_easy_top1,
        "n": difficulty_rates["easy"]["n"],
    }
    passed = bool(difficulty_gate["passed"] and mode_gate["passed"] and easy_gate["passed"])
    return {
        "passed": passed,
        "min_gap": min_gap,
        "min_easy_top1": min_easy_top1,
        "by_difficulty": difficulty_rates,
        "by_mode": mode_rates,
        "gates": {
            "difficulty_monotonic": difficulty_gate,
            "mode_monotonic": mode_gate,
            "easy_raw_minimum": easy_gate,
        },
        "details": details,
    }


def run_calibrate_difficulty_command() -> None:
    documents = load_documents(annotated=False)
    if not documents:
        documents = load_documents(annotated=True)
    characters = {character.person_id: character for character in load_characters()}
    report = run_difficulty_calibration(documents, characters, load_attack_pairs())
    save_report("raw", "calibration", report)
