from __future__ import annotations

import logging
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from src.state import PipelineState
from src.utils.span_utils import merge_entity_lists
from src.utils.entity_utils import normalize_entity_profile, normalize_entity_type
from src.nodes.llm.llm_client import LLMClient, load_full_config
from src.nodes.llm.provider import LLMProvider, get_llm_client

logger = logging.getLogger("LLMReviewNode")


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float = 0.9) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_all(text: str, needle: str, *, case_sensitive: bool = True) -> List[Dict[str, Any]]:
    if not needle:
        return []
    haystack = text if case_sensitive else text.lower()
    target = needle if case_sensitive else needle.lower()
    method = "exact" if case_sensitive else "case_insensitive"
    out: List[Dict[str, Any]] = []
    start = 0
    while True:
        idx = haystack.find(target, start)
        if idx == -1:
            break
        out.append({"start": idx, "end": idx + len(needle), "method": method})
        start = idx + max(1, len(needle))
    return out


def _best_fuzzy_match(text: str, needle: str) -> Optional[Dict[str, Any]]:
    needle = needle.strip()
    if len(needle) < 4 or len(text) > 12000:
        return None
    target = needle.lower()
    n = len(needle)
    delta = max(3, min(12, n // 4))
    best: Optional[Dict[str, Any]] = None
    for size in range(max(2, n - delta), min(len(text), n + delta) + 1):
        max_start = len(text) - size
        for start in range(0, max_start + 1):
            candidate = text[start:start + size]
            ratio = SequenceMatcher(None, candidate.lower(), target).ratio()
            if best is None or ratio > best["score"]:
                best = {
                    "start": start,
                    "end": start + size,
                    "method": "fuzzy",
                    "score": ratio,
                }
    if best and best.get("score", 0.0) >= 0.86:
        return best
    return None


def _validate_llm_entity_offsets(
    ent: Dict[str, Any],
    original_text: str,
) -> List[Dict[str, Any]]:
    ent_text = str(ent.get("text") or ent.get("value") or "").strip()
    evidence = str(ent.get("evidence") or ent.get("context") or "").strip()
    search_text = ent_text or evidence[:120].strip()
    if len(search_text) < 2:
        return []

    start = _as_int(ent.get("start"))
    end = _as_int(ent.get("end"))
    if start is not None and end is not None and 0 <= start < end <= len(original_text):
        observed = original_text[start:end]
        ratio = SequenceMatcher(None, observed.lower(), search_text.lower()).ratio()
        if observed == search_text:
            return [{"start": start, "end": end, "method": "llm_offsets"}]
        if observed.lower() == search_text.lower():
            return [{"start": start, "end": end, "method": "llm_offsets_case_insensitive"}]
        if ratio >= 0.86:
            return [{"start": start, "end": end, "method": "llm_offsets_fuzzy", "score": ratio}]

    matches = _find_all(original_text, search_text, case_sensitive=True)
    if matches:
        return matches

    matches = _find_all(original_text, search_text, case_sensitive=False)
    if matches:
        return matches

    fuzzy = _best_fuzzy_match(original_text, search_text)
    if fuzzy:
        return [fuzzy]

    if evidence and evidence != search_text:
        matches = _find_all(original_text, evidence[:120], case_sensitive=False)
        if matches:
            for match in matches:
                match["method"] = "evidence_context"
            return matches

    return []

def load_prompt(filename: str) -> str:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../prompts", filename))
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class LLMReviewNode:
    """
    Acts as an additive reviewer. It reads the partially anonymized text
    and the original text to find if any sensitive information leaked.
    Adds new found entities to the state.
    """

    def __init__(self) -> None:
        self._default_client = LLMClient(role="detect")
        self.system_prompt = load_prompt("llm_review_system.txt")
        self.user_prompt_template = load_prompt("llm_review_user.txt")

    def _client(self, runtime: Dict[str, Any]) -> "LLMProvider":
        return get_llm_client(role="detect", runtime=runtime, default=self._default_client)

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        logger.info("--- Node: LLM Review (Additive) ---")

        # --- Feature gate ---
        cfg = load_full_config()
        if not cfg.get("features", {}).get("llm_detection", True):
            logger.info("LLM Review disabled (feature flag).")
            return {}
        runtime = state.get("config", {})
        if runtime.get("llm_detection") is False or runtime.get("disable_llm", False):
            logger.info("LLM Review disabled (runtime config).")
            return {}

        original_text: str = state.get("original_text", "")
        anonymized_text: str = state.get("text", "")
        existing_entities: List[Any] = state.get("entities", [])
        entity_profile = normalize_entity_profile(
            runtime.get("entity_profile") or runtime.get("gliner_label_profile")
        )

        if not original_text.strip():
            return {}

        # If it's too long, we might need chunking, but for most bench datasets, full text is optimal for review.
        user_prompt = self.user_prompt_template.format(
            original_text=original_text,
            anonymized_text=anonymized_text
        )

        client = self._client(runtime)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        logger.info(f"Sending review prompt to LLM (approx {len(user_prompt)} chars).")
        try:
            response = client.chat(messages)
            parsed = LLMClient.extract_json(response)
            if isinstance(parsed, list):
                new_entities = parsed
            elif isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
                new_entities = parsed.get("entities", [])
            else:
                snippet = (response or "")[:220].replace("\n", " ")
                logger.warning(
                    "LLM Review returned non-JSON content; ignoring this pass. "
                    f"Response snippet: {snippet}"
                )
                new_entities = []

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            new_entities = []

        # Validate and map back to offsets in original_text
        validated_entities = []
        for ent in new_entities:
            if not isinstance(ent, dict):
                continue
            matches = _validate_llm_entity_offsets(ent, original_text)
            if not matches:
                continue

            raw_type = ent.get("type") or ent.get("label") or ent.get("category") or "QUASI_ID"
            normalized_type = normalize_entity_type(raw_type, profile=entity_profile)
            for match in matches:
                idx = int(match["start"])
                end = int(match["end"])
                validated_entities.append({
                    "start": idx,
                    "end": end,
                    "type": normalized_type,
                    "value": original_text[idx:end],
                    "source": "llm_review",
                    "score": _as_float(match.get("score", ent.get("score", 0.9)), default=0.9),
                    "llm_reason": ent.get("reason") or ent.get("llm_reason", ""),
                    "validation_method": match.get("method", "unknown"),
                })

        logger.info(f"LLM Review found {len(validated_entities)} new valid entity occurrences.")

        if not validated_entities:
            return {}

        merged = merge_entity_lists(existing_entities, validated_entities, resolve_overlapping=True, strategy="priority_longest")
        return {"entities": merged}
