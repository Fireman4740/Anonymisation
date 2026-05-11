# -*- coding: utf-8 -*-
"""
LLM Verification Node — Post-detection entity validation and correction.

Receives the full entity list (regex + NER + LLM detection) and asks the LLM to:
  1. Correct mistyped entities (e.g. ORG labeled as PER)
  2. Extend incomplete spans (e.g. "Jean" → "Jean Dupont")

Design principle: this node NEVER removes entities.  Recall is sacred.
The only allowed actions are keep / retype / extend.

Uses sentence-level chunking so that even small-context models can process long
documents.  Each chunk is verified independently and the results are merged.

Feature flag  : config.json → features.llm_verification
Runtime bypass: state.config.llm_verification = False  or  disable_llm = True
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.state import PipelineState
from src.utils.entity_utils import normalize_entity_profile, normalize_entity_type
from src.nodes.llm.llm_client import LLMClient, load_full_config, estimate_max_prompt_chars
from src.utils.text_utils import build_chunks

logger = logging.getLogger("LLMVerificationNode")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a PII verification expert. Your goal is to IMPROVE detected entities.
You must NEVER remove entities — every detected entity stays.

For each entity, decide ONE action:
- "keep": entity is correct, no change needed (DEFAULT — use this when unsure)
- "retype": entity type is WRONG (provide the correct type)
- "extend": entity span is INCOMPLETE (provide the full correct span)

Guidelines:
- When in doubt, always "keep". False negatives are worse than false positives.
- Partial person names should be EXTENDED to the full name if visible in text
- If a span covers only a first name but the full name is nearby, extend it
- Fix mistyped labels: e.g. an organization labeled PER should be retyped to ORG
- Addresses that include only the city should be extended to the full address if visible

Respond with ONLY a valid JSON array, nothing else."""

_USER_PROMPT = """\
Text:
---
{text}
---

Detected entities to verify:
{entities_json}

For each entity above, return a JSON array of decisions.
Allowed actions: keep, retype, extend. Do NOT remove any entity.
Example: [{{"id":0,"action":"keep"}},{{"id":1,"action":"retype","new_type":"ORG","reason":"is an organization, not a person"}},{{"id":2,"action":"extend","new_start":5,"new_end":18,"new_text":"Jean Dupont","reason":"first name only, full name visible"}}]

IMPORTANT: Output ONLY valid JSON array. No explanation, no markdown."""

# Prompt overhead (system + template without {text}/{entities_json})
_PROMPT_OVERHEAD_CHARS = 1100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entities_in_chunk(
    entities: List[Dict[str, Any]], chunk_offset: int, chunk_end: int
) -> List[Dict[str, Any]]:
    """Return entities that overlap with the given chunk range."""
    return [
        e for e in entities
        if e.get("start", 0) < chunk_end and e.get("end", 0) > chunk_offset
    ]


def _format_entities_for_prompt(
    entities: List[Dict[str, Any]], chunk_offset: int
) -> str:
    """Format entities as a compact JSON-like list for the prompt."""
    lines = []
    for i, e in enumerate(entities):
        start = e.get("start", 0) - chunk_offset
        end = e.get("end", 0) - chunk_offset
        val = e.get("value") or e.get("text") or "?"
        etype = e.get("type", "?")
        source = e.get("source", "?")
        lines.append(
            f'  {{"id":{i},"type":"{etype}","text":"{val}",'
            f'"start":{start},"end":{end},"source":"{source}"}}'
        )
    return "[\n" + ",\n".join(lines) + "\n]"


def _apply_decisions(
    entities: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    full_text: str,
    chunk_offset: int,
    entity_profile: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Apply LLM decisions to the entity list.

    Returns a new list with removes filtered out, retypes applied,
    and extends updated with validated offsets.
    """
    # Build decision map by id
    decision_map: Dict[int, Dict] = {}
    for d in decisions:
        if isinstance(d, dict) and "id" in d:
            try:
                decision_map[int(d["id"])] = d
            except (ValueError, TypeError):
                continue

    result: List[Dict[str, Any]] = []
    for i, entity in enumerate(entities):
        decision = decision_map.get(i)

        if decision is None:
            # No decision → keep as-is
            result.append(entity)
            continue

        action = str(decision.get("action", "keep")).lower().strip()

        if action == "remove":
            # Recall-first policy: never remove entities, treat as keep
            logger.debug(
                f"Verification: ignoring remove for '{entity.get('value')}' "
                f"({entity.get('type')}) — recall-first policy"
            )
            result.append(entity)
            continue

        if action == "retype":
            new_type = decision.get("new_type")
            if new_type and isinstance(new_type, str):
                updated = dict(entity)
                updated["type"] = normalize_entity_type(new_type, profile=entity_profile)
                updated["source"] = "llm_verified"
                logger.debug(
                    f"Verification: retyping '{entity.get('value')}' "
                    f"{entity.get('type')} → {updated['type']}"
                )
                result.append(updated)
            else:
                result.append(entity)
            continue

        if action == "extend":
            new_text = decision.get("new_text", "")
            new_start = decision.get("new_start")
            new_end = decision.get("new_end")

            if new_text and isinstance(new_text, str) and len(new_text) >= 2:
                # Try provided offsets first (relative to chunk, shift to absolute)
                abs_start = None
                abs_end = None
                if isinstance(new_start, int) and isinstance(new_end, int):
                    abs_start = new_start + chunk_offset
                    abs_end = new_end + chunk_offset
                    if abs_end <= len(full_text):
                        actual = full_text[abs_start:abs_end]
                        if actual.lower() != new_text.lower():
                            abs_start = None  # Offsets don't match

                # Fallback: search near original entity
                if abs_start is None:
                    search_start = max(0, entity.get("start", 0) - 50)
                    search_end = min(len(full_text), entity.get("end", 0) + 50)
                    region = full_text[search_start:search_end]
                    idx = region.find(new_text)
                    if idx == -1:
                        idx = region.lower().find(new_text.lower())
                    if idx != -1:
                        abs_start = search_start + idx
                        abs_end = abs_start + len(new_text)

                if abs_start is not None and abs_end is not None:
                    updated = dict(entity)
                    updated["start"] = abs_start
                    updated["end"] = abs_end
                    updated["value"] = full_text[abs_start:abs_end]
                    updated["source"] = "llm_verified"
                    logger.debug(
                        f"Verification: extending '{entity.get('value')}' → "
                        f"'{updated['value']}'"
                    )
                    result.append(updated)
                else:
                    # Couldn't validate extension — keep original
                    result.append(entity)
            else:
                result.append(entity)
            continue

        # "keep" or unknown action → keep as-is
        result.append(entity)

    return result


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class LLMVerificationNode:
    """
    Post-detection verification node.

    Reviews all detected entities and applies corrections (retype, extend spans)
    based on LLM analysis.  This node NEVER removes entities — recall is sacred.

    Feature flag: config.json → features.llm_verification
    Runtime override: state.config.llm_verification = False → skip
    """

    def __init__(self) -> None:
        self._default_client = LLMClient(role="verify")

    def _client(self, runtime: Dict[str, Any]) -> LLMClient:
        if runtime.get("llm_provider"):
            return LLMClient.create(role="verify", state_config=runtime)
        return self._default_client

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        logger.info("--- Node: LLM Verification ---")

        # --- Feature gate ---
        cfg = load_full_config()
        if not cfg.get("features", {}).get("llm_verification", True):
            logger.info("LLM Verification disabled (feature flag).")
            return {}
        runtime = state.get("config", {})
        entity_profile = normalize_entity_profile(
            runtime.get("entity_profile") or runtime.get("gliner_label_profile")
        )
        if runtime.get("llm_verification") is False:
            logger.info("LLM Verification disabled (runtime config).")
            return {}
        if runtime.get("disable_llm", False):
            return {}

        text: str = state.get("original_text", state.get("text", ""))
        entities: List[Dict] = state.get("entities", [])

        if not text.strip() or not entities:
            return {}

        # --- Build chunks ---
        client = self._client(runtime)
        max_prompt_chars = estimate_max_prompt_chars(
            client.model, reserved_output_tokens=2048
        )
        max_text_per_chunk = max(500, max_prompt_chars - _PROMPT_OVERHEAD_CHARS)
        chunks = build_chunks(text, max_text_per_chunk)

        if not chunks:
            return {}

        logger.info(
            f"LLM Verification: {len(entities)} entities, {len(text)} chars → "
            f"{len(chunks)} chunk(s) (model={client.model})"
        )

        # --- Build messages for each chunk that has entities ---
        chunk_entity_map: List[List[Dict]] = []
        message_sets: List[List[Dict[str, str]]] = []
        active_chunk_indices: List[int] = []

        for i, chunk in enumerate(chunks):
            chunk_offset = chunk["offset"]
            chunk_end = chunk_offset + len(chunk["text"])
            chunk_ents = _entities_in_chunk(entities, chunk_offset, chunk_end)

            if not chunk_ents:
                chunk_entity_map.append([])
                continue

            chunk_entity_map.append(chunk_ents)
            active_chunk_indices.append(i)

            entities_json = _format_entities_for_prompt(chunk_ents, chunk_offset)
            user_msg = _USER_PROMPT.format(
                text=chunk["text"][:max_text_per_chunk],
                entities_json=entities_json,
            )
            message_sets.append([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])

        if not message_sets:
            logger.info("LLM Verification: no chunks with entities — skipping")
            return {}

        # --- Send all chunks ---
        responses = client.chat_batch(
            message_sets, temperature=0.0, max_tokens=2048, max_workers=2
        )

        # --- Apply decisions per chunk ---
        # Track which original entities were sent to any chunk (by object id)
        sent_to_chunk: set = set()
        verified_entities: List[Dict] = []

        for resp_i, chunk_idx in enumerate(active_chunk_indices):
            chunk_ents = chunk_entity_map[chunk_idx]
            chunk_offset = chunks[chunk_idx]["offset"]
            raw = responses[resp_i] if resp_i < len(responses) else None

            # Mark all chunk entities as processed
            for e in chunk_ents:
                sent_to_chunk.add(id(e))

            if not raw:
                # LLM didn't respond — keep all chunk entities as-is
                verified_entities.extend(chunk_ents)
                continue

            parsed = LLMClient.extract_json(raw)
            if not isinstance(parsed, list):
                logger.debug(
                    f"LLM Verification chunk {chunk_idx}: "
                    f"unexpected format — {str(raw)[:150]}"
                )
                verified_entities.extend(chunk_ents)
                continue

            # Apply decisions for this chunk's entities
            updated = _apply_decisions(
                chunk_ents, parsed, text, chunk_offset, entity_profile=entity_profile
            )
            verified_entities.extend(updated)

        # Add entities that were not in any processed chunk
        for e in entities:
            if id(e) not in sent_to_chunk:
                verified_entities.append(e)

        retyped = sum(1 for e in verified_entities if e.get("source") == "llm_verified")
        logger.info(
            f"LLM Verification: {len(entities)} → {len(verified_entities)} entities "
            f"({retyped} retyped/extended, {len(verified_entities)} kept)"
        )

        return {"entities": verified_entities}
