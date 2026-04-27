from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from src.state import PipelineState
from src.utils.span_utils import merge_entity_lists
from src.utils.entity_utils import normalize_entity_profile, normalize_entity_type
from src.nodes.llm.llm_client import LLMClient, load_full_config

logger = logging.getLogger("LLMReviewNode")

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

    def _client(self, runtime: Dict[str, Any]) -> LLMClient:
        if runtime.get("llm_provider"):
            return LLMClient.create(role="detect", state_config=runtime)
        return self._default_client

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

        # Validate and map back to exact offsets in original_text
        validated_entities = []
        for ent in new_entities:
            ent_text = ent.get("text", "")
            if not ent_text or len(ent_text) < 2:
                continue
            
            # Find all occurrences in original_text
            start_idx = 0
            while True:
                idx = original_text.find(ent_text, start_idx)
                if idx == -1:
                    # Also try case-insensitive
                    idx = original_text.lower().find(ent_text.lower(), start_idx)
                    
                if idx == -1:
                    break
                    
                # We found one occurrence. Check if it's already masked.
                    # Actually, simply add it to the list. AnonymizationNode handles duplicates nicely!
                    validated_entities.append({
                        "start": idx,
                        "end": idx + len(ent_text),
                        "type": normalize_entity_type(
                            ent.get("type", "QUASI_ID"),
                            profile=entity_profile,
                        ),
                        "value": original_text[idx: idx + len(ent_text)],
                        "source": "llm_review",
                    "score": 0.9,
                    "llm_reason": ent.get("reason", "")
                })
                start_idx = idx + len(ent_text)

        logger.info(f"LLM Review found {len(validated_entities)} new valid entity occurrences.")

        if not validated_entities:
            return {}

        merged = merge_entity_lists(existing_entities, validated_entities, resolve_overlapping=True, strategy="priority_longest")
        return {"entities": merged}
