# -*- coding: utf-8 -*-
"""
LLM Audit Node — Adversarial privacy auditor (RUPTA attacker phase).

Given the anonymized text, the LLM acts as an adversarial attacker and tries
to infer sensitive attributes about the individual.  It returns a privacy_score
(0 = fully anonymous, 100 = easily re-identified) and a list of leaked attributes
with concrete suggestions to fix them.

Uses sentence-level chunking so that even small-context models (8K tokens) can
process long documents.  The final score is the MAX across all chunks (worst case).

Feature flag  : config.json → features.llm_audit
Runtime bypass: state.config.disable_llm = True
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.state import PipelineState
from src.nodes.llm.llm_client import LLMClient, load_full_config, estimate_max_prompt_chars
from src.utils.text_utils import build_chunks

logger = logging.getLogger("LLMAuditNode")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a privacy auditor. Attempt to re-identify the person in the anonymized text.

Check for:
1. Residual direct identifiers (names, phone, email, SSN)
2. Quasi-identifier combinations (age+location+job)
3. Unique/rare attributes (specific disease, niche hobby)
4. Temporal details aiding re-identification

Only flag genuine risks. Respond with ONLY valid JSON.
"""

_USER_PROMPT = """\
Anonymized text chunk:
---
{text}
---

Assess re-identification risk for this chunk. Return ONLY this JSON (no markdown, no explanation):
{{"privacy_score": <0-100>, "leaked_attributes": [{{"attribute": "<name|age|location|occupation|race|health|financial|other>", "evidence": "<exact quote>", "confidence": <0.0-1.0>, "suggestion": "<fix>"}}], "assessment": "<1 sentence>"}}

IMPORTANT: Output ONLY valid JSON."""

# Prompt overhead (system + template without {text})
_PROMPT_OVERHEAD_CHARS = 700


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class LLMAuditNode:
    """
    Adversarial audit node.  Sets ``privacy_score`` and ``llm_feedback`` in the
    pipeline state.

    Long texts are split into sentence chunks; each chunk is audited separately.
    The final ``privacy_score`` is the MAX across all chunks (worst-case).
    Leaked attributes are aggregated from all chunks (de-duplicated by evidence).

    On any failure, defaults to privacy_score=0 (safe → no paraphrase triggered)
    so the pipeline degrades gracefully.
    """

    def __init__(self) -> None:
        self._default_client = LLMClient(role="audit")

    def _client(self, runtime: Dict[str, Any]) -> LLMClient:
        if runtime.get("llm_provider"):
            return LLMClient.create(role="audit", state_config=runtime)
        return self._default_client

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        logger.info("--- Node: LLM Audit ---")

        # --- Feature gate ---
        cfg = load_full_config()
        if not cfg.get("features", {}).get("llm_audit", True):
            logger.info("LLM Audit disabled (feature flag).")
            return {"privacy_score": 0, "llm_feedback": {}}
        runtime = state.get("config", {})
        if runtime.get("llm_audit") is False:
            logger.info("LLM Audit disabled (runtime config).")
            return {"privacy_score": 0, "llm_feedback": {}}
        if runtime.get("disable_llm", False):
            return {"privacy_score": 0, "llm_feedback": {}}

        text: str = state.get("text", "")
        if not text.strip():
            return {"privacy_score": 0, "llm_feedback": {}}

        # --- Build chunks ---
        client = self._client(runtime)
        max_prompt_chars = estimate_max_prompt_chars(client.model, reserved_output_tokens=2048)
        max_text_per_chunk = max(500, max_prompt_chars - _PROMPT_OVERHEAD_CHARS)
        chunks = build_chunks(text, max_text_per_chunk)

        if not chunks:
            return {"privacy_score": 0, "llm_feedback": {}}

        logger.info(
            f"LLM Audit: {len(text)} chars → {len(chunks)} chunk(s) "
            f"(max {max_text_per_chunk} chars/chunk, model={client.model})"
        )

        # --- Build messages for each chunk ---
        message_sets = [
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT.format(
                    text=chunk["text"][:max_text_per_chunk]
                )},
            ]
            for chunk in chunks
        ]

        # --- Send all chunks ---
        responses = client.chat_batch(
            message_sets, temperature=0.1, max_tokens=2048, max_workers=2
        )

        # --- Aggregate results: max score, union of leaked attributes ---
        max_score = 0
        all_leaked: List[Dict] = []
        seen_evidence: set = set()
        assessments: List[str] = []

        for i, raw in enumerate(responses):
            if not raw:
                continue
            parsed = LLMClient.extract_json(raw)
            if not isinstance(parsed, dict):
                logger.debug(f"LLM Audit chunk {i}: unexpected format — {str(raw)[:150]}")
                continue

            # Score
            try:
                score = max(0, min(100, int(parsed.get("privacy_score", 0))))
            except (TypeError, ValueError):
                score = 0
            max_score = max(max_score, score)

            # Leaked attributes (de-duplicate by evidence text)
            for attr in parsed.get("leaked_attributes", []):
                if isinstance(attr, dict):
                    evidence = str(attr.get("evidence", ""))
                    if evidence and evidence not in seen_evidence:
                        seen_evidence.add(evidence)
                        all_leaked.append(attr)

            # Assessment
            assessment = str(parsed.get("assessment", "")).strip()
            if assessment:
                assessments.append(assessment)

        combined_assessment = " | ".join(assessments) if assessments else ""

        logger.info(
            f"LLM Audit: score={max_score}, leaks={len(all_leaked)}, "
            f"chunks_responded={sum(1 for r in responses if r)}/{len(chunks)}"
        )

        return {
            "privacy_score": max_score,
            "llm_feedback": {
                "leaked_attributes": all_leaked,
                "assessment": combined_assessment,
            },
        }
