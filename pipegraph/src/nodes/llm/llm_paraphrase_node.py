# -*- coding: utf-8 -*-
"""
LLM Paraphrase Node — Targeted privacy-preserving rewrite (RUPTA defender phase).

Receives the anonymized text and the leaked attributes identified by the audit
node, then asks the LLM to minimally generalise / rephrase only the problematic
segments.  Increments the `iteration` counter for the RUPTA loop.

Uses sentence-level chunking so that even small-context models can handle long
documents.  Each chunk is paraphrased independently and reassembled.

Feature flag  : config.json → features.llm_paraphrase
Runtime bypass: state.config.disable_llm = True
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.state import PipelineState
from src.nodes.llm.llm_client import LLMClient, load_full_config, estimate_max_prompt_chars

logger = logging.getLogger("LLMParaphraseNode")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a privacy-preserving text editor. Minimally rewrite problematic segments to eliminate re-identification risks.

Rules:
- Generalize: "32 ans" → "début trentaine", "Lyon" → "grande ville française"
- Preserve all non-private content exactly
- Do NOT add new information or change meaning
- Do NOT over-anonymize
- Return the COMPLETE rewritten text chunk, nothing else
"""

_USER_PROMPT = """\
Text chunk to improve:
---
{text}
---

Fix ONLY these residual privacy risks found in this chunk:
{issues}

Intensity: {intensity}

Return ONLY the full rewritten text chunk. No preamble, no explanation, no markdown.\
"""

_INTENSITY_LABELS = {
    1: "minimal — remove only the most obvious identifiers, preserve maximum utility",
    2: "moderate — generalise quasi-identifiers that combine to create re-identification risk",
    3: "aggressive — heavily generalise or remove anything that could aid re-identification",
}

# Prompt overhead (system + template without {text})
_PROMPT_OVERHEAD_CHARS = 800


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])|(?:\n\s*\n)', text)
    return [p for p in parts if p and p.strip()]


def _build_chunks(text: str, max_chars: int) -> List[Dict[str, Any]]:
    """
    Group sentences into chunks with their offset in the original text.
    Returns list of {"text": ..., "offset": ..., "separator": ...}
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [{"text": text, "offset": 0, "separator": ""}] if text.strip() else []

    chunks: List[Dict[str, Any]] = []
    current_text = ""
    current_offset = 0
    search_from = 0

    for sent in sentences:
        idx = text.find(sent, search_from)
        if idx == -1:
            idx = search_from

        if not current_text:
            current_text = sent
            current_offset = idx
        elif len(current_text) + len(sent) + 1 <= max_chars:
            gap = text[current_offset + len(current_text):idx]
            current_text += gap + sent
        else:
            # Capture the separator between this chunk and the next
            chunk_end = current_offset + len(current_text)
            sep = text[chunk_end:idx]
            chunks.append({"text": current_text, "offset": current_offset, "separator": sep})
            current_text = sent
            current_offset = idx

        search_from = idx + len(sent)

    if current_text:
        chunks.append({"text": current_text, "offset": current_offset, "separator": ""})

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_issues(leaked_attrs: List[Dict]) -> str:
    """Format leaked attributes for the prompt."""
    if not leaked_attrs:
        return "(none)"
    lines = []
    for attr in leaked_attrs:
        attribute = attr.get("attribute", "?")
        evidence = attr.get("evidence", "?")
        suggestion = attr.get("suggestion", "generalise or remove")
        confidence = float(attr.get("confidence", 1.0))
        lines.append(
            f"- [{attribute}] "
            f"Evidence: \"{evidence}\" "
            f"→ {suggestion} "
            f"(confidence {confidence:.0%})"
        )
    return "\n".join(lines)


def _find_relevant_leaks(
    leaked_attrs: List[Dict], chunk_text: str
) -> List[Dict]:
    """Return only leaked attributes whose evidence appears in this chunk."""
    relevant = []
    chunk_lower = chunk_text.lower()
    for attr in leaked_attrs:
        evidence = str(attr.get("evidence", "")).strip()
        if evidence and evidence.lower() in chunk_lower:
            relevant.append(attr)
    return relevant


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class LLMParaphraseNode:
    """
    Privacy-preserving paraphrase node.

    Rewrites/generalises the leaked zones identified by LLMAuditNode, then
    increments `iteration` so the RUPTA router can track progress.

    Long texts are split into sentence chunks; only chunks containing leaked
    evidence are sent to the LLM.  Others are kept as-is.
    """

    def __init__(self) -> None:
        self._default_client = LLMClient(role="paraphrase")

    def _client(self, runtime: Dict[str, Any]) -> LLMClient:
        if runtime.get("llm_provider"):
            return LLMClient.create(role="paraphrase", state_config=runtime)
        return self._default_client

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        logger.info("--- Node: LLM Paraphrase ---")

        # --- Increment iteration regardless of outcome ---
        iteration: int = state.get("iteration", 0) + 1

        # --- Feature gate ---
        cfg = load_full_config()
        if not cfg.get("features", {}).get("llm_paraphrase", True):
            logger.info("LLM Paraphrase disabled (feature flag).")
            return {"iteration": iteration}
        runtime = state.get("config", {})
        if runtime.get("llm_paraphrase") is False:
            logger.info("LLM Paraphrase disabled (runtime config).")
            return {"iteration": iteration}
        if runtime.get("disable_llm", False):
            return {"iteration": iteration}

        text: str = state.get("text", "")
        llm_feedback: Dict = state.get("llm_feedback", {})
        leaked_attrs: List[Dict] = llm_feedback.get("leaked_attributes", [])

        # Only address high-confidence leaks
        significant = [a for a in leaked_attrs if float(a.get("confidence", 0)) >= 0.5]

        if not significant:
            logger.info("LLM Paraphrase: no significant leaks — skipping rewrite")
            return {"iteration": iteration}

        # --- Intensity from config ---
        policy = cfg.get("policy_defaults", {})
        intensity = max(1, min(3, int(policy.get("paraphrase_intensity", 1))))
        intensity_label = _INTENSITY_LABELS[intensity]

        # --- Build chunks ---
        client = self._client(runtime)
        max_prompt_chars = estimate_max_prompt_chars(client.model, reserved_output_tokens=4096)
        max_text_per_chunk = max(500, max_prompt_chars - _PROMPT_OVERHEAD_CHARS)
        chunks = _build_chunks(text, max_text_per_chunk)

        if not chunks:
            return {"iteration": iteration}

        # --- Identify which chunks need rewriting ---
        chunks_to_rewrite: List[int] = []
        chunk_issues: List[List[Dict]] = []
        for i, chunk in enumerate(chunks):
            relevant = _find_relevant_leaks(significant, chunk["text"])
            chunk_issues.append(relevant)
            if relevant:
                chunks_to_rewrite.append(i)

        if not chunks_to_rewrite:
            logger.info("LLM Paraphrase: no leaked evidence found in any chunk — skipping")
            return {"iteration": iteration}

        logger.info(
            f"LLM Paraphrase: {len(chunks)} chunks, "
            f"{len(chunks_to_rewrite)} need rewriting "
            f"(model={client.model})"
        )

        # --- Build messages only for chunks that need rewriting ---
        message_sets = []
        for idx in chunks_to_rewrite:
            user_msg = _USER_PROMPT.format(
                text=chunks[idx]["text"][:max_text_per_chunk],
                issues=_format_issues(chunk_issues[idx]),
                intensity=f"{intensity} — {intensity_label}",
            )
            message_sets.append([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])

        responses = client.chat_batch(
            message_sets, temperature=0.15, max_tokens=4096, max_workers=2
        )

        # --- Reassemble the full text ---
        rewritten_chunks: List[Optional[str]] = [None] * len(chunks)
        for resp_i, chunk_idx in enumerate(chunks_to_rewrite):
            raw = responses[resp_i] if resp_i < len(responses) else None
            if raw and raw.strip():
                rewritten = raw.strip()
                original_len = len(chunks[chunk_idx]["text"])
                # Sanity check: rewrite shouldn't be drastically shorter
                if len(rewritten) >= original_len * 0.3:
                    rewritten_chunks[chunk_idx] = rewritten
                else:
                    logger.warning(
                        f"LLM Paraphrase chunk {chunk_idx}: response too short "
                        f"({len(rewritten)} vs {original_len}) — keeping original"
                    )

        # Build final text
        parts: List[str] = []
        for i, chunk in enumerate(chunks):
            if rewritten_chunks[i] is not None:
                parts.append(rewritten_chunks[i])  # type: ignore[arg-type]
            else:
                parts.append(chunk["text"])
            # Add separator to next chunk
            if i < len(chunks) - 1:
                parts.append(chunk["separator"])

        new_text = "".join(parts)

        # Final sanity check on the full reassembled text
        if len(new_text) < len(text) * 0.4:
            logger.warning(
                f"LLM Paraphrase: reassembled text suspiciously short "
                f"({len(new_text)} vs {len(text)}) — discarding"
            )
            return {"iteration": iteration}

        rewritten_count = sum(1 for r in rewritten_chunks if r is not None)
        logger.info(
            f"LLM Paraphrase: iteration={iteration}, "
            f"rewrote {rewritten_count}/{len(chunks)} chunks, "
            f"fixed {len(significant)} leak(s), "
            f"length {len(text)} → {len(new_text)}"
        )

        return {
            "text": new_text,
            "iteration": iteration,
        }
