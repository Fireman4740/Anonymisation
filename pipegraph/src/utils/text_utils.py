"""Shared text chunking utilities for LLM nodes."""

import re
from typing import Any, Dict, List


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])|(?:\n\s*\n)')


def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p for p in parts if p and p.strip()]


def build_chunks(text: str, max_chars: int) -> List[Dict[str, Any]]:
    """
    Group sentences into chunks that fit within max_chars.

    Returns list of {"text": str, "offset": int, "separator": str}
    where offset is the chunk's start position in the original text
    and separator is the text between this chunk and the next.
    """
    sentences = split_sentences(text)
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
            chunk_end = current_offset + len(current_text)
            sep = text[chunk_end:idx]
            chunks.append({"text": current_text, "offset": current_offset, "separator": sep})
            current_text = sent
            current_offset = idx

        search_from = idx + len(sent)

    if current_text:
        chunks.append({"text": current_text, "offset": current_offset, "separator": ""})

    return chunks
