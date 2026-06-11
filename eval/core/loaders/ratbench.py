"""
RAT-Bench dataset loader.

Downloads and caches the RAT-Bench dataset from HuggingFace, then converts
profiles into the (doc_id, text, gt_spans) format used by the pipegraph
evaluation harness.

RAT-Bench profile schema:
  - id: int
  - profile: dict of all attributes
  - direct_identifiers: dict (name, SSN, email, phone, credit card, address)
  - indirect_identifiers: dict (race, sex, DOB, occupation, marital status …)
  - features: list[str]
  - difficulty: int (1, 2, 3)
  - scenario: str
  - text: str  ← the text to anonymize
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eval.core.metrics import normalize_spans

logger = logging.getLogger("RAT-Bench-Loader")

Span = Tuple[int, int, str]

# ---------------------------------------------------------------------------
# Mapping from RAT-Bench attribute keys to normalised PII labels
# ---------------------------------------------------------------------------

DIRECT_ID_LABEL_MAP: Dict[str, str] = {
    "name": "PERSON",
    "email": "EMAIL",
    "phone number": "PHONE",
    "SSN": "SSN",
    "credit card number": "CREDIT_CARD",
    "address": "ADDRESS",
}

INDIRECT_ID_LABEL_MAP: Dict[str, str] = {
    # Human-readable profile keys
    "sex": "DEMOGRAPHIC",
    "race": "DEMOGRAPHIC",
    "citizenship status": "DEMOGRAPHIC",
    "educational attainment": "DEMOGRAPHIC",
    "state of residence": "LOCATION",
    "occupation": "DEMOGRAPHIC",
    "marital status": "DEMOGRAPHIC",
    "employment status": "DEMOGRAPHIC",
    "date of birth": "DATE",
    "zip code": "LOCATION",
    # PUMS code keys (from indirect_identifiers dict)
    "SEX": "DEMOGRAPHIC",
    "RAC2P": "DEMOGRAPHIC",
    "CIT": "DEMOGRAPHIC",
    "SCHL": "DEMOGRAPHIC",
    "ST": "LOCATION",
    "OCCP": "DEMOGRAPHIC",
    "MAR": "DEMOGRAPHIC",
    "ESR": "DEMOGRAPHIC",
    "DOB": "DATE",
    "DOB-Day": "DATE",
    "DOB-Month": "DATE",
    "DOB-Year": "DATE",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_all_occurrences(text: str, needle: str) -> List[Tuple[int, int]]:
    """Find all exact occurrences of *needle* in *text*."""
    if not needle or not text:
        return []
    out: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        out.append((idx, idx + len(needle)))
        start = idx + 1
    return out


def _find_all_occurrences_ci(text: str, needle: str) -> List[Tuple[int, int]]:
    """Case-insensitive occurrence search."""
    if not needle or not text:
        return []
    lt = text.lower()
    ln = needle.lower()
    out: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = lt.find(ln, start)
        if idx == -1:
            break
        out.append((idx, idx + len(needle)))
        start = idx + 1
    return out


def _fuzzy_name_occurrences(text: str, full_name: str) -> List[Tuple[int, int]]:
    """Match 'First … Last' with a bounded gap (handles middle names)."""
    parts = [p for p in full_name.split() if p]
    if len(parts) < 2:
        return _find_all_occurrences_ci(text, full_name)
    first, last = parts[0], parts[-1]
    max_gap = 80
    pattern = re.compile(
        rf"(?<!\w){re.escape(first)}(?!\w).{{0,{max_gap}}}?(?<!\w){re.escape(last)}(?!\w)",
        re.IGNORECASE,
    )
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _normalize_number_with_spaces(text: str, value: str) -> List[Tuple[int, int]]:
    """Match a number that may appear with spaces/dashes in the text.

    E.g. profile has ``"38749649909130"`` but text contains
    ``"3874 9649 9091 30"`` or ``"3874-9649-9091-30"``.
    """
    digits = re.sub(r"\D", "", value)
    if len(digits) < 6:
        return []
    # Build a pattern that allows optional spaces/dashes between each digit
    pat = r"[\s\-]?".join(re.escape(d) for d in digits)
    return [(m.start(), m.end()) for m in re.finditer(pat, text)]


_MONTH_NAMES = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def _fuzzy_date_occurrences(text: str, value: str) -> List[Tuple[int, int]]:
    """Match dates across common format variations.

    Handles ``"2 October 1994"`` vs ``"October 2, 1994"`` etc.
    """
    # Try to extract day, month-name, year from the value
    m = re.match(
        r"(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})",
        value.strip(),
        re.IGNORECASE,
    )
    if not m:
        m = re.match(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})",
            value.strip(),
            re.IGNORECASE,
        )
        if m:
            month_name, day_str, year_str = m.group(1), m.group(2), m.group(3)
        else:
            return []
    else:
        day_str, month_name, year_str = m.group(1), m.group(2), m.group(3)

    day = int(day_str)
    mn = month_name.lower()

    # Build patterns for common date formats containing these components
    patterns = [
        # "October 2, 1994" / "October 2 1994"
        rf"(?<!\w){re.escape(month_name)}\s+{day}\s*,?\s*{re.escape(year_str)}(?!\w)",
        # "2 October 1994"
        rf"(?<!\w){day}\s+{re.escape(month_name)}\s+{re.escape(year_str)}(?!\w)",
        # "2nd October 1994" etc.
        rf"(?<!\w){day}(?:st|nd|rd|th)\s+{re.escape(month_name)}\s+{re.escape(year_str)}(?!\w)",
        # "10/02/1994" or "02/10/1994" (ambiguous but try both)
        rf"(?<!\d){_MONTH_NAMES.get(mn, '00')}/{day:02d}/{re.escape(year_str)}(?!\d)",
        rf"(?<!\d){day:02d}/{_MONTH_NAMES.get(mn, '00')}/{re.escape(year_str)}(?!\d)",
    ]
    results: List[Tuple[int, int]] = []
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            results.append((match.start(), match.end()))
    return results


def _fuzzy_singular_plural(text: str, value: str) -> List[Tuple[int, int]]:
    """Match a value allowing singular/plural and case differences.

    E.g. ``"MEDICAL ASSISTANTS"`` matches ``"medical assistant"`` and vice versa.
    """
    v = value.strip()
    if len(v) < 4:
        return []
    # Generate singular/plural variants
    variants = {v.lower()}
    if v.lower().endswith("s"):
        variants.add(v.lower()[:-1])
    else:
        variants.add(v.lower() + "s")
    if v.lower().endswith("es"):
        variants.add(v.lower()[:-2])

    results: List[Tuple[int, int]] = []
    text_lower = text.lower()
    for var in variants:
        start = 0
        while True:
            idx = text_lower.find(var, start)
            if idx == -1:
                break
            results.append((idx, idx + len(var)))
            start = idx + 1
    return results


def _spans_for_value(text: str, value: str, label: str) -> List[Span]:
    """Locate *value* in *text* and return labeled spans.

    Strategy (in order):
      1. Exact match
      2. Case-insensitive match
      3. Label-specific fuzzy matchers:
         - PERSON → name with gap tolerance
         - CREDIT_CARD / SSN / PHONE → digits with optional spaces/dashes
         - DATE → cross-format date matching
      4. Singular/plural fallback
    """
    value_str = str(value).strip()
    if not value_str or value_str.lower() in ("none", "n/a", ""):
        return []

    occ = _find_all_occurrences(text, value_str)
    if not occ:
        occ = _find_all_occurrences_ci(text, value_str)
    if not occ and label == "PERSON":
        occ = _fuzzy_name_occurrences(text, value_str)
    if not occ and label in ("CREDIT_CARD", "SSN", "PHONE"):
        occ = _normalize_number_with_spaces(text, value_str)
    if not occ and label == "DATE":
        occ = _fuzzy_date_occurrences(text, value_str)
    if not occ:
        occ = _fuzzy_singular_plural(text, value_str)

    return [(s, e, label) for s, e in occ]


def _dedupe_spans(spans: Iterable[Span]) -> List[Span]:
    """Remove duplicates and sort."""
    return normalize_spans(spans)


# ---------------------------------------------------------------------------
# Ground-truth span extraction from RAT-Bench profiles
# ---------------------------------------------------------------------------

def gt_spans_from_ratbench_profile(profile: Dict[str, Any]) -> List[Span]:
    """
    Extract ground-truth spans by locating each known PII value in the text.

    RAT-Bench does not provide character offsets — only the attribute values.
    We search for each direct & indirect identifier value inside the text
    and return (start, end, label) spans.
    """
    text = str(profile.get("text", ""))
    if not text:
        return []

    spans: List[Span] = []

    # 1) Direct identifiers — high priority
    direct_ids: Dict[str, Any] = profile.get("direct_identifiers") or {}
    for attr_key, attr_value in direct_ids.items():
        label = DIRECT_ID_LABEL_MAP.get(attr_key, "SENSITIVE")
        spans.extend(_spans_for_value(text, attr_value, label))

    # 2) Indirect identifiers — contextual/quasi-identifiers
    indirect_ids: Dict[str, Any] = profile.get("indirect_identifiers") or {}
    for attr_key, attr_value in indirect_ids.items():
        label = INDIRECT_ID_LABEL_MAP.get(attr_key, "SENSITIVE")
        val_str = str(attr_value).strip()

        # Skip coded values that won't appear verbatim (e.g. PUMS codes)
        if not val_str or val_str.lower() in ("none", "n/a"):
            continue

        # For occupation, try the description part if it contains TYPE:/DESCRIPTION:
        if attr_key in ("occupation", "OCCP") and "DESCRIPTION" in val_str:
            dloc = val_str.find("DESCRIPTION")
            desc = val_str[dloc + 13:].strip().rstrip(")")
            if desc:
                spans.extend(_spans_for_value(text, desc, "OCCUPATION"))
            continue

        spans.extend(_spans_for_value(text, val_str, label))

    return _dedupe_spans(spans)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    """Return the cache directory for RAT-Bench data."""
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(eval_dir, "datasets", "RAT-Bench", "cache")


def download_ratbench(
    language: str = "english",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Download the RAT-Bench dataset from HuggingFace and cache it locally.

    Returns the path to the cached JSON file.
    """
    cache = cache_dir or _cache_dir()
    os.makedirs(cache, exist_ok=True)

    cached_path = os.path.join(cache, f"ratbench_{language}.json")
    if os.path.exists(cached_path) and not force_download:
        logger.info(f"Using cached RAT-Bench ({language}): {cached_path}")
        return cached_path

    logger.info(f"Downloading RAT-Bench dataset ({language}) from HuggingFace...")
    records: List[Dict[str, Any]] = []

    # Attempt 1: standard load_dataset (generates local Arrow cache)
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("imperial-cpg/rat-bench", language, split="train")
        records = [dict(row) for row in ds]  # type: ignore
        logger.info(f"Loaded {len(records)} records via load_dataset")
    except Exception as e:
        logger.warning(f"load_dataset failed: {e}")

    # Attempt 2: streaming mode — bypasses Arrow generation, reads records directly
    if not records:
        try:
            from datasets import load_dataset  # type: ignore
            logger.warning("Retrying with streaming=True to bypass Arrow generation errors...")
            ds = load_dataset("imperial-cpg/rat-bench", language, split="train", streaming=True)
            records = [dict(row) for row in ds]  # type: ignore
            logger.warning(f"Loaded {len(records)} records via streaming")
        except Exception as e:
            logger.warning(f"Streaming load_dataset also failed: {e}")

    # Attempt 3: direct HuggingFace datasets server API (paginated)
    if not records:
        logger.warning("Falling back to direct URL download...")
        records = _download_via_url(language)

    if not records:
        raise RuntimeError(
            f"All download attempts failed for RAT-Bench language={language}. "
            "Check network connectivity and HuggingFace availability."
        )

    with open(cached_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(f"Cached {len(records)} profiles to {cached_path}")
    return cached_path


def _download_via_url(language: str) -> List[Dict[str, Any]]:
    """Fallback: download via HuggingFace datasets server API with pagination."""
    import urllib.request

    base_url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset=imperial-cpg/rat-bench&config={language}&split=train"
    )
    PAGE_SIZE = 100
    records: List[Dict[str, Any]] = []
    offset = 0

    while True:
        url = f"{base_url}&offset={offset}&length={PAGE_SIZE}"
        req = urllib.request.Request(url, headers={"User-Agent": "Anonymisation-Eval/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        rows = data.get("rows", [])
        if not rows and "error" in data:
            raise RuntimeError(f"HuggingFace datasets server error: {data['error']}")
        if not rows:
            break
        records.extend(row.get("row", row) for row in rows)

        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    logger.info(f"Downloaded {len(records)} records via API for language={language}")
    return records


def load_ratbench_profiles(
    language: str = "english",
    level: Optional[int] = None,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load RAT-Bench profiles, optionally filtered by difficulty level.

    Args:
        language: Dataset language ('english', 'mandarin', 'spanish')
        level: Filter by difficulty level (1, 2, or 3). None = all levels.
        cache_dir: Override cache directory.
        limit: Maximum number of profiles to return.

    Returns:
        List of profile dicts.
    """
    cached_path = download_ratbench(language=language, cache_dir=cache_dir)

    with open(cached_path, "r", encoding="utf-8") as f:
        profiles: List[Dict[str, Any]] = json.load(f)

    if level is not None:
        profiles = [p for p in profiles if p.get("difficulty") == level]

    if limit is not None:
        profiles = profiles[:limit]

    logger.info(
        f"Loaded {len(profiles)} RAT-Bench profiles "
        f"(language={language}, level={level}, limit={limit})"
    )
    return profiles


# ---------------------------------------------------------------------------
# Build docs tuple compatible with eval.core.pipeline.build_report
# ---------------------------------------------------------------------------

def build_docs_from_ratbench(
    language: str = "english",
    level: Optional[int] = None,
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Tuple[str, str, List[Span]]]:
    """
    Build evaluation documents from RAT-Bench.

    Returns:
        List of (doc_id, text, ground_truth_spans) tuples compatible
        with the existing pipegraph evaluation harness.
    """
    profiles = load_ratbench_profiles(
        language=language, level=level, cache_dir=cache_dir, limit=limit
    )

    docs: List[Tuple[str, str, List[Span]]] = []
    skipped = 0

    for profile in profiles:
        doc_id = f"ratbench_{profile.get('id', len(docs))}_L{profile.get('difficulty', '?')}"
        text = str(profile.get("text", ""))
        if not text.strip():
            skipped += 1
            continue

        gt_spans = gt_spans_from_ratbench_profile(profile)
        docs.append((doc_id, text, gt_spans))

    if skipped:
        logger.warning(f"Skipped {skipped} profiles with empty text")

    logger.info(
        f"Built {len(docs)} evaluation docs from RAT-Bench "
        f"(language={language}, level={level})"
    )
    return docs


def get_ratbench_metadata(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract RAT-Bench-specific metadata for enriched reporting.

    Returns dict with scenario, difficulty, feature list, and identifier counts.
    """
    return {
        "difficulty": profile.get("difficulty"),
        "scenario": profile.get("scenario"),
        "features": profile.get("features", []),
        "n_direct_ids": len(profile.get("direct_identifiers") or {}),
        "n_indirect_ids": len(profile.get("indirect_identifiers") or {}),
        "direct_id_types": list((profile.get("direct_identifiers") or {}).keys()),
        "indirect_id_types": list((profile.get("indirect_identifiers") or {}).keys()),
    }


# ---------------------------------------------------------------------------
# Text-leak evaluation (RAT-Bench native philosophy)
# ---------------------------------------------------------------------------

def evaluate_text_leaks(
    original_text: str,
    anonymized_text: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate how many PII values from the profile still leak in the
    anonymized text.  This is complementary to span-based evaluation
    and aligned with RAT-Bench's re-identification risk philosophy.

    For each identifier (direct + indirect), we check if its value
    (or a significant fragment) appears in the anonymized output.

    Returns:
        Dict with per-attribute leak status and aggregate leak rate.
    """
    anon_lower = anonymized_text.lower()
    results: Dict[str, Dict[str, Any]] = {}

    # Check direct identifiers
    direct_ids: Dict[str, Any] = profile.get("direct_identifiers") or {}
    for attr_key, attr_value in direct_ids.items():
        val = str(attr_value).strip()
        if not val or val.lower() in ("none", "n/a"):
            continue
        leaked = val.lower() in anon_lower
        # For names, also check individual parts
        if not leaked and attr_key == "name":
            parts = [p for p in val.split() if len(p) > 2]
            leaked = any(p.lower() in anon_lower for p in parts)
        results[attr_key] = {
            "value": val,
            "category": "direct",
            "leaked": leaked,
        }

    # Check indirect identifiers
    indirect_ids: Dict[str, Any] = profile.get("indirect_identifiers") or {}
    for attr_key, attr_value in indirect_ids.items():
        val = str(attr_value).strip()
        if not val or val.lower() in ("none", "n/a"):
            continue
        # For complex coded values (e.g. occupation with TYPE:/DESCRIPTION:),
        # only check the descriptive part
        if "DESCRIPTION" in val:
            dloc = val.find("DESCRIPTION")
            val = val[dloc + 13:].strip().rstrip(")")
        if len(val) < 3:
            continue
        leaked = val.lower() in anon_lower
        results[attr_key] = {
            "value": val,
            "category": "indirect",
            "leaked": leaked,
        }

    n_total = len(results)
    n_leaked = sum(1 for r in results.values() if r["leaked"])
    n_direct = sum(1 for r in results.values() if r["category"] == "direct")
    n_direct_leaked = sum(
        1 for r in results.values() if r["category"] == "direct" and r["leaked"]
    )
    n_indirect = sum(1 for r in results.values() if r["category"] == "indirect")
    n_indirect_leaked = sum(
        1 for r in results.values() if r["category"] == "indirect" and r["leaked"]
    )

    return {
        "leak_rate": round(n_leaked / n_total, 4) if n_total > 0 else 0.0,
        "direct_leak_rate": round(n_direct_leaked / n_direct, 4) if n_direct > 0 else 0.0,
        "indirect_leak_rate": round(n_indirect_leaked / n_indirect, 4) if n_indirect > 0 else 0.0,
        "n_total_attributes": n_total,
        "n_leaked": n_leaked,
        "n_protected": n_total - n_leaked,
        "details": results,
    }
