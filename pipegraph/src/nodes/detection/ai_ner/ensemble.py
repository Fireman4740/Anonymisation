# -*- coding: utf-8 -*-
"""
NER ensemble utilities (GLiNER only - HF legacy removed)

Provides GLiNER pipelines with GPU optimization support.
All dependencies are optional; if a library is missing the corresponding
function will return an empty list instead of failing.

GPU optimisation:
 - Auto-detect CUDA / MPS (Apple) if PyTorch is available.
 - Move GLiNER models to the best device when possible.
 - Optional float16 on CUDA (configurable) with safe fallback.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from collections import OrderedDict
import gc
import os
import warnings

from src.utils.entity_utils import normalize_entity_profile

# ---------------- Logging ----------------
_DEF_DEBUG = os.getenv("NER_DEBUG", "0").lower() in {"1", "true", "yes"}


def _log(msg: str) -> None:
    if _DEF_DEBUG:
        print(f"[ner_ensemble] {msg}")


# ---------------- Optional imports (guarded) ----------------
try:
    from gliner import GLiNER

    _GLINER_AVAILABLE = True
except Exception:
    _GLINER_AVAILABLE = False
    GLiNER = None

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Optional torch import for device / dtype selection
try:
    import torch
except Exception:
    torch = None

# ---------------- Device & dtype helpers ----------------
_GLN_DEVICE: Optional[Union[str, int]] = None
_HALF_PRECISION = False


def _detect_devices() -> None:
    """
    Detect the best available device (cuda > mps > cpu) and set globals.

    Environment variables:
      NER_FORCE_DEVICE in {"cuda","mps","cpu"} to force a device
      NER_HALF_PRECISION in {"1","true","yes"} to enable float16 on CUDA
    """
    global _GLN_DEVICE, _HALF_PRECISION
    forced = os.getenv("NER_FORCE_DEVICE", "").strip().lower()
    use_half = os.getenv("NER_HALF_PRECISION", "0").lower() in {"1", "true", "yes"}

    # Defaults
    _GLN_DEVICE = "cpu"

    if torch is not None:
        # Forced selection
        if forced == "cuda" and torch.cuda.is_available():
            _GLN_DEVICE = "cuda"
        elif (
            forced == "mps"
            and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            _GLN_DEVICE = "mps"
        elif forced == "cpu":
            _GLN_DEVICE = "cpu"
        # Auto detection when not forced
        elif forced == "":
            if torch.cuda.is_available():
                _GLN_DEVICE = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                _GLN_DEVICE = "mps"

    _HALF_PRECISION = bool(use_half and _GLN_DEVICE == "cuda")


_detect_devices()

# ---------------- Sentence splitting helpers ----------------
_ABBR = {
    # FR
    "m.",
    "mme.",
    "mlle.",
    "dr.",
    "pr.",
    "art.",
    "n°",
    "p. ex.",
    "cf.",
    "etc.",
    # EN
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "art.",
    "no.",
    "nos.",
    "vol.",
    "inc.",
    "jr.",
    "sr.",
    "co.",
    "vs.",
    "st.",
    "e.g.",
    "i.e.",
    "cf.",
    "u.s.",
    "u.k.",
    # months abbr
    "jan.",
    "feb.",
    "mar.",
    "apr.",
    "aug.",
    "sept.",
    "oct.",
    "nov.",
    "dec.",
    "janv.",
    "févr.",
    "avr.",
    "juil.",
    "sept.",
    "oct.",
    "nov.",
    "déc.",
}


def split_sentences(text: str) -> List[Tuple[int, int]]:
    """Split text into sentences with offset spans."""
    spans: List[Tuple[int, int]] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        # Hard break on double newline
        if ch == "\n":
            j = i
            while j < n and text[j] == "\n":
                j += 1
            if j - i >= 2:
                if start < i:
                    spans.append((start, i))
                start = j
                i = j
                continue
        # Soft break on sentence punctuation
        if ch in ".!?":
            j = i - 1
            while j >= start and text[j].isalpha():
                j -= 1
            token = text[j + 1 : i + 1].lower()
            if token in _ABBR:
                i += 1
                continue
            m = i + 1
            while m < n and text[m].isspace():
                m += 1
            if m >= n or (m < n and (text[m].isupper() or text[m] in "\"'(['')")):
                spans.append((start, m))
                start = m
                i = m
                continue
        i += 1
    if start < n:
        spans.append((start, n))
    return [(s, e) for s, e in spans if e - s > 0]


# ---------------- GLiNER presets, labels, options ----------------
_GLINER_PRESETS: Dict[str, List[str]] = {
    "fast": ["urchade/gliner_small-v2.1"],
    "balanced": ["urchade/gliner_medium-v2.1"],
    "accuracy": ["urchade/gliner_large-v2.1", "urchade/gliner_multi-v2.1"],
    "pii": ["urchade/gliner_multi_pii-v1"],
    "multitask": ["knowledgator/gliner-multitask-v1.0"],
    # High-perf combo PII-focused : précision >> généricité
    "best": [
        "urchade/gliner_large-v2.1",      # meilleur NER général,  poids 1.10
        "urchade/gliner_multi_pii-v1",    # spécialisé PII,         poids 1.15
    ],
    # Ensemble complet (plus lent, pour expérimentation)
    "full": [
        "EmergentMethods/gliner_medium_news-v2.1",
        "numind/NuNER_Zero-span",
        "urchade/gliner_large-v2.1",
        "urchade/gliner_multi-v2.1",
    ],
}

# Model weights (higher weight = stronger vote)
_GLINER_MODEL_WEIGHTS: Dict[str, float] = {
    "EmergentMethods/gliner_medium_news-v2.1": 1.25,
    "numind/NuNER_Zero-span": 1.20,
    "urchade/gliner_large-v2.1": 1.10,
    "urchade/gliner_multi-v2.1": 1.05,
    "urchade/gliner_medium-v2.1": 1.00,
    "urchade/gliner_small-v2.1": 0.90,
    "urchade/gliner_multi_pii-v1": 1.15,
    "knowledgator/gliner-multitask-v1.0": 1.05,
}

_GLINER_WEIGHTING = os.getenv("GLINER_WEIGHTING", "1").lower() in {"1", "true", "yes", "weighted"}
_GLINER_PRESET = os.getenv("GLINER_PRESET", "balanced").lower()
if _GLINER_PRESET not in _GLINER_PRESETS:
    _GLINER_PRESET = "balanced"
_GLINER_ONNX = os.getenv("GLINER_ONNX", "0").lower() in {"1", "true", "yes"}
_GLINER_ATTENTION = os.getenv("GLINER_ATTENTION", "").lower()  # "eager" or ""
_GLINER_FORCE_HALF = os.getenv("GLINER_HALF", "0").lower() in {"1", "true", "yes"}

# Helpful label sets
_GLINER_PII_LABELS = [
    "Person",
    "Organization",
    "Location",
    "Email address",
    "Phone number",
    "Address",
    "Bank account number",
    "Credit card number",
    "IBAN",
    "Passport number",
    "Driver's license number",
    "National ID number",
    "Username",
    "IP address",
    "Date",
    "Time",
]

# Extended label list (generic NER + frequent PII)
GLINER_ALL_LABELS: List[str] = [
    # Generic NER
    "Person",
    "Organization",
    "Location",
    "GPE",
    "Facility",
    "Product",
    "Event",
    "Work of Art",
    "Law",
    "Language",
    "Date",
    "Time",
    "Percent",
    "Money",
    "Quantity",
    "Ordinal",
    "Cardinal",
    # PII / Sensitive
    "Email address",
    "URL",
    "IP address",
    "Username",
    "Social media handle",
    "Phone number",
    "Mobile phone number",
    "Landline phone number",
    "Address",
    "Postal code",
    "Bank account number",
    "IBAN",
    "Credit card number",
    "Credit card brand",
    "CVV",
    "CVC",
    "Credit card expiration date",
    "Passport number",
    "Passport expiration date",
    "Driver's license number",
    "Identity card number",
    "National ID number",
    "Tax identification number",
    "Social security number",
    "National health insurance number",
    "Health insurance id number",
    "Health insurance number",
    "Date of birth",
    "Medical condition",
    "Medication",
    "Registration number",
    "Student id number",
    "Insurance number",
    "Insurance company",
    "License plate number",
    "Vehicle registration number",
    "Serial number",
    "Transaction number",
    "Digital signature",
    "Flight number",
    "Train ticket number",
    "Visa number",
    # Country-specific
    "CPF",
    "CNPJ",
]

# ---------------------------------------------------------------------------
# Focused PII-only label set
# Use this instead of GLINER_ALL_LABELS to eliminate false positives on
# generic NER categories (Product, Event, Law, Money, Ordinal, etc.)
# ---------------------------------------------------------------------------
GLINER_PII_FOCUSED_LABELS: List[str] = [
    # Identité
    "Person",
    "Date of birth",
    "Age",
    "Sex",
    "Nationality",
    "Occupation",
    "Race",
    # Contact
    "Email address",
    "Phone number",
    "Mobile phone number",
    "Address",
    "Postal code",
    # Financier
    "Credit card number",
    "IBAN",
    "Bank account number",
    # Documents d'identité
    "Social security number",
    "National ID number",
    "Passport number",
    "Driver's license number",
    # Numérique
    "IP address",
    "Username",
    # Localisation (pour state/city des profils)
    "Location",
    # Attributs démographiques indirects (RAT-Bench indirect identifiers)
    "Marital status",
    "Employment status",
    "Educational background",
    "Citizenship status",
]

# ---------------------------------------------------------------------------
# News / CoNLL-friendly label set.
# Ask GLiNER for generic NER categories first, then project them downstream
# into the CoNLL schema (PER/ORG/LOC/MISC).
# ---------------------------------------------------------------------------
GLINER_NEWS_NER_LABELS: List[str] = [
    "Person",
    "Organization",
    "Location",
    "GPE",
    "Facility",
    "Event",
    "Product",
    "Language",
    "Law",
    "Work of Art",
    "Nationality",
]

GLINER_HYBRID_LABELS: List[str] = sorted(
    set(GLINER_PII_FOCUSED_LABELS) | set(GLINER_NEWS_NER_LABELS)
)


def get_gliner_labels(
    *,
    profile: Optional[str] = None,
    preset: Optional[str] = None,
) -> List[str]:
    norm_profile = normalize_entity_profile(profile)
    if norm_profile in {"news_ner", "conll2003"}:
        return GLINER_NEWS_NER_LABELS
    if norm_profile == "hybrid":
        return GLINER_HYBRID_LABELS
    if preset == "pii":
        return _GLINER_PII_LABELS
    return GLINER_PII_FOCUSED_LABELS


def _normalize_gliner_label(lbl: str) -> Optional[str]:
    """Normalize GLiNER label to uppercase."""
    lab = (lbl or "").strip()
    if not lab:
        return None
    return lab.upper()


# ---------------- Chunk aggregation helper ----------------

def aggregate_chunks(
    spans: List[Tuple[int, int]], text: str, max_chars: int = 400
) -> List[Tuple[int, int]]:
    """
    Aggregate consecutive sentence spans into larger chunks for efficient batch inference.

    Reduces the number of model forward passes by grouping small sentences
    together up to max_chars characters.
    """
    if not spans:
        return []
    result: List[Tuple[int, int]] = []
    cur_start, cur_end = spans[0]
    for s, e in spans[1:]:
        if e - cur_start <= max_chars:
            cur_end = e
        else:
            result.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    result.append((cur_start, cur_end))
    return result


# ---------------- GLiNER ----------------
# Use centralized thread-safe model cache
try:
    from src.utils.model_cache import get_model_cache

    _MODEL_CACHE = get_model_cache("gliner")
except ImportError:
    # Fallback: create a simple inline cache if utils not available
    from collections import OrderedDict

    class _SimpleLRUCache:
        """Minimal fallback LRU cache."""

        def __init__(self, capacity: int = 4):
            self.capacity = capacity
            self._cache: Dict[str, Any] = {}

        def get(self, key: str) -> Optional[Any]:
            return self._cache.get(key)

        def put(self, key: str, model: Any) -> None:
            if len(self._cache) >= self.capacity and key not in self._cache:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self._cache[key] = model

        def __contains__(self, key: str) -> bool:
            return key in self._cache

    _MODEL_CACHE = _SimpleLRUCache(capacity=int(os.getenv("NER_MODEL_CACHE_SIZE", "4")))


def _move_gliner_model_to_device(mdl):
    """Move GLiNER model to best available device."""
    if torch is None:
        return mdl
    try:
        device = _GLN_DEVICE or "cpu"
        if device in {"cuda", "mps"}:
            target = getattr(mdl, "model", mdl)
            if hasattr(target, "to"):
                try:
                    target.to(device)
                except Exception as e:
                    _log(f"GLiNER .to() failed, staying on CPU ({e})")
            if _HALF_PRECISION and device == "cuda":
                try:
                    target.half()
                except Exception:
                    _log("GLiNER half() failed")
    except Exception as e:
        _log(f"GLiNER move error: {e}")
    return mdl


def _load_gliner_models(model_names: List[str]):
    """Load GLiNER models on best device using LRU Cache."""
    if not _GLINER_AVAILABLE:
        return []

    loaded_models = []

    for name in model_names:
        # Check cache first
        cached_model = _MODEL_CACHE.get(name)
        if cached_model:
            loaded_models.append((name, cached_model))
            continue

        # Load new
        kwargs: Dict[str, Any] = {}
        if _GLINER_ATTENTION == "eager":
            kwargs["_attn_implementation"] = "eager"
        if _GLINER_ONNX:
            kwargs["load_onnx_model"] = True

        mdl = None
        try:
            mdl = GLiNER.from_pretrained(name, **kwargs)
        except Exception:
            try:
                mdl = GLiNER.from_pretrained(name)
            except Exception:
                continue

        if mdl:
            mdl = _move_gliner_model_to_device(mdl)
            if (
                torch is not None
                and _GLN_DEVICE == "cuda"
                and (_HALF_PRECISION or _GLINER_FORCE_HALF)
            ):
                try:
                    getattr(mdl, "model", mdl).half()
                except Exception:
                    pass

            _MODEL_CACHE.put(name, mdl)
            loaded_models.append((name, mdl))

    return loaded_models


def run_gliner(
    text: str,
    model_names: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    threshold: float = 0.35,
    device: Optional[str] = None,  # unused; kept for API compatibility
    preset: Optional[str] = None,
    auto_labels: bool = True,
    label_profile: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run one or more GLiNER models and vote across models.

    Args:
        text: Input text
        model_names: List of model names to use
        labels: Entity labels to detect
        threshold: Confidence threshold
        device: (unused, kept for compatibility)
        preset: Preset name from _GLINER_PRESETS
        auto_labels: Use automatic labels if none provided

    Returns:
        List of entities with start, end, entity_group, votes
    """
    if preset is None:
        preset = _GLINER_PRESET
    if model_names is None:
        model_names = _GLINER_PRESETS.get(preset, _GLINER_PRESETS["balanced"])

    if labels is None and auto_labels:
        labels = get_gliner_labels(profile=label_profile, preset=preset)

    if not _GLINER_AVAILABLE:
        return []

    models = _load_gliner_models(model_names)
    if not models:
        return []

    spans = split_sentences(text)
    # Agréger les petites phrases pour réduire le nombre de forward passes
    spans = aggregate_chunks(spans, text, max_chars=400)
    votes: Dict[Tuple[int, int, str], float] = {}

    for name, mdl in models:
        weight = _GLINER_MODEL_WEIGHTS.get(name, 1.0) if _GLINER_WEIGHTING else 1.0

        for s, e in spans:
            chunk = text[s:e]
            try:
                ents = mdl.predict_entities(chunk, labels, threshold=threshold)
            except Exception as e:
                msg = str(e).lower()
                if (
                    any(k in msg for k in ["cuda", "oom", "out of memory"])
                    and torch is not None
                    and _GLN_DEVICE in {"cuda", "mps"}
                ):
                    _log("GLiNER OOM/device error -> trying CPU for this model")
                    try:
                        # Reload on CPU
                        new_mdl = GLiNER.from_pretrained(name)
                        models.append((f"{name}_cpu", new_mdl))
                        ents = new_mdl.predict_entities(chunk, labels, threshold=threshold)
                    except Exception:
                        ents = []
                else:
                    ents = []

            for ent in ents or []:
                start = ent.get("start")
                end = ent.get("end")
                label = ent.get("label") or ent.get("type")
                if (start is None or end is None) and ent.get("text"):
                    idx = chunk.find(ent["text"])
                    if idx != -1:
                        start = idx
                        end = idx + len(ent["text"])
                if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                    continue
                g_label = _normalize_gliner_label(str(label))
                if not g_label:
                    continue
                gs, ge = s + int(start), s + int(end)
                key = (gs, ge, g_label)
                votes[key] = votes.get(key, 0.0) + weight

    results = [
        {"start": s, "end": e, "entity_group": lab, "votes": v} for (s, e, lab), v in votes.items()
    ]
    results.sort(key=lambda x: (x["start"], x["end"]))
    return results


# ---------------- Merge helper ----------------
def merge_ner_lists(*ner_lists) -> List[Dict[str, Any]]:
    """Merge multiple NER lists, removing duplicates."""
    seen = set()
    out = []
    for lst in ner_lists:
        for ent in lst or []:
            s, e = int(ent.get("start", -1)), int(ent.get("end", -1))
            lab = str(ent.get("entity_group", "")).upper()
            if not (0 <= s < e) or not lab:
                continue
            key = (s, e, lab)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "start": s,
                    "end": e,
                    "entity_group": lab,
                    **{k: v for k, v in ent.items() if k not in {"start", "end", "entity_group"}},
                }
            )
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


# ---------------- Warm-up helper ----------------
def warm_up_models(
    gliner_preset: Optional[str] = None,
    gliner_labels: Optional[List[str]] = None,
    gliner_threshold: float = 0.35,
) -> None:
    """
    Preload GLiNER models to reduce cold-start latency.

    Args:
        gliner_preset: one of _GLINER_PRESETS keys (e.g. "balanced", "pii")
        gliner_labels: labels to use for a small GLiNER warm-up call
        gliner_threshold: threshold for the warm-up call
    """
    try:
        preset = gliner_preset or _GLINER_PRESET
        model_names = _GLINER_PRESETS.get(preset, _GLINER_PRESETS["balanced"])
        _load_gliner_models(model_names)
        # Prime internal caches with a tiny run
        if gliner_labels is None:
            gliner_labels = ["Person", "Organization", "Location"]
        _ = run_gliner(
            "Warmup text.",
            model_names=model_names,
            labels=gliner_labels,
            threshold=gliner_threshold,
        )
    except Exception as e:
        _log(f"Warm-up GLiNER failed: {e}")


__all__ = [
    "run_gliner",
    "merge_ner_lists",
    "warm_up_models",
    "aggregate_chunks",
    "GLINER_ALL_LABELS",
    "GLINER_PII_FOCUSED_LABELS",
    "GLINER_NEWS_NER_LABELS",
    "GLINER_HYBRID_LABELS",
    "get_gliner_labels",
    "_GLINER_PRESETS",
    "_GLINER_MODEL_WEIGHTS",
    "_normalize_gliner_label",
    "split_sentences",
]
