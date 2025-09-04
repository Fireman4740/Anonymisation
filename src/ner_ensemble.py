# -*- coding: utf-8 -*-
"""
NER ensemble utilities extracted from orchestrator.

Provides optional DeepPavlov, GLiNER and HF (transformers) pipelines plus
helpers for sentence splitting and entity list merging.

All dependencies are optional; if a library is missing the corresponding
function will return an empty list instead of failing.

GPU optimisation:
 - Auto-detect CUDA / MPS (Apple) if PyTorch is available.
 - Move HF NER pipeline and GLiNER models to the best device when possible.
 - Optional float16 on CUDA (configurable) with safe fallback.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Union
import os
import traceback
from contextlib import nullcontext as _nullcontext

# ---------------- Logging ----------------
_DEF_DEBUG = os.getenv("NER_DEBUG", "0").lower() in {"1", "true", "yes"}


def _log(msg: str) -> None:
    if _DEF_DEBUG:
        print(f"[ner_ensemble] {msg}")


# ---------------- Optional imports (guarded) ----------------
try:  # pragma: no cover
    from deeppavlov import configs as dp_configs  # type: ignore
    from deeppavlov import build_model as dp_build_model  # type: ignore

    _DP_AVAILABLE = True
except Exception:  # pragma: no cover
    _DP_AVAILABLE = False  # type: ignore
    dp_configs = None  # type: ignore
    dp_build_model = None  # type: ignore

try:  # pragma: no cover
    from gliner import GLiNER  # type: ignore

    _GLINER_AVAILABLE = True
except Exception:  # pragma: no cover
    _GLINER_AVAILABLE = False  # type: ignore
    GLiNER = None  # type: ignore

try:  # pragma: no cover
    from transformers import (  # type: ignore
        pipeline as hf_pipeline,
        AutoTokenizer as HFTokenizer,
        AutoModelForTokenClassification as HFModel,
    )
except Exception:  # pragma: no cover
    hf_pipeline = HFTokenizer = HFModel = None  # type: ignore

# Optional torch import for device / dtype selection
try:  # pragma: no cover
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# ---------------- Device & dtype helpers ----------------
_DEF_DEVICE: Union[int, str] = -1  # HF pipeline device id; -1 for CPU
_GLN_DEVICE: Optional[Union[str, int]] = None
_HALF_PRECISION = False


def _detect_devices() -> None:
    """
    Detect the best available device (cuda > mps > cpu) and set globals.

    Environment variables:
      NER_FORCE_DEVICE in {"cuda","mps","cpu"} to force a device
      NER_HALF_PRECISION in {"1","true","yes"} to enable float16 on CUDA
    """
    global _DEF_DEVICE, _GLN_DEVICE, _HALF_PRECISION
    forced = os.getenv("NER_FORCE_DEVICE", "").strip().lower()
    use_half = os.getenv("NER_HALF_PRECISION", "0").lower() in {"1", "true", "yes"}

    # Defaults
    _DEF_DEVICE, _GLN_DEVICE = -1, "cpu"

    if torch is not None:
        # Forced selection
        if forced == "cuda" and torch.cuda.is_available():
            _DEF_DEVICE, _GLN_DEVICE = 0, "cuda"
        elif (
            forced == "mps"
            and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            _DEF_DEVICE, _GLN_DEVICE = 0, "mps"
        elif forced == "cpu":
            _DEF_DEVICE, _GLN_DEVICE = -1, "cpu"
        # Auto detection when not forced
        elif forced == "":
            if torch.cuda.is_available():
                _DEF_DEVICE, _GLN_DEVICE = 0, "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                _DEF_DEVICE, _GLN_DEVICE = 0, "mps"

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
            if m >= n or (m < n and (text[m].isupper() or text[m] in "\"'([“”‘’)")):
                spans.append((start, m))
                start = m
                i = m
                continue
        i += 1
    if start < n:
        spans.append((start, n))
    return [(s, e) for s, e in spans if e - s > 0]


# ---------------- GLiNER presets, labels, options ----------------
# Presets map to model checkpoints (priority order = higher vote weight first)
_GLINER_PRESETS: Dict[str, List[str]] = {
    "fast": ["urchade/gliner_small-v2.1"],
    "balanced": ["urchade/gliner_medium-v2.1"],
    "accuracy": ["urchade/gliner_large-v2.1", "urchade/gliner_multi-v2.1"],
    "pii": ["urchade/gliner_multi_pii-v1"],
    "multitask": ["knowledgator/gliner-multitask-v1.0"],
    # High-perf combo (approx confidence descending)
    "best": [
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
# Only set _attn_implementation="eager" to force standard attention;
# prefer not passing "flash": GLiNER auto-detects FlashDeBERTa if installed.
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

# Extended label list (generic NER + frequent PII) for convenience
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
    # Country-specific (optional)
    "CPF",
    "CNPJ",
]


def _normalize_gliner_label(lbl: str) -> Optional[str]:
    lab = (lbl or "").strip()
    if not lab:
        return None
    # Keep uppercase text; do not replace spaces to preserve readability
    return lab.upper()


# ---------------- HF NER ----------------
_HF_NER = None
HF_NER_MODEL_PATH = "Davlan/bert-base-multilingual-cased-ner-hrl"


def get_hf_ner(
    device: Optional[Union[int, str]] = None, half: Optional[bool] = None
):
    """
    Lazy load the HF NER pipeline on best device.

    device: int | str | None (None => auto)
    half: if True, try float16 on GPU
    """
    global _HF_NER
    if device is None:
        device = _DEF_DEVICE
    if half is None:
        half = _HALF_PRECISION

    if _HF_NER is None and hf_pipeline is not None and HFModel is not None and HFTokenizer is not None:
        try:
            model_kwargs: Dict[str, Any] = {}
            if torch is not None and half and device != -1:
                try:
                    model_kwargs["torch_dtype"] = torch.float16
                except Exception:
                    pass
            _log(f"Loading HF NER on device={device} half={half}")
            model = HFModel.from_pretrained(HF_NER_MODEL_PATH, **model_kwargs)
            if torch is not None and half and device != -1:
                try:
                    model = model.half()  # type: ignore
                except Exception:
                    _log("Half precision cast failed, continuing in full precision")
            tokenizer = HFTokenizer.from_pretrained(HF_NER_MODEL_PATH)
            _HF_NER = hf_pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device if isinstance(device, int) else (0 if device != -1 else -1),
            )
        except Exception as e:
            _log(f"Failed HF load on device {device}: {e}. Falling back to CPU.")
            if _DEF_DEBUG:
                traceback.print_exc()
            try:
                model = HFModel.from_pretrained(HF_NER_MODEL_PATH)
                tokenizer = HFTokenizer.from_pretrained(HF_NER_MODEL_PATH)
                _HF_NER = hf_pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=-1,
                )
            except Exception as e2:
                _log(f"HF CPU fallback failed: {e2}")
                _HF_NER = None
    return _HF_NER


def run_hf_ner_chunked(
    text: str,
    max_tokens: int = 384,
    stride: int = 64,
    min_conf: float = 0.55,
    device: Optional[Union[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Windowed NER with HuggingFace pipeline. Uses best device automatically.
    """
    nlp = get_hf_ner(device=device)
    if nlp is None:
        return []

    def _process() -> List[Dict[str, Any]]:
        tok = nlp.tokenizer
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping") or []
        if not offsets:
            return []

        # Build sliding windows on token offsets
        wins: List[Tuple[int, int]] = []
        i = 0
        L = len(offsets)
        while i < L:
            j = min(i + max_tokens, L)
            cs, ce = int(offsets[i][0]), int(offsets[j - 1][1])
            if ce > cs:
                wins.append((cs, ce))
            if j == L:
                break
            i = max(0, j - stride)

        seen = set()
        ents: List[Dict[str, Any]] = []
        for cs, ce in wins:
            chunk = text[cs:ce]
            try:
                out = nlp(chunk)
            except Exception as e:
                _log(f"HF inference error on chunk: {e}")
                out = []
            for ent in out:
                score = float(ent.get("score", 0.0))
                if score < min_conf:
                    continue
                s = cs + int(ent.get("start", 0))
                e = cs + int(ent.get("end", 0))
                lab = str(ent.get("entity_group") or ent.get("entity") or "").upper()
                if not lab or e <= s:
                    continue
                key = (s, e, lab)
                if key in seen:
                    continue
                seen.add(key)
                ents.append({"start": s, "end": e, "entity_group": lab, "score": score})
        ents.sort(key=lambda x: (x["start"], x["end"]))
        return ents

    try:
        return _process()
    except Exception as e:
        # Potential OOM => try to rebuild CPU
        msg = str(e).lower()
        if any(k in msg for k in ["cuda", "oom", "out of memory"]) and device != -1:
            _log("OOM/Device error with HF on GPU -> rebuilding on CPU")
            global _HF_NER
            _HF_NER = None
            nlp2 = get_hf_ner(device=-1, half=False)
            if nlp2 is None:
                return []
            return _process()
        return []


# ---------------- DeepPavlov ----------------
_DP_MODELS = None

# DeepPavlov standard tags (plus extended PII used in merges).
# Ref: http://docs.deeppavlov.ai/en/master/features/models/NER.html#7.-ner-tags-list
DEEPPAVLOV_ENTITY_TAGS = [
    "PERSON",
    "NORP",
    "FACILITY",
    "ORGANIZATION",
    "GPE",
    "LOCATION",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
    # Additional PII we may merge-in from other detectors
    "MAIL",
    "URL",
    "IP",
    "TELEPHONE",
    "USERNAME",
]


def _maybe_move_deeppavlov_model(mdl) -> None:  # pragma: no cover - best effort
    if torch is None:
        return
    try:
        if _GLN_DEVICE in {"cuda", "mps"}:
            if hasattr(mdl, "to"):
                try:
                    mdl.to(_GLN_DEVICE)
                except Exception as e:
                    _log(f"DeepPavlov main .to() failed: {e} -> staying on CPU")
            for attr in dir(mdl):
                comp = getattr(mdl, attr, None)
                if hasattr(comp, "to"):
                    try:
                        comp.to(_GLN_DEVICE)
                    except Exception:
                        continue
    except Exception as e:
        _log(f"DeepPavlov move error: {e}")


def _load_dp_models(config_names: List[str]):
    global _DP_MODELS
    if not _DP_AVAILABLE:
        return []

    if _DP_MODELS is None:
        _DP_MODELS = []

    loaded = {n for n, _ in _DP_MODELS}
    models = list(_DP_MODELS)

    for cname in config_names:
        if cname in loaded:
            continue
        try:
            if not hasattr(dp_configs, "ner"):
                continue
            cfg = getattr(dp_configs.ner, cname)
            # install=True per docs (download + install deps)
            mdl = dp_build_model(cfg, download=True, install=True)
            _maybe_move_deeppavlov_model(mdl)
            models.append((cname, mdl))
        except Exception:
            continue

    _DP_MODELS = models
    return models


def _normalize_dp_label(label: str) -> Optional[str]:
    lab = label
    if not lab:
        return None
    if "-" in lab:
        try:
            _, lab = lab.split("-", 1)
        except ValueError:
            pass
    lab = lab.strip().upper()

    # Normalize synonyms
    if lab in {"PER", "PERSON"}:
        lab = "PERSON"
    if lab in {"ORG", "ORGANIZATION", "ORGANISATION"}:
        lab = "ORGANIZATION"
    if lab in {"LOC", "LOCATION"}:
        lab = "LOCATION"
    if lab in {"GPE"}:
        lab = "GPE"
    if lab in {"FAC", "FACILITY"}:
        lab = "FACILITY"
    if lab in {"WORK_OF_ART", "WORKOFART"}:
        lab = "WORK_OF_ART"
    if lab in {"PCT", "PERCENT"}:
        lab = "PERCENT"
    if lab in {"MONEY"}:
        lab = "MONEY"
    if lab in {"QUANTITY", "QUANT"}:
        lab = "QUANTITY"
    if lab in {"ORD", "ORDINAL"}:
        lab = "ORDINAL"
    if lab in {"CARD", "CARDINAL"}:
        lab = "CARDINAL"
    if lab in {"EVENT"}:
        lab = "EVENT"
    if lab in {"LAW"}:
        lab = "LAW"
    if lab in {"LANG", "LANGUAGE"}:
        lab = "LANGUAGE"
    if lab in {"PRODUCT"}:
        lab = "PRODUCT"
    if lab in {"NORP"}:
        lab = "NORP"

    # PII aliases we may map (for merging)
    if lab in {"EMAIL", "MAIL", "E-MAIL"}:
        lab = "MAIL"
    if lab == "URL":
        lab = "URL"
    if lab in {"IP", "IP_ADDRESS", "IPV4", "IPV6"}:
        lab = "IP"
    if lab in {"PHONE", "TELEPHONE", "PHONE_NUMBER", "MOBILE"}:
        lab = "TELEPHONE"
    if lab in {"USERNAME", "USER", "HANDLE", "ACCOUNT"}:
        lab = "USERNAME"
    if lab in {"DATE"}:
        lab = "DATE"
    if lab in {"TIME"}:
        lab = "TIME"

    return lab if lab in set(DEEPPAVLOV_ENTITY_TAGS) else None


def _decode_bio_to_spans(sentence: str, tokens: list, tags: list, sent_start: int):
    ents: List[Tuple[int, int, str]] = []
    pos = 0

    def align(tok: str, p: int) -> int:
        return sentence.find(tok, p)

    cur_label: Optional[str] = None
    cur_s = cur_e = None

    for tok, tag in zip(tokens, tags):
        label = "O" if not tag or tag == "O" else tag
        if label == "O":
            if cur_label is not None:
                ents.append((cur_s, cur_e, cur_label))  # type: ignore[arg-type]
                cur_label = cur_s = cur_e = None
            i = align(tok, pos)
            if i != -1:
                pos = i + len(tok)
            continue

        bio, typ = label.split("-", 1)
        norm = _normalize_dp_label(typ)
        i = align(tok, pos)
        if i == -1:
            continue
        tok_s, tok_e = i, i + len(tok)
        pos = tok_e

        if norm is None:
            if cur_label is not None:
                ents.append((cur_s, cur_e, cur_label))  # type: ignore[arg-type]
            cur_label = cur_s = cur_e = None
            continue

        if bio == "B" or (cur_label is not None and norm != cur_label):
            if cur_label is not None:
                ents.append((cur_s, cur_e, cur_label))  # type: ignore[arg-type]
            cur_label, cur_s, cur_e = norm, tok_s, tok_e
        else:
            if cur_label is None:
                cur_label, cur_s, cur_e = norm, tok_s, tok_e
            else:
                cur_e = tok_e

    if cur_label is not None:
        ents.append((cur_s, cur_e, cur_label))  # type: ignore[arg-type]

    return [
        {"start": sent_start + s, "end": sent_start + e, "entity_group": lab}
        for s, e, lab in ents
        if e > s
    ]


def _dp_predict(model, sentences: list):
    try:
        out = model(sentences)
    except Exception:
        return []
    if isinstance(out, (list, tuple)) and len(out) == 2:
        toks_batch, tags_batch = out[0], out[1]
        res = []
        for tks, tgs in zip(toks_batch, tags_batch):
            res.append((tks, tgs))
        return res
    return []


def run_deeppavlov_ner_ensemble(
    text: str, dp_config_names: List[str], mode: str = "union", min_votes: int = 1, use_gpu: bool = True
):
    """
    Run multiple DeepPavlov NER configs and vote.
    mode: "union" (>=min_votes) or "consensus" (>=all models)
    """
    if not _DP_AVAILABLE:
        return []

    try:
        if use_gpu and torch is not None and _GLN_DEVICE in {"cuda", "mps"}:
            models = _load_dp_models(dp_config_names)
        else:
            models = _load_dp_models(dp_config_names)
    except Exception as e:
        _log(f"DeepPavlov GPU load failed -> CPU fallback: {e}")
        models = _load_dp_models(dp_config_names)

    sent_spans = split_sentences(text)
    if not models:
        return []

    votes: Dict[Tuple[int, int, str], int] = {}
    for _name, model in models:
        for s, e in sent_spans:
            sent = text[s:e]
            preds = _dp_predict(model, [sent])
            if not preds:
                continue
            toks, tags = preds[0]
            spans = _decode_bio_to_spans(sent, toks, tags, s)
            for ent in spans:
                key = (ent["start"], ent["end"], ent["entity_group"])
                votes[key] = votes.get(key, 0) + 1

    total_models = len(models)
    results = []
    for (s, e, lab), v in votes.items():
        if mode == "consensus":
            if v >= max(min_votes, total_models):
                results.append({"start": s, "end": e, "entity_group": lab, "votes": v})
        else:
            if v >= max(1, min_votes):
                results.append({"start": s, "end": e, "entity_group": lab, "votes": v})
    results.sort(key=lambda x: (x["start"], x["end"]))
    return results


# ---------------- GLiNER ----------------
_GLINER_MODELS = None


def _move_gliner_model_to_device(mdl):  # pragma: no cover - best effort
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
    global _GLINER_MODELS
    if not _GLINER_AVAILABLE:
        return []

    if _GLINER_MODELS is None:
        _GLINER_MODELS = []

    loaded_names = {name for name, _ in _GLINER_MODELS}
    models = list(_GLINER_MODELS)

    for name in model_names:
        if name in loaded_names:
            continue
        kwargs: Dict[str, Any] = {}
        if _GLINER_ATTENTION == "eager":
            kwargs["_attn_implementation"] = "eager"  # force standard attention
        if _GLINER_ONNX:
            kwargs["load_onnx_model"] = True
        try:
            mdl = GLiNER.from_pretrained(name, **kwargs)
        except Exception:
            try:
                mdl = GLiNER.from_pretrained(name)
            except Exception:
                continue

        mdl = _move_gliner_model_to_device(mdl)
        if torch is not None and _GLN_DEVICE == "cuda" and (_HALF_PRECISION or _GLINER_FORCE_HALF):
            try:
                getattr(mdl, "model", mdl).half()
            except Exception:
                pass

        models.append((name, mdl))

    _GLINER_MODELS = models
    return models


def run_gliner(
    text: str,
    model_names: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    threshold: float = 0.35,
    device: Optional[str] = None,  # unused; kept for API compatibility
    preset: Optional[str] = None,
    auto_labels: bool = True,
):
    """
    Run one or more GLiNER models and vote across models.
    """
    if preset is None:
        preset = _GLINER_PRESET
    if model_names is None:
        model_names = _GLINER_PRESETS.get(preset, _GLINER_PRESETS["balanced"])  # type: ignore

    if labels is None and auto_labels:
        labels = _GLINER_PII_LABELS if preset == "pii" else ["Person", "Organization", "Location"]

    if not _GLINER_AVAILABLE:
        return []

    models = _load_gliner_models(model_names)
    if not models:
        return []

    spans = split_sentences(text)
    votes: Dict[Tuple[int, int, str], float] = {}

    for name, mdl in models:
        weight = _GLINER_MODEL_WEIGHTS.get(name, 1.0) if _GLINER_WEIGHTING else 1.0

        for s, e in spans:
            chunk = text[s:e]
            try:
                ents = mdl.predict_entities(chunk, labels, threshold=threshold)
            except Exception as e:
                msg = str(e).lower()
                if any(k in msg for k in ["cuda", "oom", "out of memory"]) and torch is not None and _GLN_DEVICE in {
                    "cuda",
                    "mps",
                }:
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

    results = [{"start": s, "end": e, "entity_group": lab, "votes": v} for (s, e, lab), v in votes.items()]
    results.sort(key=lambda x: (x["start"], x["end"]))
    return results


# ---------------- Merge helper ----------------
def merge_ner_lists(*ner_lists):
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
    dp_configs: Optional[List[str]] = None,
    load_hf: bool = True,
    gliner_labels: Optional[List[str]] = None,
    gliner_threshold: float = 0.35,
) -> None:
    """
    Preload models to reduce cold-start latency.

    gliner_preset: one of _GLINER_PRESETS keys (e.g. "balanced", "pii")
    dp_configs: list of DeepPavlov config names (e.g. ["ner_ontonotes_bert"])
    load_hf: if True, build HF NER pipeline
    gliner_labels: labels to use for a small GLiNER warm-up call
    gliner_threshold: threshold for the warm-up call
    """
    # HF
    if load_hf:
        try:
            get_hf_ner()
        except Exception as e:
            _log(f"Warm-up HF NER failed: {e}")

    # DeepPavlov
    if dp_configs:
        try:
            _load_dp_models(dp_configs)
        except Exception as e:
            _log(f"Warm-up DeepPavlov failed: {e}")

    # GLiNER
    try:
        preset = gliner_preset or _GLINER_PRESET
        model_names = _GLINER_PRESETS.get(preset, _GLINER_PRESETS["balanced"])
        _load_gliner_models(model_names)
        # Prime internal caches with a tiny run
        if gliner_labels is None:
            gliner_labels = ["Person", "Organization", "Location"]
        _ = run_gliner("Warmup text.", model_names=model_names, labels=gliner_labels, threshold=gliner_threshold)
    except Exception as e:
        _log(f"Warm-up GLiNER failed: {e}")


# ---------------- Fast mode (reduced checks) ----------------
# Activate with: export NER_FAST=1
if os.getenv("NER_FAST", "0").lower() in {"1", "true", "yes"}:
    _log("NER_FAST activated (GPU optimizations, fewer checks)")

    # Force best device if available
    if torch is not None and torch.cuda.is_available():
        _FAST_DEVICE = "cuda"
    elif torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _FAST_DEVICE = "mps"
    else:
        _FAST_DEVICE = "cpu"

    # ---- HF (batch) ----
    _HF_NER = None  # override any previous instance

    def get_hf_ner(device: Optional[Union[int, str]] = None, half: Optional[bool] = None):  # type: ignore
        global _HF_NER
        if _HF_NER is not None:
            return _HF_NER
        if hf_pipeline is None or HFModel is None or HFTokenizer is None:
            return None
        device = device or (0 if _FAST_DEVICE != "cpu" else -1)
        half = half if half is not None else (_FAST_DEVICE == "cuda" and _HALF_PRECISION)
        model_kwargs: Dict[str, Any] = {}
        if half and torch is not None and _FAST_DEVICE == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        model = HFModel.from_pretrained(HF_NER_MODEL_PATH, **model_kwargs)
        if half and torch is not None and _FAST_DEVICE == "cuda":
            try:
                model.half()  # type: ignore
            except Exception:
                pass
        tok = HFTokenizer.from_pretrained(HF_NER_MODEL_PATH)
        _HF_NER = hf_pipeline(
            "ner",
            model=model,
            tokenizer=tok,
            aggregation_strategy="simple",
            device=device,
        )
        return _HF_NER

    def run_hf_ner_chunked(  # type: ignore
        text: str, max_tokens: int = 384, stride: int = 64, min_conf: float = 0.55, device: Optional[Union[int, str]] = None
    ) -> List[Dict[str, Any]]:
        nlp = get_hf_ner(device=device)
        if nlp is None:
            return []
        tok = nlp.tokenizer
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping") or []
        if not offsets:
            return []
        # Windows
        wins: List[Tuple[int, int]] = []
        i = 0
        L = len(offsets)
        while i < L:
            j = min(i + max_tokens, L)
            cs, ce = int(offsets[i][0]), int(offsets[j - 1][1])
            if ce > cs:
                wins.append((cs, ce))
            if j == L:
                break
            i = max(0, j - stride)
        # Batch pipeline to reduce Python overhead
        chunks = [text[s:e] for s, e in wins]
        with (torch.inference_mode() if (torch is not None and _FAST_DEVICE != "cpu") else _nullcontext()):
            outs = nlp(chunks)
        ents: List[Dict[str, Any]] = []
        seen = set()
        for (cs, ce), out in zip(wins, outs):
            for ent in out:
                score = float(ent.get("score", 0.0))
                if score < min_conf:
                    continue
                s = cs + int(ent.get("start", 0))
                e = cs + int(ent.get("end", 0))
                if e <= s:
                    continue
                lab = str(ent.get("entity_group") or ent.get("entity") or "").upper()
                if not lab:
                    continue
                key = (s, e, lab)
                if key in seen:
                    continue
                seen.add(key)
                ents.append({"start": s, "end": e, "entity_group": lab, "score": score})
        ents.sort(key=lambda x: (x["start"], x["end"]))
        return ents

    # ---- DeepPavlov (simple loop) ----
    def _load_dp_models(config_names: List[str]):  # type: ignore
        global _DP_MODELS
        if not _DP_AVAILABLE:
            return []
        if _DP_MODELS is None:
            _DP_MODELS = []
        loaded = {n for n, _ in _DP_MODELS}
        for cname in config_names:
            if cname in loaded:
                continue
            if not hasattr(dp_configs, "ner"):
                continue
            cfg = getattr(dp_configs.ner, cname)
            mdl = dp_build_model(cfg, download=True, install=True)
            if torch is not None and _FAST_DEVICE in {"cuda", "mps"}:
                try:
                    for attr in [mdl] + [getattr(mdl, a) for a in dir(mdl) if not a.startswith("_")]:
                        if hasattr(attr, "to"):
                            try:
                                attr.to(_FAST_DEVICE)
                            except Exception:
                                pass
                except Exception:
                    pass
            _DP_MODELS.append((cname, mdl))
        return _DP_MODELS

    def run_deeppavlov_ner_ensemble(  # type: ignore
        text: str, dp_config_names: List[str], mode: str = "union", min_votes: int = 1, use_gpu: bool = True
    ):
        if not _DP_AVAILABLE:
            return []
        models = _load_dp_models(dp_config_names)
        if not models:
            return []
        sent_spans = split_sentences(text)
        votes: Dict[Tuple[int, int, str], int] = {}
        for _name, model in models:
            for s, e in sent_spans:
                toks_tags = _dp_predict(model, [text[s:e]])
                if not toks_tags:
                    continue
                toks, tags = toks_tags[0]
                for ent in _decode_bio_to_spans(text[s:e], toks, tags, s):
                    key = (ent["start"], ent["end"], ent["entity_group"])
                    votes[key] = votes.get(key, 0) + 1
        total = len(models)
        out = []
        for (s, e, lab), v in votes.items():
            if mode == "consensus":
                if v >= max(min_votes, total):
                    out.append({"start": s, "end": e, "entity_group": lab, "votes": v})
            else:
                if v >= max(1, min_votes):
                    out.append({"start": s, "end": e, "entity_group": lab, "votes": v})
        out.sort(key=lambda x: (x["start"], x["end"]))
        return out

    # ---- GLiNER simplified ----
    def _load_gliner_models(model_names: List[str]):  # type: ignore
        global _GLINER_MODELS
        if not _GLINER_AVAILABLE:
            return []
        if _GLINER_MODELS is None:
            _GLINER_MODELS = []
        loaded = {n for n, _ in _GLINER_MODELS}
        for name in model_names:
            if name in loaded:
                continue
            kwargs: Dict[str, Any] = {}
            # In fast mode we respect the same env switches
            if _GLINER_ATTENTION == "eager":
                kwargs["_attn_implementation"] = "eager"
            if _GLINER_ONNX:
                kwargs["load_onnx_model"] = True
            try:
                mdl = GLiNER.from_pretrained(name, **kwargs)
            except Exception:
                try:
                    mdl = GLiNER.from_pretrained(name)
                except Exception:
                    continue
            if torch is not None and _FAST_DEVICE in {"cuda", "mps"}:
                target = getattr(mdl, "model", mdl)
                if hasattr(target, "to"):
                    try:
                        target.to(_FAST_DEVICE)
                    except Exception:
                        pass
                if _FAST_DEVICE == "cuda" and (_HALF_PRECISION or _GLINER_FORCE_HALF):
                    try:
                        target.half()
                    except Exception:
                        pass
            _GLINER_MODELS.append((name, mdl))
        return _GLINER_MODELS

    def run_gliner(  # type: ignore
        text: str,
        model_names=None,
        labels=None,
        threshold: float = 0.35,
        device: Optional[str] = None,
        preset: Optional[str] = None,
        auto_labels: bool = True,
    ):
        if preset is None:
            preset = _GLINER_PRESET
        if model_names is None:
            model_names = _GLINER_PRESETS.get(preset, _GLINER_PRESETS["balanced"])  # type: ignore
        if labels is None and auto_labels:
            labels = _GLINER_PII_LABELS if preset == "pii" else ["Person", "Organization", "Location"]
        if not _GLINER_AVAILABLE:
            return []
        models = _load_gliner_models(model_names)
        if not models:
            return []
        spans = split_sentences(text)
        votes: Dict[Tuple[int, int, str], float] = {}
        for model_name, mdl in models:
            weight = _GLINER_MODEL_WEIGHTS.get(model_name, 1.0) if _GLINER_WEIGHTING else 1.0
            for s, e in spans:
                chunk = text[s:e]
                try:
                    ents = mdl.predict_entities(chunk, labels, threshold=threshold) or []
                except Exception:
                    ents = []
                for ent in ents:
                    start, end = ent.get("start"), ent.get("end")
                    if (start is None or end is None) and ent.get("text"):
                        idx = chunk.find(ent["text"])
                        if idx != -1:
                            start, end = idx, idx + len(ent["text"])
                    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                        continue
                    g = _normalize_gliner_label(str(ent.get("label") or ent.get("type")))
                    if not g:
                        continue
                    gs, ge = s + start, s + end
                    votes[(gs, ge, g)] = votes.get((gs, ge, g), 0.0) + weight
        res = [{"start": s, "end": e, "entity_group": lab, "votes": v} for (s, e, lab), v in votes.items()]
        res.sort(key=lambda x: (x["start"], x["end"]))
        return res


__all__ = [
    "run_deeppavlov_ner_ensemble",
    "run_gliner",
    "run_hf_ner_chunked",
    "merge_ner_lists",
    "warm_up_models",
    "GLINER_ALL_LABELS",
    "DEEPPAVLOV_ENTITY_TAGS",
]