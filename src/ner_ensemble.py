"""NER ensemble utilities extracted from orchestrator.

Provides optional DeepPavlov, GLiNER and HF (transformers) pipelines plus
helpers for sentence splitting and entity list merging.

All dependencies are optional; if a library is missing the corresponding
function will return an empty list instead of failing.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re

# ---- Optional imports (guarded) ----
try:  # pragma: no cover
    from deeppavlov import configs as dp_configs  # type: ignore
    from deeppavlov import build_model as dp_build_model  # type: ignore
    _DP_AVAILABLE = True
except Exception:  # pragma: no cover
    _DP_AVAILABLE = False  # type: ignore
    dp_configs = None      # type: ignore
    dp_build_model = None  # type: ignore

try:  # pragma: no cover
    from gliner import GLiNER  # type: ignore
    _GLINER_AVAILABLE = True
except Exception:  # pragma: no cover
    _GLINER_AVAILABLE = False  # type: ignore
    GLiNER = None             # type: ignore

try:  # pragma: no cover
    from transformers import (  # type: ignore
        pipeline as hf_pipeline,
        AutoTokenizer as HFTokenizer,
        AutoModelForTokenClassification as HFModel,
    )
except Exception:  # pragma: no cover
    hf_pipeline = HFTokenizer = HFModel = None  # type: ignore

# ---- Sentence splitting helpers ----
_ABBR = {
    # FR
    "m.", "mme.", "mlle.", "dr.", "pr.", "art.", "n°", "p. ex.", "cf.", "etc.",
    # EN
    "mr.", "mrs.", "ms.", "dr.", "prof.", "art.", "no.", "nos.", "vol.", "inc.",
    "jr.", "sr.", "co.", "vs.", "st.", "e.g.", "i.e.", "cf.", "u.s.", "u.k.",
    # mois abrégés
    "jan.", "feb.", "mar.", "apr.", "aug.", "sept.", "oct.", "nov.", "dec.",
    "janv.", "févr.", "avr.", "juil.", "sept.", "oct.", "nov.", "déc."
}

def split_sentences(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch == "\n":  # double newline => hard break
            j = i
            while j < n and text[j] == "\n":
                j += 1
            if j - i >= 2:
                if start < i:
                    spans.append((start, i))
                start = j
                i = j
                continue
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
            if m >= n or (m < n and (text[m].isupper() or text[m] in "\"'([“”‘’)") ):
                spans.append((start, m))
                start = m
                i = m
                continue
        i += 1
    if start < n:
        spans.append((start, n))
    return [(s, e) for s, e in spans if e - s > 0]

# ---- HF NER ----
_HF_NER = None
HF_NER_MODEL_PATH = "Davlan/bert-base-multilingual-cased-ner-hrl"

def get_hf_ner():  # lazy load
    global _HF_NER
    if _HF_NER is None and hf_pipeline is not None and HFModel is not None and HFTokenizer is not None:
        try:  # pragma: no cover
            _HF_NER = hf_pipeline(
                "ner",
                model=HFModel.from_pretrained(HF_NER_MODEL_PATH),
                tokenizer=HFTokenizer.from_pretrained(HF_NER_MODEL_PATH),
                aggregation_strategy="simple",
            )
        except Exception:
            _HF_NER = None
    return _HF_NER

def run_hf_ner_chunked(text: str, max_tokens: int = 384, stride: int = 64, min_conf: float = 0.55) -> List[Dict[str, Any]]:
    nlp = get_hf_ner()
    if nlp is None:
        return []
    tok = nlp.tokenizer
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.get("offset_mapping") or []
    if not offsets:
        return []
    wins: List[Tuple[int, int]] = []
    i = 0
    while i < len(offsets):
        j = min(i + max_tokens, len(offsets))
        cs, ce = int(offsets[i][0]), int(offsets[j - 1][1])
        if ce > cs:
            wins.append((cs, ce))
        if j == len(offsets):
            break
        i = max(0, j - stride)
    seen = set()
    ents: List[Dict[str, Any]] = []
    for cs, ce in wins:
        chunk = text[cs:ce]
        try:
            out = nlp(chunk)
        except Exception:
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

# ---- DeepPavlov ----
_DP_MODELS = None

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
            mdl = dp_build_model(cfg, download=True)
            models.append((cname, mdl))
        except Exception:
            continue
    _DP_MODELS = models
    return models

def _normalize_dp_label(label: str):
    lab = label
    if "-" in lab:
        _, lab = lab.split("-", 1)
    lab = lab.upper()
    if lab in {"PER", "PERSON"}: return "PER"
    if lab in {"ORG", "ORGANIZATION"}: return "ORG"
    if lab in {"GPE", "LOC", "LOCATION", "FAC"}: return "LOC"
    if lab in {"EMAIL", "MAIL", "E-MAIL"}: return "MAIL"
    if lab in {"URL"}: return "URL"
    if lab in {"IP", "IP_ADDRESS", "IPV4", "IPV6"}: return "IP"
    if lab in {"PHONE", "TELEPHONE", "PHONE_NUMBER"}: return "TELEPHONE"
    if lab in {"USERNAME", "USER", "HANDLE", "ACCOUNT"}: return "USERNAME"
    if lab in {"DATE", "TIME"}: return "DATE"
    return None

def _decode_bio_to_spans(sentence: str, tokens: list, tags: list, sent_start: int):
    ents = []
    pos = 0
    def align(tok: str, pos: int):
        return sentence.find(tok, pos)
    cur_label = cur_s = cur_e = None
    for tok, tag in zip(tokens, tags):
        label = "O" if not tag or tag == "O" else tag
        if label == "O":
            if cur_label is not None:
                ents.append((cur_s, cur_e, cur_label))
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
                ents.append((cur_s, cur_e, cur_label))
            cur_label = cur_s = cur_e = None
            continue
        if bio == "B" or (cur_label is not None and norm != cur_label):
            if cur_label is not None:
                ents.append((cur_s, cur_e, cur_label))
            cur_label, cur_s, cur_e = norm, tok_s, tok_e
        else:
            if cur_label is None:
                cur_label, cur_s, cur_e = norm, tok_s, tok_e
            else:
                cur_e = tok_e
    if cur_label is not None:
        ents.append((cur_s, cur_e, cur_label))
    return [{"start": sent_start + s, "end": sent_start + e, "entity_group": lab} for s, e, lab in ents if e > s]

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

def run_deeppavlov_ner_ensemble(text: str, dp_config_names: List[str], mode: str = "union", min_votes: int = 1):
    if not _DP_AVAILABLE:
        return []
    sent_spans = split_sentences(text)
    models = _load_dp_models(dp_config_names)
    if not models:
        return []
    votes: Dict[Tuple[int,int,str], int] = {}
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

# ---- GLiNER ----
_GLINER_MODELS = None

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
        try:
            mdl = GLiNER.from_pretrained(name)
            models.append((name, mdl))
        except Exception:
            continue
    _GLINER_MODELS = models
    return models

def _normalize_gliner_label(lbl: str):
    l = (lbl or "").strip().lower()
    if l in {"per","person","individual","name"}: return "PER"
    if l in {"org","organization","organisation","company","agency","institution"}: return "ORG"
    if l in {"loc","location","place","city","country","address","gpe","facility"}: return "LOC"
    if l in {"email","email address","mail","e-mail"}: return "MAIL"
    if l in {"phone","phone number","telephone","mobile"}: return "TELEPHONE"
    if l in {"ip","ip address","ipv4","ipv6"}: return "IP"
    if l in {"url","link","uri","website"}: return "URL"
    if l in {"username","user","handle","account"}: return "USERNAME"
    if l in {"date","time"}: return "DATE"
    return None

GLINER_ALL_LABELS = [
    "Person","Organization","Location",
    "Email address","Phone number","IP address","URL","Username",
    "Account number","Credit card number","IBAN","Tax ID","National ID",
    "Passport number","Driver license number","Vehicle plate","Case number",
    "Medical record number","SSN","Address","City","Country",
    "Project","Team","Service","API","App","Host","Ticket",
    "Date","Time",
]

def run_gliner(text: str, model_names=None, labels=None, threshold: float = 0.35):
    if model_names is None:
        model_names = ["urchade/gliner_large-v2.1", "urchade/gliner_multi-v2.1"]
    if labels is None:
        labels = ["Person", "Organization", "Location"]
    if not _GLINER_AVAILABLE:
        return []
    models = _load_gliner_models(model_names)
    if not models:
        return []
    spans = split_sentences(text)
    votes: Dict[Tuple[int,int,str], int] = {}
    for _, mdl in models:
        for s, e in spans:
            chunk = text[s:e]
            try:
                ents = mdl.predict_entities(chunk, labels, threshold=threshold)
            except Exception:
                ents = []
            for ent in ents or []:
                start = ent.get("start")
                end = ent.get("end")
                label = ent.get("label") or ent.get("type")
                if (start is None or end is None) and ent.get("text"):
                    idx = chunk.find(ent["text"])
                    if idx != -1:
                        start = idx; end = idx + len(ent["text"])
                if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                    continue
                g_label = _normalize_gliner_label(str(label))
                if not g_label:
                    continue
                gs, ge = s + int(start), s + int(end)
                key = (gs, ge, g_label)
                votes[key] = votes.get(key, 0) + 1
    results = [{"start": s, "end": e, "entity_group": lab, "votes": v} for (s,e,lab), v in votes.items()]
    results.sort(key=lambda x: (x["start"], x["end"]))
    return results

# ---- Merge helper ----

def merge_ner_lists(*ner_lists):
    seen = set(); out = []
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
            out.append({"start": s, "end": e, "entity_group": lab, **({k:v for k,v in ent.items() if k not in {"start","end","entity_group"}})})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out

__all__ = [
    "run_deeppavlov_ner_ensemble",
    "run_gliner",
    "run_hf_ner_chunked",
    "merge_ner_lists",
    "GLINER_ALL_LABELS",
]
