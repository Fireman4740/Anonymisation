"""High-level orchestration for multi-level anonymization with optional LLM layer.

Fix: utilisation d'import relatifs pour fonctionner lors de l'import `from src.orchestrator import anonymize_text`.
"""
from typing import List, Tuple, Dict, Any, Optional
import re

try:
    from .policy import preset, AnonymizationPolicy  # type: ignore
    from .llm_reasoner_openrouter import LLMReasoner, SeedSpan, DetectionPlan  # type: ignore
    from .openrouter_client import OpenRouterClient  # type: ignore
    from .utils_pseudo import PseudoMapper  # type: ignore
    from .text_sanitizer import regexes_based_replacements  # type: ignore
    from .ner_ensemble import (
        run_deeppavlov_ner_ensemble,
        run_gliner,
        run_hf_ner_chunked,
        merge_ner_lists,
        GLINER_ALL_LABELS,
    )  # type: ignore
except Exception:  # fallback exécution directe
    from policy import preset, AnonymizationPolicy  # type: ignore
    from llm_reasoner_openrouter import LLMReasoner, SeedSpan, DetectionPlan  # type: ignore
    from openrouter_client import OpenRouterClient  # type: ignore
    from utils_pseudo import PseudoMapper  # type: ignore
    from text_sanitizer import regexes_based_replacements  # type: ignore
    from ner_ensemble import (
        run_deeppavlov_ner_ensemble,
        run_gliner,
        run_hf_ner_chunked,
        merge_ner_lists,
        GLINER_ALL_LABELS,
    )  # type: ignore


"""Ajout NER ensemble extrait vers ner_ensemble et amélioration dates FR/EN."""

def placeholder_from_policy(policy: AnonymizationPolicy, etype: str, surface: str, mapper: PseudoMapper) -> str:
    etype_up = etype.upper()
    if policy.placeholder_style == "typed":
        return mapper.placeholder(etype_up, surface)
    if policy.placeholder_style == "generic":
        return f"[{etype_up}]"
    return "[REDACTED]"


def build_seeds(value: str, regex_hits: List[Tuple[int, int, str]], ner_entities: List[dict]) -> List[SeedSpan]:
    seeds: List[SeedSpan] = []
    for s, e, tag in regex_hits:
        seeds.append(SeedSpan(type=tag.strip("<>").upper(), start=s, end=e, surface=value[s:e]))
    for ent in ner_entities:
        try:
            seeds.append(
                SeedSpan(
                    type=str(ent.get("entity_group", "MISC")).upper(),
                    start=int(ent["start"]),
                    end=int(ent["end"]),
                    surface=value[int(ent["start"]): int(ent["end"])],
                )
            )
        except Exception:
            continue
    return seeds


def merge_non_overlapping(
    base: List[Tuple[int, int, str, Dict[str, Any]]],
    extra: List[Tuple[int, int, str, Dict[str, Any]]],
) -> List[Tuple[int, int, str, Dict[str, Any]]]:
    """Fusion priorité: llm-entity > ner > regex > llm-generalization, puis span le plus court."""
    def priority(meta: Dict[str, Any]) -> int:
        src = (meta or {}).get("source", "")
        if src == "llm-entity":
            return 3
        if src == "ner":
            return 2
        if src == "regex":
            return 1
        return 0  # généralisation ou autre

    all_items = list(base) + list(extra)
    all_items.sort(key=lambda x: (x[0], x[1]))
    result: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for s, e, rep, meta in all_items:
        keep = True
        for i, (rs, re, rrep, rmeta) in enumerate(result):
            if not (e <= rs or s >= re):  # chevauchement
                p_new, p_old = priority(meta), priority(rmeta)
                len_new, len_old = (e - s), (re - rs)
                if (p_new > p_old) or (p_new == p_old and len_new < len_old):
                    result[i] = (s, e, rep, meta)
                keep = False
                break
        if keep:
            result.append((s, e, rep, meta))
    result.sort(key=lambda x: x[0])
    return result

def _find_near(text: str, surface: str, hint: int, window: int = 64) -> int:
    a = max(0, hint - window)
    b = min(len(text), hint + window)
    k = text.find(surface, a, b)
    return k if k != -1 else text.find(surface)

def _trim_to_alnum(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    return start, end


def generalize_dates(text: str, policy: AnonymizationPolicy) -> Tuple[str, List[Dict[str, Any]]]:
    """Généralise les dates (EN + FR) conformément à la granularité policy (month/quarter/year)."""
    changes: List[Dict[str, Any]] = []
    if policy.date_granularity in {"none"}:
        return text, changes

    MONTHS = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
        "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,"sept":9,"oct":10,"nov":11,"dec":12,
        "janvier":1,"février":2,"fevrier":2,"mars":3,"avril":4,"mai":5,"juin":6,"juillet":7,"août":8,"aout":8,"septembre":9,"octobre":10,"novembre":11,"décembre":12,"decembre":12,
        "janv":1,"févr":2,"fevr":2,"avr":4,"juil":7,"sept":9,"oct":10,"nov":11,"déc":12,"dec":12,
    }

    def month_to_num(name: str) -> int:
        return MONTHS.get(name.strip(". ").lower(), 0)

    def make_rep(y: str, mo: str) -> str:
        if policy.date_granularity in {"week", "month"}:
            return f"[DATE_{y}-{mo}]"
        if policy.date_granularity == "quarter":
            q = (int(mo) - 1) // 3 + 1
            return f"[DATE_{y}-Q{q}]"
        if policy.date_granularity == "year":
            return f"[DATE_{y}]"
        return "[DATE]"

    patterns = [
        re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
        re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.IGNORECASE),
        re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})\b", re.IGNORECASE),
        re.compile(r"\b(\d{1,2}|1er)\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|janv\.?,?|févr\.?,?|fevr\.?,?|avr\.?,?|juil\.?,?|sept\.?,?|oct\.?,?|nov\.?,?|déc\.?,?|dec\.?)\s+(\d{4})\b", re.IGNORECASE),
        re.compile(r"\b(\d{1,2})[\/.\-](\d{1,2})[\/.\-](\d{2,4})\b"),
    ]

    out = text
    delta = 0
    matches = []
    for pat in patterns:
        for m in pat.finditer(text):
            matches.append((pat, m.start(), m.end(), m))
    matches.sort(key=lambda x: x[1])

    for pat, s0, e0, m in matches:
        s = s0 + delta; e = e0 + delta
        g = m.groups(); y = None; mo = None
        if len(g) == 3 and re.match(r"^\d{4}$", g[0] or ""):
            y, mo = g[0], g[1]
        elif len(g) == 3 and g[1] and g[1][0].isalpha() and (g[0].isdigit() or g[0].lower() == "1er"):
            y = g[2]; mo = f"{month_to_num(g[1]):02d}"
        elif len(g) == 3 and g[0] and g[0][0].isalpha() and (g[1].isdigit() or g[1].lower() == "1er"):
            y = g[2]; mo = f"{month_to_num(g[0]):02d}"
        elif len(g) == 3 and g[0].isdigit() and g[1].isdigit():
            d1, m2, y3 = int(g[0]), int(g[1]), g[2]
            if len(y3) == 2:
                y3 = f"20{y3}" if int(y3) <= 29 else f"19{y3}"
            y, mo = y3, f"{m2:02d}"
        if not (y and mo and mo.isdigit() and 1 <= int(mo) <= 12):
            continue
        rep = make_rep(y, mo)
        out = out[:s] + rep + out[e:]
        diff = len(rep) - (e - s)
        delta += diff
        changes.append({
            "etype": "DATE",
            "start": s, "end": e,
            "surface": m.group(0),
            "replacement": rep,
            "policy": policy.date_granularity,
            "source": "policy-generalization",
        })
    return out, changes


def generalize_org_placeholders(text: str, policy: AnonymizationPolicy) -> Tuple[str, List[Dict[str, Any]]]:
    changes: List[Dict[str, Any]] = []
    if policy.org_policy not in {"generalize", "redact"}:
        return text, changes

    def repl(_m: re.Match) -> str:
        return "[ORG]" if policy.org_policy == "generalize" else "[REDACTED]"

    pattern = re.compile(r"\[ORG_[A-Z0-9]+\]")
    out = text
    for m in list(pattern.finditer(text)):
        s, e = m.start(), m.end()
        new = repl(m)
        if new != m.group(0):
            out = out[:s] + new + out[e:]
            changes.append(
                {
                    "etype": "ORG",
                    "start": s,
                    "end": e,
                    "surface": m.group(0),
                    "replacement": new,
                    "policy": policy.org_policy,
                    "source": "hardening",
                }
            )
    return out, changes


def escalate_policy_inline(policy: AnonymizationPolicy) -> None:
    order_date = ["none", "week", "month", "quarter", "year", "redact"]
    try:
        i = order_date.index(policy.date_granularity)
        if i + 1 < len(order_date):
            policy.date_granularity = order_date[i + 1]  # type: ignore
    except Exception:
        policy.date_granularity = "month"  # type: ignore

    order_ip = ["exact", "public_private", "cidr24", "redact"]
    try:
        i = order_ip.index(policy.ip_policy)
        if i + 1 < len(order_ip):
            policy.ip_policy = order_ip[i + 1]  # type: ignore
    except Exception:
        policy.ip_policy = "cidr24"  # type: ignore

    order_org = ["replace", "categorize", "generalize", "redact"]
    try:
        i = order_org.index(policy.org_policy)
        if i + 1 < len(order_org):
            policy.org_policy = order_org[i + 1]  # type: ignore
    except Exception:
        policy.org_policy = "generalize"  # type: ignore

    policy.paraphrase_intensity = min(3, int(policy.paraphrase_intensity or 1) + 1)


def anonymize_text(
    value: str,
    scope_id: str,
    secret_salt: str,
    level: str = "L2",
    openrouter_models: Optional[Dict[str, str]] = None,
    ner_results: Optional[List[dict]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy = preset(level)
    # Overrides optionnels (ex: {"llm_paraphrase": False})
    if overrides:
        for k, v in overrides.items():
            if hasattr(policy, k):  # sécurité minimale
                try:
                    setattr(policy, k, v)
                except Exception:
                    pass
    mapper = PseudoMapper(secret=secret_salt, scope_id=scope_id)

    # Step 1: regex
    regex_hits = regexes_based_replacements(value)
    skip_tags = set()
    if overrides and isinstance(overrides.get("skip_regex_tags"), (list, tuple, set)):
        skip_tags = {str(t).upper() for t in overrides["skip_regex_tags"]}
    if skip_tags:
        regex_hits = [(s, e, tag) for (s, e, tag) in regex_hits if str(tag).strip("<>").upper() not in skip_tags]

    base_reps: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for s, e, tag in regex_hits:
        surface = value[s:e]
        etype = tag.strip("<>").upper()
        ph = placeholder_from_policy(policy, etype, surface, mapper)
        base_reps.append((s, e, ph, {"etype": etype, "surface": surface, "source": "regex"}))

    # Step 1bis: NER — fusion externe (ner_results) + interne (DP + GLiNER + HF)
    use_dp = True
    dp_cfgs = ["ner_conll2003_bert", "ner_ontonotes_bert", "ner_ontonotes_bert_mult"]
    use_gliner = True
    gl_models = ["urchade/gliner_large-v2.1", "urchade/gliner_multi-v2.1"]
    gl_labels = GLINER_ALL_LABELS
    gl_thresh = 0.35
    ner_mode = "union"
    ner_min_votes = 1
    disable_internal = False

    if overrides:
        use_dp = bool(overrides.get("ner_use_deeppavlov", use_dp))
        dp_cfgs = list(overrides.get("ner_dp_configs", dp_cfgs))
        use_gliner = bool(overrides.get("ner_use_gliner", use_gliner))
        gl_models = list(overrides.get("gliner_models", gl_models))
        gl_labels = list(overrides.get("gliner_labels", gl_labels))
        gl_thresh = float(overrides.get("gliner_threshold", gl_thresh))
        ner_mode = str(overrides.get("ner_mode", ner_mode)).lower()
        ner_min_votes = int(overrides.get("ner_min_votes", ner_min_votes))
        disable_internal = bool(overrides.get("disable_internal_ner", disable_internal))

    local_ner = ner_results or []
    if not disable_internal:
        dp_ents = run_deeppavlov_ner_ensemble(value, dp_cfgs, mode=ner_mode, min_votes=ner_min_votes) if use_dp else []
        gl_ents = run_gliner(value, model_names=gl_models, labels=gl_labels, threshold=gl_thresh) if use_gliner else []
        hf_ents = run_hf_ner_chunked(value) if (not dp_ents and not gl_ents) else []
        local_ner = merge_ner_lists(local_ner, dp_ents, gl_ents, hf_ents)

    def _map_label(lab: str) -> str:
        L = lab.upper()
        if L in {"PER","PERSON"}: return "PER"
        if L in {"ORG","ORGANIZATION"}: return "ORG"
        if L in {"LOC","LOCATION","GPE","FACILITY","FAC"}: return "LOC"
        if L in {"MAIL","EMAIL","EMAIL ADDRESS","E-MAIL"}: return "MAIL"
        if L in {"TELEPHONE","PHONE","PHONE NUMBER"}: return "TELEPHONE"
        if L in {"IP","IP ADDRESS","IPV4","IPV6"}: return "IP"
        if L in {"URL","URI","LINK"}: return "URL"
        if L in {"USERNAME","USER","HANDLE","ACCOUNT"}: return "USERNAME"
        return ""

    for ent in (local_ner or []):
        etype = _map_label(str(ent.get("entity_group", "")))
        if not etype or etype.startswith("DATE"):
            continue
        s = ent.get("start"); e = ent.get("end")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(value):
            surface = value[s:e]
            ph = placeholder_from_policy(policy, etype, surface, mapper)
            base_reps.append((s, e, ph, {"etype": etype, "surface": surface, "source": "ner"}))

    base_reps = merge_non_overlapping([], base_reps)

    # Step 2: LLM Reasoner (si activé et clé dispo)
    reasoner = None
    reps_all: List[Tuple[int, int, str, Dict[str, Any]]] = []
    llm_error: Optional[str] = None
    llm_used: bool = False
    models_used: Optional[Dict[str, str]] = None
    if policy.llm_detection:
        try:
            # Modèles par défaut sélectionnés pour compatibilité JSON
            models = openrouter_models or {
                "detect": "openai/gpt-4.1-mini",
                "paraphrase": "openai/gpt-4.1-mini",
                "audit": "openai/gpt-4.1-mini",
            }
            models_used = models.copy()
            client = OpenRouterClient()
            reasoner = LLMReasoner(client, model_detect=models["detect"], model_paraphrase=models.get("paraphrase"), model_audit=models.get("audit"))
            seeds = build_seeds(value, regex_hits, local_ner or [])
            plan = reasoner.detect_and_plan(value, seeds, policy.to_dict())
            llm_used = True

            cluster_to_placeholder: Dict[str, str] = {}
            for ent in plan.entities:
                c_id = ent.get("cluster_id") or ent.get("canonical") or ent.get("id")
                et = ent.get("placeholder_type") or ent.get("type") or "GEN"
                surf = str(ent.get("canonical") or ent.get("surface") or "")
                if c_id and et and surf and c_id not in cluster_to_placeholder:
                    cluster_to_placeholder[c_id] = placeholder_from_policy(policy, str(et), surf, mapper)

            llm_reps: List[Tuple[int, int, str, Dict[str, Any]]] = []
            skip_dates = str(policy.date_granularity or "none") == "none"
            for ent in plan.entities:
                if ent.get("action") not in {"REPLACE", "REDACT", "GENERALIZE"}:
                    continue
                etype_norm = str(ent.get("placeholder_type") or ent.get("type") or "GEN").upper()
                if skip_dates and etype_norm.startswith("DATE"):
                    # Respecte la policy: ne pas toucher aux dates si granularité none
                    continue
                s, e = ent.get("start"), ent.get("end")
                surf = ent.get("surface", "")
                if not isinstance(s, int) or not isinstance(e, int) or not surf:
                    continue
                if not (0 <= s < e <= len(value)) or value[s:e] != surf:
                    idx = _find_near(value, surf, s if isinstance(s, int) else 0)
                    if idx == -1:
                        continue
                    s, e = idx, idx + len(surf)
                # Trim pour éviter ponctuation/espaces adjacents
                ts, te = _trim_to_alnum(value, s, e)
                if te - ts <= 0:
                    continue
                if (ts, te) != (s, e):
                    surf = value[ts:te]
                    s, e = ts, te
                if ent.get("action") == "REDACT":
                    rep = "[REDACTED]"
                else:
                    c_id = ent.get("cluster_id") or ent.get("canonical") or ent.get("id")
                    rep = cluster_to_placeholder.get(c_id) or placeholder_from_policy(
                        policy, ent.get("placeholder_type", ent.get("type", "GEN")), surf, mapper
                    )
                llm_reps.append((s, e, rep, {"etype": etype_norm, "surface": surf, "source": "llm-entity"}))

            for gen in plan.generalizations:
                gtype_norm = str(gen.get("type", "")).upper()
                if skip_dates and gtype_norm.startswith("DATE"):
                    # Ignore la généralisation DATE si policy none
                    continue
                s, e = gen.get("start"), gen.get("end")
                surf = gen.get("surface")
                rep = gen.get("replacement")
                if isinstance(s, int) and isinstance(e, int) and rep and surf and 0 <= s < e <= len(value):
                    if value[s:e] != surf:
                        idx = _find_near(value, surf, s)
                        if idx != -1:
                            s, e = idx, idx + len(surf)
                    if 0 <= s < e <= len(value):
                        llm_reps.append((s, e, rep, {"etype": gtype_norm or gen.get("type", "GENERALIZATION"), "surface": surf, "source": "llm-generalization"}))

            # NOTE: on ignore désormais plan.edits (free-form) pour éviter des corruptions partielles de mots.

            reps_all = merge_non_overlapping(base_reps, llm_reps)
        except Exception as e:
            # On tombe en mode regex/NER uniquement mais on expose l'erreur pour diagnostic
            llm_error = f"detect_failed: {type(e).__name__}: {e}"
            reps_all = base_reps
    else:
        reps_all = base_reps

    # Step 3: appliquer
    text = value
    applied_meta: List[Dict[str, Any]] = []
    for s, e, rep, meta in reversed(reps_all):
        text = text[:s] + rep + text[e:]
        applied_meta.append({"start": s, "end": e, "replacement": rep, **meta})

    # Step 4: généralisation dates
    text, gen_changes = generalize_dates(text, policy)
    applied_meta.extend(gen_changes)

    # Step 5: paraphrase
    if reasoner and policy.llm_paraphrase and policy.paraphrase_intensity > 0:
        try:
            temp = 0.2 + 0.1 * policy.paraphrase_intensity
            text = reasoner.paraphrase(text, temperature=temp, ensure_placeholders_preserved=True)
        except Exception as e:
            llm_error = (llm_error or "") + f"; paraphrase_failed: {type(e).__name__}: {e}" if llm_error else f"paraphrase_failed: {type(e).__name__}: {e}"

    # Step 6: audit + durcissement
    risk_report = {"risk_score": 0, "findings": [], "recommendations": []}
    if reasoner and policy.llm_audit:
        try:
            risk_report = reasoner.audit(text)
        except Exception as e:
            llm_error = (llm_error or "") + f"; audit_failed: {type(e).__name__}: {e}" if llm_error else f"audit_failed: {type(e).__name__}: {e}"

    rounds = 0
    while isinstance(risk_report.get("risk_score"), int) and risk_report["risk_score"] > policy.risk_threshold and rounds < int(policy.max_hardening_rounds or 0):
        rounds += 1
        escalate_policy_inline(policy)
        text, org_changes = generalize_org_placeholders(text, policy)
        applied_meta.extend(org_changes)
        try:
            temp = 0.2 + 0.1 * policy.paraphrase_intensity
            text = reasoner.paraphrase(text, temperature=temp, ensure_placeholders_preserved=True)
        except Exception:
            pass
        try:
            risk_report = reasoner.audit(text)
        except Exception as e:
            llm_error = (llm_error or "") + f"; hardening_audit_failed: {type(e).__name__}: {e}" if llm_error else f"hardening_audit_failed: {type(e).__name__}: {e}"
            break

    if policy.mapping_retention == "discard":
        mapper.cache.clear()

    return {
        "text": text,
        "audit": {
            "entities": applied_meta,
            "risk": risk_report,
            "llm_error": llm_error,
            "llm_used": llm_used,
            "models": models_used,
        },
        "policy": policy.to_dict(),
    }