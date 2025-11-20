"""High-level orchestration for multi-level anonymization with optional LLM layer.

Fix: utilisation d'import relatifs pour fonctionner lors de l'import `from src.orchestrator import anonymize_text`.
"""
from typing import List, Tuple, Dict, Any, Optional
import re

from .policy import preset, AnonymizationPolicy
from ..llm.reasoner import LLMReasoner, SeedSpan, DetectionPlan
from ..llm.openrouter_client import OpenRouterClient
from ..utils.utils_pseudo import PseudoMapper
from ..utils.text_sanitizer import regexes_based_replacements
from ..services.ner import run_gliner, merge_ner_lists
from ..services.ner.ensemble import GLINER_ALL_LABELS
from ..rupta.privacy_evaluator import evaluate_reidentification_risk
from ..rupta.utility_evaluator import evaluate_utility_preservation

try:  # GPU pipeline is optional
    from ..services.ner.gpu_optimizer import create_optimized_pipeline, load_gpu_config
    _GPU_OPTIMIZER_AVAILABLE = True
except Exception:  # pragma: no cover - GPU stack not available
    _GPU_OPTIMIZER_AVAILABLE = False
    create_optimized_pipeline = None  # type: ignore
    load_gpu_config = None  # type: ignore


"""Ajout NER ensemble extrait vers ner_ensemble et amélioration dates FR/EN."""

# Pipeline GPU global (cache pour éviter de le recréer à chaque appel)
_GPU_PIPELINE = None
_GPU_CONFIG_LOADED = False


def _get_ner_pipeline():
    """
    Retourne le pipeline NER optimisé si le mode GPU est activé, sinon None.
    
    Le pipeline est créé une seule fois et mis en cache pour éviter le surcoût
    de rechargement des modèles.
    """
    global _GPU_PIPELINE, _GPU_CONFIG_LOADED
    
    if not _GPU_OPTIMIZER_AVAILABLE:
        return None
    
    # Ne charger la config qu'une seule fois
    if not _GPU_CONFIG_LOADED:
        _GPU_CONFIG_LOADED = True
        try:
            config = load_gpu_config()
            if config.get("enabled"):
                _GPU_PIPELINE = create_optimized_pipeline(config)
                if _GPU_PIPELINE:
                    print(f"[orchestrator] Mode NER GPU activé (batch_size={config.get('batch_size')}, models={config.get('max_parallel_models')})")
        except Exception as e:
            print(f"[orchestrator] Échec initialisation GPU pipeline: {e}")
            _GPU_PIPELINE = None
    
    return _GPU_PIPELINE


def reset_ner_pipeline():
    """
    Réinitialise le pipeline NER GPU.
    
    Utile si la configuration a changé et que vous voulez forcer
    le rechargement des modèles.
    """
    global _GPU_PIPELINE, _GPU_CONFIG_LOADED
    _GPU_PIPELINE = None
    _GPU_CONFIG_LOADED = False
    print("[orchestrator] Pipeline NER réinitialisé")


def get_ner_mode():
    """Retourne le mode NER actuel ('gpu' ou 'standard')."""
    pipeline = _get_ner_pipeline()
    return "gpu" if pipeline is not None else "standard"


_DEFAULT_REASONER_MODELS = {
    "detect": "openai/gpt-4.1-mini",
    "paraphrase": "openai/gpt-4.1-mini",
    "audit": "openai/gpt-4.1-mini",
}


def _resolve_reasoner_models(
    overrides: Optional[Dict[str, str]],
    config_models: Optional[Dict[str, str]],
) -> Dict[str, str]:
    models = dict(_DEFAULT_REASONER_MODELS)
    if config_models:
        models.update({k: v for k, v in config_models.items() if isinstance(v, str) and v})
    if overrides:
        models.update({k: str(v) for k, v in overrides.items() if isinstance(v, str) and v})
    return models

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

    # Step 1bis: NER — fusion externe (ner_results) + interne (GLiNER + HF)
    use_gliner = True
    gl_models = ["urchade/gliner_large-v2.1", "urchade/gliner_multi-v2.1"]
    gl_labels = GLINER_ALL_LABELS
    gl_thresh = 0.35
    ner_mode = "union"
    ner_min_votes = 1
    disable_internal = False

    if overrides:
        use_gliner = bool(overrides.get("ner_use_gliner", use_gliner))
        gl_models = list(overrides.get("gliner_models", gl_models))
        gl_labels = list(overrides.get("gliner_labels", gl_labels))
        gl_thresh = float(overrides.get("gliner_threshold", gl_thresh))
        ner_mode = str(overrides.get("ner_mode", ner_mode)).lower()
        ner_min_votes = int(overrides.get("ner_min_votes", ner_min_votes))
        disable_internal = bool(overrides.get("disable_internal_ner", disable_internal))

    local_ner = ner_results or []
    if not disable_internal:
        # Tenter d'utiliser le pipeline GPU optimisé
        gpu_pipeline = _get_ner_pipeline()
        
        if gpu_pipeline is not None:
            # Mode GPU : utilisation du pipeline optimisé
            try:
                gpu_entities = gpu_pipeline.predict(value)
                # Convertir au format attendu par merge_ner_lists
                local_ner = merge_ner_lists(local_ner, gpu_entities)
            except Exception as e:
                # Fallback vers mode standard en cas d'erreur
                print(f"[orchestrator] Erreur GPU pipeline, fallback vers mode standard: {e}")
                gl_ents = run_gliner(value, model_names=gl_models, labels=gl_labels, threshold=gl_thresh) if use_gliner else []
                local_ner = merge_ner_lists(local_ner, gl_ents)
        else:
            # Mode standard : utilisation de run_gliner et run_hf_ner_chunked
            gl_ents = run_gliner(value, model_names=gl_models, labels=gl_labels, threshold=gl_thresh) if use_gliner else []
            local_ner = merge_ner_lists(local_ner, gl_ents)

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
    ner_snapshot = [
        {
            "start": int(ent.get("start", 0)),
            "end": int(ent.get("end", 0)),
            "entity_group": str(ent.get("entity_group", "")),
            **{k: v for k, v in ent.items() if k not in {"start", "end", "entity_group"}},
        }
        for ent in (local_ner or [])
        if isinstance(ent, dict)
    ]

    # Step 2: LLM Reasoner (si activé et clé dispo)
    reasoner = None
    reps_all: List[Tuple[int, int, str, Dict[str, Any]]] = []
    llm_error: Optional[str] = None
    llm_used: bool = False
    models_used: Optional[Dict[str, str]] = None
    if policy.llm_detection:
        try:
            client = OpenRouterClient.from_config()
            models = _resolve_reasoner_models(openrouter_models, client.config_models)
            detect_model = models.get("detect") or models.get("paraphrase") or models.get("audit") or _DEFAULT_REASONER_MODELS["detect"]
            paraphrase_model = models.get("paraphrase") or detect_model
            audit_model = models.get("audit") or detect_model
            models_used = {
                "detect": detect_model,
                "paraphrase": paraphrase_model,
                "audit": audit_model,
            }
            reasoner = LLMReasoner(
                client,
                model_detect=detect_model,
                model_paraphrase=paraphrase_model,
                model_audit=audit_model,
            )
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

            reps_all = merge_non_overlapping(base_reps, llm_reps)
        except Exception as e:
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

    # Step 7: RUPTA Privacy-Utility Optimization (si activé en L1)
    rupta_metrics = {}
    if policy.rupta_enabled and reasoner:
        ground_truth_people = overrides.get("rupta_ground_truth_people") if overrides else None
        ground_truth_label = overrides.get("rupta_ground_truth_label") if overrides else None
        
        if ground_truth_people and ground_truth_label:
            try:
                # Optimisation RUPTA itérative
                best_text = text
                best_reward = 0.0
                iterations_done = 0
                
                for iteration in range(policy.rupta_max_iterations):
                    iterations_done += 1
                    
                    # Évaluer privacy
                    eval_client = reasoner.client if hasattr(reasoner, "client") else OpenRouterClient.from_config()
                    detect_for_eval = (models_used or {}).get("detect") if models_used else None
                    audit_for_eval = (models_used or {}).get("audit") if models_used else None
                    paraphrase_for_eval = (models_used or {}).get("paraphrase") if models_used else None

                    privacy_eval = evaluate_reidentification_risk(
                        client=eval_client,
                        anonymized_text=text,
                        ground_truth_people=ground_truth_people,
                        p_threshold=policy.rupta_p_threshold,
                        model=audit_for_eval or detect_for_eval
                    )

                    utility_eval = evaluate_utility_preservation(
                        client=eval_client,
                        anonymized_text=text,
                        ground_truth_label=ground_truth_label,
                        model=paraphrase_for_eval or detect_for_eval
                    )
                    
                    # Calculer la récompense combinée
                    privacy_rank = privacy_eval.get("rank", 1)
                    privacy_reward = 1.0 if privacy_rank == 999 else min(privacy_rank / policy.rupta_p_threshold, 1.0)
                    
                    utility_score = utility_eval.get("confidence_score", 0)
                    utility_correct = utility_eval.get("correct_prediction", False)
                    utility_reward = (utility_score / 100.0) if utility_correct else 0.0
                    
                    # Récompense combinée (50% privacy, 50% utility)
                    combined_reward = 0.5 * privacy_reward + 0.5 * utility_reward
                    
                    # Vérifier si on a atteint les objectifs
                    privacy_ok = privacy_rank == 999 or (policy.rupta_privacy_threshold and privacy_rank >= policy.rupta_privacy_threshold)
                    utility_ok = utility_correct and utility_score >= policy.rupta_utility_threshold
                    
                    # Si meilleur résultat, on le garde
                    if combined_reward > best_reward:
                        best_reward = combined_reward
                        best_text = text
                        rupta_metrics = {
                            "privacy": privacy_eval,
                            "utility": utility_eval,
                            "iterations": iterations_done,
                            "final_reward": best_reward,
                            "privacy_reward": privacy_reward,
                            "utility_reward": utility_reward
                        }
                    
                    # Si objectifs atteints, on arrête
                    if privacy_ok and utility_ok:
                        break
                    
                    # Sinon, on continue l'optimisation (paraphrase + ajustements)
                    if iteration < policy.rupta_max_iterations - 1:
                        # Identifier les problèmes
                        if not privacy_ok and privacy_eval.get("sensitive_entities"):
                            # Il y a des entités sensibles, on les masque davantage
                            sensitive = privacy_eval.get("sensitive_entities", [])
                            for entity in sensitive[:3]:  # Max 3 entités à traiter
                                if entity in text:
                                    # Remplacer par un placeholder plus générique
                                    text = text.replace(entity, "[REDACTED]")
                        
                        # Paraphraser légèrement pour améliorer le tradeoff
                        if reasoner and policy.llm_paraphrase:
                            try:
                                temp = 0.3 + 0.1 * iteration  # Température croissante
                                text = reasoner.paraphrase(
                                    text, 
                                    temperature=temp, 
                                    ensure_placeholders_preserved=True
                                )
                            except Exception:
                                pass  # Si paraphrase échoue, on continue
                
                # Utiliser le meilleur texte trouvé
                text = best_text
                
            except Exception as e:
                llm_error = (llm_error or "") + f"; rupta_failed: {type(e).__name__}: {e}" if llm_error else f"rupta_failed: {type(e).__name__}: {e}"
                rupta_metrics = {"error": str(e)}

    return {
        "anonymized_text": text,
        "audit": {
            "entities": applied_meta,
            "risk": risk_report,
            "llm_error": llm_error,
            "llm_used": llm_used,
            "models": models_used,
        },
        "rupta_metrics": rupta_metrics,
        "policy": policy.to_dict(),
        "ner_entities": ner_snapshot,
        "ner": {
            "mode": get_ner_mode(),
            "entities": ner_snapshot,
        },
    }