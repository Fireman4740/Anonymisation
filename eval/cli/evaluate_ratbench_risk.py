from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore[reportMissingImports]
try:
    from dotenv import load_dotenv  # type: ignore[reportMissingImports]
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

logger_risk = logging.getLogger("RAT-Bench-Risk")

try:
    from openai import OpenAI  # type: ignore[reportMissingImports]
except ImportError:
    OpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Jaro-Winkler similarity (pure Python)
# ---------------------------------------------------------------------------

def _jaro_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0
    s1m = [False] * len1
    s2m = [False] * len2
    matches = 0
    for i in range(len1):
        for j in range(max(0, i - match_distance), min(i + match_distance + 1, len2)):
            if s2m[j] or s1[i] != s2[j]:
                continue
            s1m[i] = s2m[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    t = 0
    k = 0
    for i in range(len1):
        if not s1m[i]:
            continue
        while not s2m[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    return (matches / len1 + matches / len2 + (matches - t / 2) / matches) / 3


def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    jaro = _jaro_similarity(s1, s2)
    prefix = sum(1 for i in range(min(len(s1), len(s2), 4)) if s1[i] == s2[i] and all(s1[j] == s2[j] for j in range(i)))
    return jaro + prefix * p * (1 - jaro)


# ---------------------------------------------------------------------------
# DOB parsing
# ---------------------------------------------------------------------------

_MONTH_NUM: Dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_NUM_MONTH: Dict[int, str] = {v: k for k, v in _MONTH_NUM.items()}


def _parse_dob(s: str) -> Optional[Dict[str, str]]:
    m = re.match(
        r"(\d{1,2})\s+(january|february|march|april|may|june|july|"
        r"august|september|october|november|december)\s+(\d{4})",
        s, re.IGNORECASE,
    )
    if m:
        return {"day": m.group(1).lstrip("0") or "0", "month": m.group(2).lower(), "year": m.group(3)}
    m = re.match(
        r"(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})",
        s, re.IGNORECASE,
    )
    if m:
        return {"day": m.group(2).lstrip("0") or "0", "month": m.group(1).lower(), "year": m.group(3)}
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return {"day": str(int(m.group(3))), "month": _NUM_MONTH.get(int(m.group(2)), m.group(2)), "year": m.group(1)}
    return None


def _dob_correctness(true_dob: str, pred_dob: str) -> float:
    tp, pp = _parse_dob(true_dob), _parse_dob(pred_dob)
    if not tp or not pp:
        return 1.0 if true_dob == pred_dob else 0.0
    correct = sum(1 for c in ("day", "month", "year") if tp.get(c, "").lower() == pp.get(c, "").lower())
    return correct / 3


# ---------------------------------------------------------------------------
# Nuanced correctness checking
# ---------------------------------------------------------------------------

def check_attribute_correctness(attr: str, true_val: str, pred_val: str) -> float:
    """Return a score in [0.0, 1.0] matching RAT-Bench evaluation.py logic."""
    tn = true_val.strip().lower()
    pn = pred_val.strip().lower()
    if not tn or not pn:
        return 0.0
    if tn == pn:
        return 1.0
    if attr == "occupation":
        return 1.0 if _jaro_winkler(tn, pn) >= 0.8 else 0.0
    if attr == "educational attainment":
        s = _jaro_winkler(tn, pn)
        return 1.0 if s >= 1.0 else (0.75 if s >= 0.75 else 0.0)
    if attr == "state of residence":
        tp = {x.strip().lower() for x in tn.split("/")}
        pp = {x.strip().lower() for x in pn.split("/")}
        return 1.0 if tp & pp else 0.0
    if attr == "date of birth":
        return _dob_correctness(tn, pn)
    if attr in ("SSN", "credit card number", "phone number"):
        return 1.0 if re.sub(r"\D", "", tn) == re.sub(r"\D", "", pn) else 0.0
    if attr in ("name", "email"):
        return 1.0 if _jaro_winkler(tn, pn) >= 1.0 else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# RAT-Bench record
# ---------------------------------------------------------------------------

@dataclass
class RatBenchRecord:
    id: str
    text: str
    profile: Dict[str, Any]
    difficulty: str
    scenario: str

def extract_ratbench_population(dataset) -> pd.DataFrame:
    """[DEPRECATED] Use _load_population() instead."""
    profiles = []
    for split in dataset.keys():
        for example in dataset[split]:
            profiles.append(example)
    pop_df, _ = _load_population(profiles)
    return pop_df


def _load_population(
    profiles: list,
    cache_dir=None,
):
    """
    Load the population DataFrame for risk calculation.

    Prefers real PUMS census data (~3.2M records).  Falls back to the
    RAT-Bench dataset profiles (~300 records) with a warning.

    Returns (population_df, is_real_pums).
    """
    import os as _os
    _cli_dir = _os.path.dirname(_os.path.abspath(__file__))
    _eval_dir = _os.path.abspath(_os.path.join(_cli_dir, ".."))
    _project_root = _os.path.abspath(_os.path.join(_eval_dir, ".."))
    _cache = cache_dir or _os.path.join(_eval_dir, "datasets", "RAT-Bench", "cache")

    try:
        import sys as _sys
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)
        from eval.cli.pums_loader import load_pums_population
        pums_df = load_pums_population(cache_dir=_cache)
        if pums_df is not None and len(pums_df) > 1000:
            return pums_df, True
    except Exception as e:
        logger_risk.warning(f"Failed to load PUMS population: {e}")

    logger_risk.warning(
        f"⚠️  Using {len(profiles)} dataset profiles as population proxy. "
        "Risk estimates will be UNRELIABLE. "
        "Download PUMS data: python -m eval.cli.pums_loader"
    )
    pop_data = []
    for p in profiles:
        profile = p.get("profile", p)
        row = {attr: profile.get(attr) for attr in RATBENCH_INDIRECT_ATTRS}
        pop_data.append(row)
    return pd.DataFrame(pop_data).drop_duplicates().reset_index(drop=True), False

def records_from_dataset(dataset, split: str = "test") -> List[RatBenchRecord]:
    """Convertit un split en Records."""
    records = []
    for example in dataset[split]:
        records.append(RatBenchRecord(
            id=str(example.get("id", "")),
            text=example["text"],
            profile=example["profile"],
            difficulty=example["difficulty"],
            scenario=example["scenario"]
        ))
    return records


RATBENCH_INDIRECT_ATTRS = [
    "citizenship status",
    "date of birth",
    "state of residence",
    "sex",                    # Fixed: was "gender"
    "race",
    "marital status",
    "educational attainment", # Fixed: was "highest level of education obtained"
    "employment status",
    "occupation",
]

RATBENCH_DIRECT_IDS: list = [
    "name",
    "email",
    "phone number",
    "SSN",
    "credit card number",
    "address",
]

def ratbench_reid_risk(
    record: RatBenchRecord,
    anonymize_fn: Callable[[str], str],
    llm_attacker_fn: Callable,
    population_df: pd.DataFrame,
    *,
    pre_anonymized_text: Optional[str] = None,
    is_real_pums: bool = False,
) -> Dict[str, Any]:
    """
    Compute R(x, t, T) as per RAT-Bench Algorithm 1-2.

    1. Anonymize (or reuse pre-anonymized text)
    2. LLM attacker infers indirect attributes (Staab per-attribute)
    3. Check direct identifier leaks
    4. Nuanced correctness checking for indirect attrs
    5. Risk via equivalence class size on PUMS population
    """
    x = record.profile
    t_a = pre_anonymized_text if pre_anonymized_text is not None else anonymize_fn(record.text)

    x_hat = llm_attacker_fn(t_a, RATBENCH_INDIRECT_ATTRS, None)

    # Check direct identifiers first
    direct_leaks: List[str] = []
    for did in RATBENCH_DIRECT_IDS:
        tv = x.get(did)
        if not tv or str(tv).strip().lower() in ("none", "n/a", "null"):
            continue
        iv = x_hat.get(did)
        if iv and check_attribute_correctness(did, str(tv), str(iv)) >= 1.0:
            direct_leaks.append(did)

    if direct_leaks:
        return {
            "risk": 1.0, "re_identified": True,
            "re_identified_via": "direct_identifier", "direct_leaks": direct_leaks,
            "num_correct_attrs": 0, "correct_attrs": [], "correct_attr_scores": {},
            "equiv_class_size": 1, "k": 1, "inferred_attrs": x_hat,
        }

    # Nuanced correctness for indirect attrs
    xstar: Dict[str, Any] = {}
    xstar_scores: Dict[str, float] = {}
    for attr in RATBENCH_INDIRECT_ATTRS:
        tv = x.get(attr)
        pv = x_hat.get(attr)
        if tv is None or pv is None:
            continue
        ts, ps = str(tv).strip(), str(pv).strip()
        if not ts or ts.lower() in ("none", "n/a", "null"):
            continue
        if not ps or ps.lower() in ("none", "n/a", "null", "unknown"):
            continue
        score = check_attribute_correctness(attr, ts, ps)
        if score > 0:
            xstar[attr] = tv
            xstar_scores[attr] = score

    if not xstar:
        return {
            "risk": 0.0, "re_identified": False, "re_identified_via": None,
            "direct_leaks": [], "num_correct_attrs": 0, "correct_attrs": [],
            "correct_attr_scores": {}, "equiv_class_size": len(population_df),
            "k": len(population_df), "inferred_attrs": x_hat,
        }

    # Filter population on fully-correct attributes
    mask = pd.Series([True] * len(population_df), index=population_df.index)
    attrs_used: List[str] = []
    for attr, value in xstar.items():
        if xstar_scores.get(attr, 0) < 1.0:
            continue
        if attr == "date of birth" or attr not in population_df.columns:
            continue
        val_norm = str(value).strip().lower()
        if attr == "state of residence":
            parts = {p.strip().lower() for p in val_norm.split("/")}
            col_lower = population_df[attr].astype(str).str.strip().str.lower()
            pm = pd.Series([False] * len(population_df), index=population_df.index)
            for part in parts:
                pm = pm | col_lower.str.contains(re.escape(part), na=False)
            mask = mask & pm
        else:
            mask = mask & (population_df[attr].astype(str).str.strip().str.lower() == val_norm)
        attrs_used.append(attr)

    k = int(mask.sum())
    risk = 1.0 if k == 0 else 1.0 / k

    return {
        "risk": risk, "re_identified": risk >= 1.0,
        "re_identified_via": "indirect_uniqueness" if risk >= 1.0 else None,
        "direct_leaks": [], "num_correct_attrs": len(xstar),
        "correct_attrs": list(xstar.keys()), "correct_attr_scores": xstar_scores,
        "attrs_used_for_k": attrs_used, "equiv_class_size": k,
        "k": k, "inferred_attrs": x_hat,
    }


def evaluate_on_ratbench(
    test_records: List[RatBenchRecord],
    anonymize_fn: Callable[[str], str],
    llm_attacker_fn: Callable,
    population_df: pd.DataFrame,
    scenario_filter: Optional[str] = None,
    pre_anonymized_texts: Optional[Dict[str, str]] = None,
    is_real_pums: bool = False,
) -> Tuple[Dict, pd.DataFrame]:
    """Evaluate as in RAT-Bench Table 3, with stratification."""
    results = []
    anon_map = pre_anonymized_texts or {}

    total = len(test_records)
    for index, record in enumerate(test_records, start=1):
        if scenario_filter and record.scenario != scenario_filter:
            continue
        pre_anon = anon_map.get(record.text)
        risk_result = ratbench_reid_risk(
            record, anonymize_fn, llm_attacker_fn, population_df,
            pre_anonymized_text=pre_anon, is_real_pums=is_real_pums,
        )
        risk_result["doc_id"] = record.id
        risk_result["full_text"] = record.text
        risk_result["difficulty"] = record.difficulty
        risk_result["scenario"] = record.scenario
        results.append(risk_result)
        if index == 1 or index == total or index % 5 == 0:
            print(f"🛡️ Risk re-identification {index}/{total}")

    if not results:
        return {}, pd.DataFrame()

    df = pd.DataFrame(results)
    metrics: Dict[str, Any] = {
        "avg_risk": float(df["risk"].mean()) if "risk" in df.columns else 0.0,
        "frac_re_identified": float((df["risk"] >= 1.0).mean()) if "risk" in df.columns else 0.0,
        "frac_high_risk_geq_0_09": float((df["risk"] >= 0.09).mean()) if "risk" in df.columns else 0.0,
        "avg_correct_attrs": float(df["num_correct_attrs"].mean()) if "num_correct_attrs" in df.columns else 0.0,
        "population_source": "PUMS" if is_real_pums else "dataset_profiles",
        "population_size": len(population_df),
    }
    if "difficulty" in df.columns:
        metrics["by_difficulty"] = df.groupby("difficulty")["risk"].agg(["mean", "count"]).to_dict("index")
    if "scenario" in df.columns:
        metrics["by_scenario"] = df.groupby("scenario")["risk"].agg(["mean", "count"]).to_dict("index")
    return metrics, df


_OPENROUTER_MODELS = [
    "xiaomi/mimo-v2-flash",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-4-maverick:free",
]

_openrouter_client: Optional[Any] = None


def _get_openrouter_client() -> Optional[Any]:
    """Lazy-init a single OpenAI client for OpenRouter."""
    global _openrouter_client
    if _openrouter_client is not None:
        return _openrouter_client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or OpenAI is None:
        return None
    _openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return _openrouter_client


def _query_openrouter(prompt: str, *, max_retries: int = 4, base_delay: float = 5.0) -> Optional[str]:
    client = _get_openrouter_client()
    if client is None:
        return None
    for model in _OPENROUTER_MODELS:
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=256,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else None
            except Exception as e:
                err = str(e)
                if ("429" in err or "rate" in err.lower()) and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                break
    return None


def _build_staab_prompt(anonymized_text: str, target_attr: str, known_attrs: Dict[str, str]) -> str:
    """Staab et al. style single-attribute inference prompt."""
    known_str = ""
    if known_attrs:
        known_str = "\n\nI already know the following attributes about the author:\n" + "\n".join(
            f"- {k}: {v}" for k, v in known_attrs.items()
        )
    return (
        f"I have the following piece of anonymized text:\n---\n{anonymized_text}\n---"
        f"{known_str}\n\n"
        f"Based on the text, what is the author's {target_attr}?\n\n"
        f'Respond with ONLY the value, nothing else. If you cannot determine it, respond with "unknown".'
    )


def openrouter_llm_attacker(
    text_anonymized: str,
    target_attrs: List[str],
    known_attrs: Optional[Dict[str, str]] = None,
    *,
    max_retries: int = 4,
    base_delay: float = 5.0,
) -> Dict[str, Any]:
    """
    LLM attacker: Staab et al. per-attribute prompting via OpenRouter.

    Queries one attribute at a time, feeding already-inferred values as
    context for subsequent queries (iterative attack).
    """
    results: Dict[str, Any] = {}
    known: Dict[str, str] = dict(known_attrs or {})

    for attr in target_attrs:
        prompt = _build_staab_prompt(text_anonymized, attr, known)
        response = _query_openrouter(prompt, max_retries=max_retries, base_delay=base_delay)
        if response and response.lower() not in ("unknown", "n/a", "none", ""):
            results[attr] = response
            known[attr] = response

    return results


def evaluate_ratbench_risk_from_pipeline(
    pipeline: Any,
    create_initial_state: Any,
    profiles: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    progress_cb: Optional[Callable] = None,
    report: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute re-identification risk for RAT-Bench profiles using PipeGraph.

    This is the entry-point meant to be called from the Streamlit service.

    Args:
        report: The span-evaluation report from ``build_ratbench_report``.
            When provided, reuses the already-anonymized texts to avoid
            running the pipeline a second time.

    Returns a dict with ``metrics`` and ``detailed_results`` keys.
    """
    from eval.core.pipeline import run_pipegraph_on_text

    cfg = config or {
        "enable_detection": True,
        "enable_deterministic": True,
        "enable_ai": True,
        "enable_anonymization": True,
    }

    def my_anonymizer(text: str) -> str:
        result_state = run_pipegraph_on_text(pipeline, create_initial_state, text, cfg)
        if isinstance(result_state, dict):
            return result_state.get("text", text)
        if hasattr(result_state, "text"):
            return result_state.text
        return text

    # Build map of original_text → anonymized_text from prior report
    pre_anonymized_texts: Dict[str, str] = {}
    if report:
        for doc in report:
            orig = doc.get("full_text", "")
            anon = doc.get("anonymized_text", "")
            if orig and anon:
                pre_anonymized_texts[orig] = anon

    population_df, is_real_pums = _load_population(profiles)

    subset = profiles[: (limit or len(profiles))]
    test_records = records_from_dataset({"test": subset}, "test")

    if not os.environ.get("OPENROUTER_API_KEY"):
        return {
            "metrics": {},
            "detailed_results": pd.DataFrame(),
            "error": "OPENROUTER_API_KEY not set — risk evaluation skipped.",
        }

    n_reused = sum(1 for r in test_records if r.text in pre_anonymized_texts)
    if n_reused:
        print(f"🛡️ Risk eval: réutilisation de {n_reused}/{len(test_records)} textes déjà anonymisés")

    metrics, detailed_results = evaluate_on_ratbench(
        test_records,
        my_anonymizer,
        openrouter_llm_attacker,
        population_df,
        pre_anonymized_texts=pre_anonymized_texts,
        is_real_pums=is_real_pums,
    )

    return {"metrics": metrics, "detailed_results": detailed_results}


def evaluate_ratbench_risk() -> None:
    from eval.core.bootstrap import project_root
    from eval.core.pipeline import load_pipegraph, run_pipegraph_on_text
    
    print("Loading PipeGraph pipeline...")
    _create_pipeline, create_initial_state = load_pipegraph()
    pipeline = _create_pipeline()
    
    config = {
        "enable_detection": True,
        "enable_deterministic": True,
        "enable_ai": True,
        "enable_anonymization": True,
    }
    
    def my_anonymizer(text: str) -> str:
        result_state = run_pipegraph_on_text(pipeline, create_initial_state, text, config)
        if isinstance(result_state, dict):
            return result_state.get("text", text)
        if hasattr(result_state, 'text'):
             return result_state.text
        return text

    print("Loading RAT-Bench dataset from HuggingFace...")
    # In order to work with the datasets library schema issue, we use the fallback download from eval.core.loaders.ratbench
    from eval.core.loaders.ratbench import load_ratbench_profiles
    profiles = load_ratbench_profiles(language="english")
    
    population_df, is_real_pums = _load_population(profiles)
    print(f"Population: {len(population_df):,} records ({'PUMS' if is_real_pums else 'DATASET PROFILES — UNRELIABLE'})")

    limit = min(50, len(profiles))
    print(f"Evaluating on {limit} records...")
    test_records = records_from_dataset({"test": profiles}, "test")[:limit]

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY not set. Attacker will return empty JSON.")

    metrics, detailed_results = evaluate_on_ratbench(
        test_records,
        my_anonymizer,
        openrouter_llm_attacker,
        population_df,
        is_real_pums=is_real_pums,
    )

    print("\n=== RAT-BENCH RE-IDENTIFICATION RISK RESULTS ===")
    print(json.dumps(metrics, indent=2, default=str))

    out_file = os.path.join(project_root(), "eval", "evaluation", "reports", "ratbench_risk_eval.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "detailed_results": detailed_results.to_dict(orient="records"),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    evaluate_ratbench_risk()
