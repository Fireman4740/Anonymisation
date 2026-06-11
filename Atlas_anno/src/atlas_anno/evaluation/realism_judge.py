from __future__ import annotations

import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from atlas_anno.config import load_config
from atlas_anno.console import log
from atlas_anno.constants import PROMPT_REALISM_JUDGE
from atlas_anno.llm import OpenRouterClient
from atlas_anno.prompts import load_prompt_spec
from atlas_anno.schemas import DocumentRecord, LLMRunMeta
from atlas_anno.settings import load_settings
from atlas_anno.storage import load_documents, save_report

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _build_user_prompt(document: DocumentRecord) -> str:
    register = document.scenario.register or str(document.metadata.get("register", ""))
    address_form = document.scenario.address_form or str(document.metadata.get("address_form", ""))
    return (
        f"domaine: {document.domain}\n"
        f"registre attendu: {register}\n"
        f"forme d'adresse attendue: {address_form}\n\n"
        f"{document.text}"
    )


def _validate_judgment(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected dict")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("scores must be a dict")
    for dim in ("naturalness", "register", "plausibility"):
        val = scores.get(dim)
        if not isinstance(val, (int, float)) or not (1 <= float(val) <= 5):
            raise ValueError(f"{dim} must be 1-5, got {val!r}")
    overall = payload.get("overall")
    if not isinstance(overall, (int, float)) or not (1 <= float(overall) <= 5):
        raise ValueError(f"overall must be 1-5, got {overall!r}")
    return payload


def _judge_one(
    document: DocumentRecord,
    client: OpenRouterClient,
    model: str,
    prompt_spec: Any,
) -> Tuple[Optional[Dict[str, Any]], LLMRunMeta]:
    user_prompt = _build_user_prompt(document)
    result, meta = client.complete_json(
        step_name="realism_judging",
        prompt_spec=prompt_spec,
        user_prompt=user_prompt,
        model=model,
        validator=_validate_judgment,
        fallback_value=None,
        temperature=0.0,
        allow_fallback=False,
    )
    return result, meta


def _pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    """Corrélation de Pearson entre deux listes (biais de longueur, Shaib et al.)."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in xs))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in ys))
    if sx < 1e-9 or sy < 1e-9:
        return None
    return round(cov / (sx * sy), 4)


def _resolve_model(key: str, settings: Any) -> str:
    if key == "reasoning":
        return settings.atlas_model_reasoning
    if key == "creative":
        return settings.atlas_model_creative
    return key


def apply_human_realism_sample(
    documents: Sequence[DocumentRecord],
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Marque `human_review_required=True` + raison 'realism_sample' sur le sous-ensemble.

    Déterministe : tri lexicographique des doc_ids + `random.Random(sample_seed)`.
    Retourne le nombre de documents marqués.
    """
    cfg = config or {}
    rate = float(cfg.get("human_sample_rate", 0.05))
    seed = int(cfg.get("sample_seed", 53))
    n_sample = max(0, int(len(documents) * rate))
    if n_sample == 0:
        return 0
    rng = random.Random(seed)
    sorted_ids = sorted(d.doc_id for d in documents)
    sampled_ids = set(rng.sample(sorted_ids, n_sample))
    docs_by_id = {d.doc_id: d for d in documents}
    for doc_id in sampled_ids:
        document = docs_by_id[doc_id]
        document.metadata["human_review_required"] = True
        reasons: List[str] = document.metadata.setdefault("review_reasons", [])
        if "realism_sample" not in reasons:
            reasons.append("realism_sample")
    return len(sampled_ids)


def run_judge_realism_command(mode: str, sample_rate: Optional[float] = None) -> None:
    config = load_config()
    realism_config = dict(config.defaults.get("realism", {}) or {})
    judge_config = dict(realism_config.get("judge", {}) or {})

    if mode == "disabled":
        log("Realism judging disabled — saving empty report.")
        save_report("raw", "realism", {
            "summary": {"mode": "disabled", "judged": 0, "errors": 0},
            "judgments": [],
        })
        return

    documents = load_documents(annotated=False)
    if not documents:
        documents = load_documents(annotated=True)
    if not documents:
        raise RuntimeError("Aucun document : lancez generate-dataset d'abord.")

    effective_rate = sample_rate if sample_rate is not None else float(judge_config.get("sample_rate", 1.0))
    if effective_rate < 1.0:
        n_sample = max(1, int(len(documents) * effective_rate))
        rng = random.Random(int(realism_config.get("sample_seed", 53)))
        sampled_ids = set(rng.sample(sorted(d.doc_id for d in documents), n_sample))
        documents = [d for d in documents if d.doc_id in sampled_ids]

    settings = load_settings()
    client = OpenRouterClient(settings)
    prompt_spec = load_prompt_spec(PROMPT_REALISM_JUDGE)
    model_keys: List[str] = list(judge_config.get("models", ["reasoning"]))
    max_workers = int(
        config.defaults.get("llm", {}).get("runtime", {}).get("reasoning_workers", 4)
    )

    # Accumulation des jugements par document et par modèle.
    judgments_by_doc: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
        d.doc_id: [] for d in documents
    }

    for model_key in model_keys:
        model_name = _resolve_model(model_key, settings)
        log(f"Judging with {model_key} ({len(documents)} documents)")

        def _make_worker(m: str, ps: Any):
            def _worker(doc: DocumentRecord) -> Tuple[Optional[Dict[str, Any]], LLMRunMeta]:
                return _judge_one(doc, client, m, ps)
            return _worker

        worker = _make_worker(model_name, prompt_spec)

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(worker, doc): doc for doc in documents}
            for future in as_completed(futures):
                doc = futures[future]
                try:
                    result, _meta = future.result()
                    if result is not None:
                        judgments_by_doc[doc.doc_id].append((model_key, result))
                except Exception as exc:
                    log(f"Error judging {doc.doc_id} with {model_key}: {exc}")

    # Agrégation des jugements par document.
    judgment_rows: List[Dict[str, Any]] = []
    token_counts: List[float] = []
    overall_scores: List[float] = []

    for doc in documents:
        doc_judgments = judgments_by_doc[doc.doc_id]
        tc = _token_count(doc.text)
        if not doc_judgments:
            judgment_rows.append({"doc_id": doc.doc_id, "error": True, "token_count": tc})
            continue

        avg_scores: Dict[str, float] = {}
        for dim in ("naturalness", "register", "plausibility"):
            avg_scores[dim] = round(
                sum(j["scores"][dim] for _, j in doc_judgments) / len(doc_judgments), 3
            )
        avg_overall = round(
            sum(j["overall"] for _, j in doc_judgments) / len(doc_judgments), 3
        )
        row: Dict[str, Any] = {
            "doc_id": doc.doc_id,
            "domain": doc.domain,
            "register": doc.scenario.register or doc.metadata.get("register", ""),
            "token_count": tc,
            "avg_scores": avg_scores,
            "avg_overall": avg_overall,
            "judgments": [
                {
                    "model_key": mk,
                    "rationale": j.get("rationale", ""),
                    "scores": j["scores"],
                    "overall": j["overall"],
                }
                for mk, j in doc_judgments
            ],
        }
        if len(doc_judgments) > 1:
            overalls = [j["overall"] for _, j in doc_judgments]
            row["inter_judge_range"] = round(max(overalls) - min(overalls), 3)

        judgment_rows.append(row)
        token_counts.append(float(tc))
        overall_scores.append(avg_overall)

    judged = sum(1 for r in judgment_rows if not r.get("error"))
    errors = sum(1 for r in judgment_rows if r.get("error"))
    avg_overall_all = round(
        sum(r["avg_overall"] for r in judgment_rows if not r.get("error")) / max(judged, 1),
        3,
    )
    correlation = _pearson_r(token_counts, overall_scores)

    report = {
        "summary": {
            "mode": mode,
            "judged": judged,
            "errors": errors,
            "models": model_keys,
            "avg_overall": avg_overall_all,
            "overall_token_count_pearson_r": correlation,
        },
        "judgments": judgment_rows,
    }
    save_report("raw", "realism", report)
    log(
        f"Realism: judged={judged} errors={errors} avg_overall={avg_overall_all} "
        f"pearson_r={correlation}"
    )
