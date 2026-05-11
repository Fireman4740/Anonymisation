from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import sacrebleu as _sacrebleu  # type: ignore[reportMissingImports]
    _SACREBLEU_AVAILABLE = True
except ImportError:
    _SACREBLEU_AVAILABLE = False

from eval.core.bootstrap import (
    ensure_pipegraph_importable as core_ensure_pipegraph_importable,
    load_pipegraph as core_load_pipegraph,
    project_root,
)
from eval.core.config import normalize_runtime_config
from eval.core.profiles import (
    apply_profile_to_config,
    mask_text_with_profile,
    profile_diagnostics,
    project_spans,
    resolve_eval_profile,
)

logger = logging.getLogger("pipegraph_eval")

Span = Tuple[int, int, str]
ProgressCallback = Callable[[int, int, str], None]
LLM_ENTITY_SOURCES = {"llm", "llm_review", "llm_verified"}
LLM_FEATURE_KEYS = ("llm_detection", "llm_verification", "llm_audit", "llm_paraphrase")


def _project_root_from_eval_dir() -> str:
    return project_root()


def ensure_pipegraph_importable() -> str:
    """Ensure `pipegraph/` is importable and returns its absolute path."""
    return core_ensure_pipegraph_importable()


def load_pipegraph() -> Tuple[Any, Any]:
    """Returns (create_pipeline_graph, create_initial_state)."""
    return core_load_pipegraph()


def calculate_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    start1, end1 = span1
    start2, end2 = span2
    return max(0, min(end1, end2) - max(start1, start2)) > 0


def _f2(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def _dedupe_spans(spans: Iterable[Span]) -> List[Span]:
    seen: set[Tuple[int, int, str]] = set()
    out: List[Span] = []
    for s in spans:
        key = (int(s[0]), int(s[1]), str(s[2]))
        if key in seen:
            continue
        seen.add(key)
        out.append((key[0], key[1], key[2]))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def compute_bleu(original: str, anonymized: str) -> float:
    """BLEU score entre le texte original et le texte anonymisé (métrique d'utilité RAT-Bench).

    Une valeur proche de 1.0 indique que le texte a peu changé (haute utilité).
    Une valeur proche de 0.0 indique une réécriture forte (faible utilité).
    """
    if not original or not anonymized:
        return 0.0
    if original == anonymized:
        return 1.0
    if _SACREBLEU_AVAILABLE:
        try:
            result = _sacrebleu.sentence_bleu(anonymized, [original])
            return round(float(result.score) / 100.0, 4)
        except Exception:
            pass
    # Fallback: bigram BLEU sans librairie
    def _ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        counts: Dict[Tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    ref_tokens = original.split()
    hyp_tokens = anonymized.split()
    if not hyp_tokens:
        return 0.0
    total_clip, total_hyp = 0, 0
    for n in (1, 2):
        ref_ng = _ngrams(ref_tokens, n)
        hyp_ng = _ngrams(hyp_tokens, n)
        clip = sum(min(c, ref_ng.get(g, 0)) for g, c in hyp_ng.items())
        total_clip += clip
        total_hyp += max(0, len(hyp_tokens) - n + 1)
    if total_hyp == 0:
        return 0.0
    import math
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) < len(ref_tokens) else 1.0
    precision = total_clip / total_hyp
    return round(bp * precision, 4)


def spans_from_pipegraph_entities(entities: Sequence[Dict[str, Any]], label_fallback: str = "PRED") -> List[Span]:
    spans: List[Span] = []
    for ent in entities:
        if "start" not in ent or "end" not in ent:
            continue
        start = int(ent["start"])
        end = int(ent["end"])
        label = ent.get("type") or ent.get("entity_type") or label_fallback
        spans.append((start, end, str(label)))
    return _dedupe_spans(spans)


def evaluate_spans(
    text: str,
    gt_spans: Sequence[Span],
    pred_spans: Sequence[Span],
    allowed_labels: Optional[frozenset] = None,
) -> Dict[str, Any]:
    """Entity-level overlap evaluation (same logic as Streamlit view).

    Returns global metrics **and** per-label TP/FP/FN breakdowns
    (``tp_by_label``, ``fp_by_label``, ``fn_by_label``) so that
    downstream consumers (e.g. ``eval.core.metrics.label_metrics``) can aggregate
    performance per entity type.

    Parameters
    ----------
    allowed_labels:
        When provided, the label scope is used **asymmetrically**:

        * **Recall** uses ALL predictions (regardless of label) so that a
          ground-truth span covered by *any* prediction is still a TP.
        * **Precision** uses only in-scope predictions, so out-of-scope
          labels (e.g. DATE on CoNLL2003) are neither TP nor FP.

        This avoids both inflating FP (penalising valid PII detections)
        and deflating recall (ignoring coverage from cross-label overlaps).
        Ground-truth spans are **never** filtered.
    """
    gt = list(gt_spans)
    all_preds = list(pred_spans)

    # Scoped predictions for precision (drop out-of-scope labels)
    if allowed_labels is not None:
        scoped_preds = [p for p in all_preds if p[2] in allowed_labels]
    else:
        scoped_preds = all_preds

    # --- Per-label counters ---
    tp_by_label: Dict[str, int] = {}
    fp_by_label: Dict[str, int] = {}
    fn_by_label: Dict[str, int] = {}
    exact_tp_by_label: Dict[str, int] = {}
    exact_fn_by_label: Dict[str, int] = {}

    # --- Strict-match counters (exact (start, end, label) triplet) ---
    # Taxonomy: Chinchor & Sundheim (1993), confirmed by CoNLL# (Rueda et al., 2024)
    strict_tp_gt = 0
    strict_fn_gt = 0
    strict_tp_pred = 0
    strict_fp_pred = 0
    n_missed = 0        # GT not covered by any pred (overlap)
    n_spurious = 0      # Scoped pred with no GT overlap
    n_boundary_error = 0  # Pred overlaps GT but offsets differ
    n_type_error = 0    # Pred has exact offsets but wrong label

    # Recall: GT covered by at least one pred (ALL preds, any label)
    tp_gt = 0
    fn_gt = 0
    exact_tp_gt = 0
    exact_fn_gt = 0
    for g in gt:
        g_start, g_end, g_label = g
        overlapping_preds = [
            p for p in all_preds
            if calculate_overlap((g_start, g_end), (p[0], p[1]))
        ]
        covered = bool(overlapping_preds)
        if covered:
            tp_gt += 1
            tp_by_label[g_label] = tp_by_label.get(g_label, 0) + 1
        else:
            fn_gt += 1
            fn_by_label[g_label] = fn_by_label.get(g_label, 0) + 1
            n_missed += 1

        exact_match = any(p[2] == g_label for p in overlapping_preds)
        if exact_match:
            exact_tp_gt += 1
            exact_tp_by_label[g_label] = exact_tp_by_label.get(g_label, 0) + 1
        else:
            exact_fn_gt += 1
            exact_fn_by_label[g_label] = exact_fn_by_label.get(g_label, 0) + 1

        # Strict recall: exact (start, end, label) match using ALL preds
        if any(p[0] == g_start and p[1] == g_end and p[2] == g_label for p in all_preds):
            strict_tp_gt += 1
        else:
            strict_fn_gt += 1

    # Precision: only scoped preds overlapping at least one GT
    tp_pred = 0
    fp_pred = 0
    gt_set = {(g[0], g[1], g[2]) for g in gt}
    for p in scoped_preds:
        p_start, p_end, p_label = p
        overlapping_gt = [g for g in gt if calculate_overlap((p_start, p_end), (g[0], g[1]))]
        if overlapping_gt:
            tp_pred += 1
            # Error classification for non-exact matches
            exact_gt_match = any(g[0] == p_start and g[1] == p_end and g[2] == p_label for g in overlapping_gt)
            if not exact_gt_match:
                same_bounds = any(g[0] == p_start and g[1] == p_end for g in overlapping_gt)
                if same_bounds:
                    n_type_error += 1
                else:
                    n_boundary_error += 1
        else:
            fp_pred += 1
            fp_by_label[p_label] = fp_by_label.get(p_label, 0) + 1
            n_spurious += 1

        # Strict precision: exact (start, end, label) match
        if (p_start, p_end, p_label) in gt_set:
            strict_tp_pred += 1
        else:
            strict_fp_pred += 1

    precision = tp_pred / (tp_pred + fp_pred) if (tp_pred + fp_pred) else 0.0
    recall = tp_gt / (tp_gt + fn_gt) if (tp_gt + fn_gt) else 0.0
    exact_label_recall = (
        exact_tp_gt / (exact_tp_gt + exact_fn_gt)
        if (exact_tp_gt + exact_fn_gt) else 0.0
    )
    strict_precision = strict_tp_pred / len(scoped_preds) if scoped_preds else 0.0
    strict_recall = strict_tp_gt / len(gt) if gt else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "exact_label_recall": exact_label_recall,
        "f2": _f2(precision, recall, beta=2.0),
        # Raw counters for robust micro aggregation across documents.
        # Precision side uses scoped predictions (post dataset filter).
        "precision_tp": tp_pred,
        "precision_fp": fp_pred,
        # Recall side uses all predictions (pre dataset filter).
        "recall_tp": tp_gt,
        "recall_fn": fn_gt,
        "exact_recall_tp": exact_tp_gt,
        "exact_recall_fn": exact_fn_gt,
        "pred_count": len(scoped_preds),
        "truth_count": len(gt),
        "leaks_count": fn_gt,
        "tp_by_label": tp_by_label,
        "fp_by_label": fp_by_label,
        "fn_by_label": fn_by_label,
        "exact_tp_by_label": exact_tp_by_label,
        "exact_fn_by_label": exact_fn_by_label,
        # Strict entity-level metrics (CoNLL-2003 / PII-Bench standard)
        "strict_precision": round(strict_precision, 4),
        "strict_recall": round(strict_recall, 4),
        "strict_f1": round(_f2(strict_precision, strict_recall, beta=1.0), 4),
        "strict_f2": round(_f2(strict_precision, strict_recall, beta=2.0), 4),
        "strict_precision_tp": strict_tp_pred,
        "strict_precision_fp": strict_fp_pred,
        "strict_recall_tp": strict_tp_gt,
        "strict_recall_fn": strict_fn_gt,
        # Error taxonomy (Chinchor & Sundheim 1993)
        "error_classification": {
            "missed": n_missed,
            "spurious": n_spurious,
            "boundary_error": n_boundary_error,
            "type_error": n_type_error,
        },
    }


def find_all_occurrences(text: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
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


def find_all_occurrences_case_insensitive(text: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    lower_text = text.lower()
    lower_needle = needle.lower()
    out: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = lower_text.find(lower_needle, start)
        if idx == -1:
            break
        out.append((idx, idx + len(needle)))
        start = idx + 1
    return out


def load_anonymization_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = raw.get("examples", [])
    if not isinstance(examples, list):
        raise ValueError("Invalid anonymization_dataset.json: expected key 'examples' as list")
    return examples


def gt_spans_from_anonymization_example(example: Dict[str, Any]) -> List[Span]:
    """Retourne les spans GT depuis `annotations`.

    Important: certains fichiers de dataset ont des offsets `start/end` désynchronisés du
    `original_text` (texte modifié après annotation). Quand le champ `text` est présent
    dans l'annotation, on recalcule start/end en cherchant la sous-chaîne dans le texte.
    """

    text = str(example.get("original_text", ""))
    spans: List[Span] = []

    for ann in example.get("annotations", []) or []:
        if not isinstance(ann, dict):
            continue

        label = str(ann.get("type", "SENSITIVE"))
        ann_text = ann.get("text")

        # Read provided offsets if present
        start_raw = ann.get("start")
        end_raw = ann.get("end")
        start = int(start_raw) if start_raw is not None else None
        end = int(end_raw) if end_raw is not None else None

        # If we have a text snippet, validate/fix offsets against the current original_text
        if isinstance(ann_text, str) and ann_text:
            current_slice_ok = (
                start is not None
                and end is not None
                and 0 <= start <= end <= len(text)
                and text[start:end] == ann_text
            )
            if not current_slice_ok:
                occ = find_all_occurrences(text, ann_text)
                if not occ:
                    occ = find_all_occurrences_case_insensitive(text, ann_text)

                if occ:
                    if start is None:
                        start, end = occ[0]
                    else:
                        # Choose occurrence closest to the originally provided start
                        original_start = start
                        best = min(occ, key=lambda p: abs(p[0] - original_start))
                        start, end = best

        if start is None or end is None:
            continue
        if start < 0 or end < 0 or start >= end or end > len(text):
            continue

        spans.append((start, end, label))

    return _dedupe_spans(spans)


def iter_tab_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def gt_spans_from_tab_record(record: Dict[str, Any]) -> List[Span]:
    text = str(record.get("text", ""))
    meta = record.get("meta", {}) or {}
    masked_entities = meta.get("masked_entities", []) or []

    spans: List[Span] = []
    for ent in masked_entities:
        ent_str = str(ent)
        for start, end in find_all_occurrences(text, ent_str):
            spans.append((start, end, "SENSITIVE"))

    return _dedupe_spans(spans)


def gt_spans_from_db_bio_record(record: Dict[str, Any]) -> List[Span]:
    text = str(record.get("text", ""))
    people = record.get("people")

    if isinstance(people, list):
        candidates = [str(x) for x in people if str(x).strip()]
    else:
        candidates = [str(people).strip()] if people is not None else []

    # Fallback: wiki_name -> "First_Last"
    if not candidates:
        wiki = str(record.get("wiki_name", "")).strip()
        if wiki:
            candidates = [wiki.replace("_", " ")]

    def _fuzzy_first_last_occurrences(full_name: str) -> List[Tuple[int, int]]:
        parts = [p for p in full_name.split() if p]
        if len(parts) < 2:
            return []
        first = parts[0]
        last = parts[-1]
        # Match "First ... Last" with a bounded gap (handles middle names/nicknames).
        # The span covers from start of First to end of Last.
        max_gap = 80
        pattern = re.compile(
            rf"(?<!\\w){re.escape(first)}(?!\\w).{{0,{max_gap}}}?(?<!\\w){re.escape(last)}(?!\\w)",
            re.IGNORECASE,
        )
        return [(m.start(), m.end()) for m in pattern.finditer(text)]

    spans: List[Span] = []
    for name in candidates:
        if not name:
            continue
        occ = find_all_occurrences(text, name)
        if not occ:
            occ = find_all_occurrences_case_insensitive(text, name)
        if not occ:
            occ = _fuzzy_first_last_occurrences(name)
        for start, end in occ:
            spans.append((start, end, "PERSON"))

    return _dedupe_spans(spans)


def run_pipegraph_on_text(
    pipeline: Any,
    create_initial_state: Any,
    text: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime_config = normalize_runtime_config(config)
    if runtime_config.get("dataset_key") or runtime_config.get("profile") or runtime_config.get("eval_profile"):
        runtime_config = apply_profile_to_config(
            runtime_config,
            dataset_key=runtime_config.get("dataset_key"),
            profile_name=runtime_config.get("profile") or runtime_config.get("eval_profile") or "auto",
            eval_mode=runtime_config.get("eval_mode"),
            masking_mode=runtime_config.get("masking_mode"),
        )
    initial_state = create_initial_state(text, runtime_config)
    return pipeline.invoke(initial_state)


def _load_pipegraph_config_safe() -> Dict[str, Any]:
    try:
        ensure_pipegraph_importable()
        from src.nodes.llm.llm_client import load_full_config  # type: ignore

        cfg = load_full_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _resolve_llm_provider(runtime_config: Dict[str, Any]) -> str:
    provider = str(runtime_config.get("llm_provider") or "").strip().lower()
    if provider:
        return provider
    cfg = _load_pipegraph_config_safe()
    return str((cfg.get("llm") or {}).get("provider") or "lmstudio").strip().lower()


def _llm_enabled(runtime_config: Dict[str, Any]) -> bool:
    if bool(runtime_config.get("disable_llm")):
        return False
    configured_flags = [runtime_config[key] for key in LLM_FEATURE_KEYS if key in runtime_config]
    if configured_flags:
        return any(bool(value) for value in configured_flags)
    return True


def _gliner_gpu_enabled(runtime_config: Dict[str, Any]) -> bool:
    if runtime_config.get("enable_ai") is False:
        return False

    provider = str(runtime_config.get("ner_provider") or "gliner").lower()
    if provider != "gliner":
        return False

    cfg = _load_pipegraph_config_safe()
    detection_cfg = (((cfg.get("pipeline") or {}).get("nodes") or {}).get("detection") or {})
    ai_cfg = (detection_cfg.get("ai_ner") or {}) if isinstance(detection_cfg, dict) else {}
    gliner_cfg = (ai_cfg.get("gliner") or {}) if isinstance(ai_cfg, dict) else {}
    ner_gpu_cfg = cfg.get("ner_gpu") or {}

    return bool(gliner_cfg.get("use_gpu") or ner_gpu_cfg.get("enabled"))


def resolve_doc_workers(
    config: Optional[Dict[str, Any]],
    requested_workers: Optional[int] = None,
    *,
    doc_count: Optional[int] = None,
) -> int:
    """Resolve document-level benchmark concurrency.

    ``requested_workers=None`` means auto mode. Explicit worker counts are
    honored and only capped to the number of documents.
    """
    doc_limit = int(doc_count) if doc_count is not None else None
    if requested_workers is not None:
        workers = max(1, int(requested_workers))
        return min(workers, doc_limit) if doc_limit and doc_limit > 0 else workers

    runtime_config = normalize_runtime_config(config)
    provider = _resolve_llm_provider(runtime_config)
    workers = 1
    if provider == "openrouter" and _llm_enabled(runtime_config):
        workers = 2 if _gliner_gpu_enabled(runtime_config) else 4

    return min(workers, doc_limit) if doc_limit and doc_limit > 0 else workers


def build_report(
    docs: Sequence[Tuple[str, str, List[Span]]],
    pipeline: Any,
    create_initial_state: Any,
    config: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    max_workers: Optional[int] = None,
    allowed_labels: Optional[frozenset] = None,
) -> List[Dict[str, Any]]:
    """Build an evaluation report by running the pipeline on each document.

    When ``max_workers`` > 1 **and** the LLM provider is ``openrouter``
    (detected via config), documents are processed in parallel threads to
    maximise API throughput.  Each LLM call already has its own retry logic
    with exponential back-off (see ``LLMClient.chat``).

    For local LM Studio (single-GPU), parallelism is capped at 1 to avoid
    contention.
    """
    # --- Determine effective concurrency -----------------------------------
    runtime_config = normalize_runtime_config(config)
    if runtime_config.get("dataset_key") or runtime_config.get("profile") or runtime_config.get("eval_profile"):
        runtime_config = apply_profile_to_config(
            runtime_config,
            dataset_key=runtime_config.get("dataset_key"),
            profile_name=runtime_config.get("profile") or runtime_config.get("eval_profile") or "auto",
            eval_mode=runtime_config.get("eval_mode"),
            masking_mode=runtime_config.get("masking_mode"),
        )
    active_profile = resolve_eval_profile(
        runtime_config.get("profile") or runtime_config.get("eval_profile") or "auto",
        dataset_key=runtime_config.get("dataset_key"),
    )
    eval_mode = str(runtime_config.get("eval_mode") or "both").lower()
    masking_mode = str(runtime_config.get("masking_mode") or "benchmark").lower()
    benchmark_allowed_labels = allowed_labels if allowed_labels is not None else active_profile.allowed_labels

    max_workers = resolve_doc_workers(runtime_config, max_workers, doc_count=len(docs))
    runtime_config["doc_workers"] = max_workers

    use_parallel = max_workers > 1 and len(docs) > 1
    if use_parallel:
        logger.info(
            f"build_report: parallel mode — {max_workers} workers, "
            f"{len(docs)} documents"
        )

    # --- Worker function (processes one document) --------------------------
    def _process_doc(
        idx: int, doc_id: str, text: str, gt_spans: List[Span]
    ) -> Dict[str, Any]:
        final_state = run_pipegraph_on_text(
            pipeline, create_initial_state, text, config=runtime_config
        )
        pred_spans = spans_from_pipegraph_entities(
            final_state.get("entities", [])
        )
        pipeline_anonymized_text = final_state.get("text", text)

        canonical_gt = project_spans(gt_spans, active_profile, target="canonical")
        canonical_predictions = project_spans(pred_spans, active_profile, target="canonical")
        benchmark_gt = project_spans(gt_spans, active_profile, target="benchmark")
        benchmark_predictions = project_spans(pred_spans, active_profile, target="benchmark")

        canonical_metrics = evaluate_spans(
            text, canonical_gt, canonical_predictions, allowed_labels=None
        )
        benchmark_metrics = evaluate_spans(
            text, benchmark_gt, benchmark_predictions, allowed_labels=benchmark_allowed_labels
        )

        if eval_mode == "canonical":
            metrics = canonical_metrics
            display_gt = canonical_gt
            display_predictions = canonical_predictions
        else:
            metrics = benchmark_metrics
            display_gt = benchmark_gt
            display_predictions = benchmark_predictions

        benchmark_anonymized_text, masked_counts = mask_text_with_profile(
            text, benchmark_predictions, active_profile
        )
        anonymized_text = (
            benchmark_anonymized_text
            if masking_mode == "benchmark"
            else pipeline_anonymized_text
        )
        return {
            "doc_id": doc_id,
            **metrics,
            "bleu_score": compute_bleu(text, anonymized_text),
            "full_text": text,
            "anonymized_text": anonymized_text,
            "pipeline_anonymized_text": pipeline_anonymized_text,
            "benchmark_anonymized_text": benchmark_anonymized_text,
            "text_snippet": text[:200],
            "raw_ground_truth": gt_spans,
            "ground_truth": display_gt,
            "predictions": display_predictions,
            "raw_predictions": pred_spans,
            "canonical_ground_truth": canonical_gt,
            "canonical_predictions": canonical_predictions,
            "benchmark_ground_truth": benchmark_gt,
            "benchmark_predictions": benchmark_predictions,
            "canonical_metrics": canonical_metrics,
            "benchmark_metrics": benchmark_metrics,
            "eval_mode": eval_mode,
            "masking_mode": masking_mode,
            "masking_profile": runtime_config.get("masking_profile", active_profile.name),
            "masking_counts": masked_counts,
            "effective_config": runtime_config,
            "runtime_diagnostics": {
                "eval_profile": active_profile.name,
                "dataset_key": runtime_config.get("dataset_key"),
                "llm_detection": runtime_config.get("llm_detection"),
                "llm_verification": runtime_config.get("llm_verification"),
                "llm_audit": runtime_config.get("llm_audit"),
                "llm_paraphrase": runtime_config.get("llm_paraphrase"),
                "rupta_enabled": runtime_config.get("rupta_enabled"),
                "llm_provider": runtime_config.get("llm_provider"),
                "llm_model": runtime_config.get("llm_model"),
                "entity_profile": runtime_config.get("entity_profile"),
                "gliner_label_profile": runtime_config.get("gliner_label_profile"),
                "gliner_labels": runtime_config.get("gliner_labels"),
                "masking_profile": runtime_config.get("masking_profile"),
            },
            "profile_diagnostics": profile_diagnostics(active_profile),
            "entity_counts": {
                "raw_predictions": len(pred_spans),
                "canonical_predictions": len(canonical_predictions),
                "benchmark_predictions": len(benchmark_predictions),
                "ground_truth": len(gt_spans),
            },
            # LLM / RUPTA
            "privacy_score": final_state.get("privacy_score"),
            "rupta_iterations": final_state.get("iteration", 0),
            "llm_entities": len(
                [
                    e for e in final_state.get("entities", [])
                    if str(e.get("source", "")).lower() in LLM_ENTITY_SOURCES
                ]
            ),
            "llm_feedback": final_state.get("llm_feedback") or {},
            "leaks": [
                {
                    "start": s,
                    "end": e,
                    "label": label,
                    "text": text[s:e],
                }
                for (s, e, label) in display_gt
                if not any(calculate_overlap((s, e), (p[0], p[1])) for p in display_predictions)
            ],
        }

    def _error_document_result(idx: int, error: str) -> Dict[str, Any]:
        doc_id, text, gt_spans = docs[idx]
        return {
            "doc_id": doc_id,
            "error": error,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f2": 0.0,
            "pred_count": 0,
            "truth_count": len(gt_spans),
            "leaks_count": len(gt_spans),
            "full_text": text,
            "anonymized_text": text,
            "pipeline_anonymized_text": text,
            "benchmark_anonymized_text": text,
            "text_snippet": text[:200],
            "raw_ground_truth": gt_spans,
            "ground_truth": gt_spans,
            "predictions": [],
            "raw_predictions": [],
            "canonical_ground_truth": gt_spans,
            "canonical_predictions": [],
            "benchmark_ground_truth": gt_spans,
            "benchmark_predictions": [],
            "canonical_metrics": {},
            "benchmark_metrics": {},
            "eval_mode": eval_mode,
            "masking_mode": masking_mode,
            "masking_profile": runtime_config.get("masking_profile", active_profile.name),
            "masking_counts": {},
            "effective_config": runtime_config,
            "runtime_diagnostics": {
                "eval_profile": active_profile.name,
                "dataset_key": runtime_config.get("dataset_key"),
                "llm_provider": runtime_config.get("llm_provider"),
                "llm_model": runtime_config.get("llm_model"),
                "doc_workers": max_workers,
            },
            "profile_diagnostics": profile_diagnostics(active_profile),
            "entity_counts": {
                "raw_predictions": 0,
                "canonical_predictions": 0,
                "benchmark_predictions": 0,
                "ground_truth": len(gt_spans),
            },
            "privacy_score": None,
            "rupta_iterations": 0,
            "llm_entities": 0,
            "llm_feedback": {},
            "leaks": [],
            "tp_by_label": {},
            "fp_by_label": {},
            "fn_by_label": {},
        }

    # --- Sequential path (LM Studio / single worker) ----------------------
    if not use_parallel:
        report: List[Dict[str, Any]] = []
        for idx, (doc_id, text, gt_spans) in enumerate(docs):
            if progress_cb is not None:
                try:
                    progress_cb(idx, len(docs), doc_id)
                except Exception:
                    pass
            report.append(_process_doc(idx, doc_id, text, gt_spans))
            if progress_cb is not None:
                try:
                    progress_cb(idx + 1, len(docs), doc_id)
                except Exception:
                    pass
        return report

    # --- Parallel path (OpenRouter) ----------------------------------------
    report = [None] * len(docs)  # type: ignore[list-item]
    completed = 0
    lock = threading.Lock()
    doc_timeout_s = float(runtime_config.get("doc_timeout_seconds") or 300)

    def _worker(idx: int) -> Tuple[int, Dict[str, Any]]:
        doc_id, text, gt_spans = docs[idx]
        return idx, _process_doc(idx, doc_id, text, gt_spans)

    def _record_progress(idx_done: int) -> None:
        nonlocal completed
        with lock:
            completed += 1
            current = completed
        if progress_cb is not None:
            try:
                progress_cb(current, len(docs), docs[idx_done][0])
            except Exception:
                pass

    pool = ThreadPoolExecutor(max_workers=min(max_workers, len(docs)))
    try:
        futures = {pool.submit(_worker, i): i for i in range(len(docs))}
        started_at = {future: time.monotonic() for future in futures}
        pending = set(futures)

        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

            for future in done:
                idx_done = futures[future]
                try:
                    idx_res, result = future.result()
                    report[idx_res] = result
                except Exception as exc:
                    logger.warning(f"build_report: document {docs[idx_done][0]} failed: {exc}")
                    report[idx_done] = _error_document_result(idx_done, str(exc))
                _record_progress(idx_done)

            now = time.monotonic()
            timed_out = [
                future
                for future in pending
                if now - started_at[future] >= doc_timeout_s
            ]
            for future in timed_out:
                pending.remove(future)
                idx_done = futures[future]
                future.cancel()
                doc_id = docs[idx_done][0]
                logger.warning(f"build_report: document {doc_id} timed out after {doc_timeout_s:.0f}s")
                report[idx_done] = _error_document_result(
                    idx_done, f"timeout ({doc_timeout_s:.0f}s)"
                )
                _record_progress(idx_done)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return [r for r in report if r is not None]  # type: ignore


def build_docs_from_anonymization_dataset(dataset_path: str, limit: Optional[int] = None) -> List[Tuple[str, str, List[Span]]]:
    examples = load_anonymization_dataset(dataset_path)
    out: List[Tuple[str, str, List[Span]]] = []
    for ex in examples[: (limit or len(examples))]:
        doc_id = str(ex.get("id", len(out)))
        text = str(ex.get("original_text", ""))
        gt = gt_spans_from_anonymization_example(ex)
        out.append((doc_id, text, gt))
    return out


def build_docs_from_tab(dataset_path: str, limit: Optional[int] = None) -> List[Tuple[str, str, List[Span]]]:
    out: List[Tuple[str, str, List[Span]]] = []
    for rec in iter_tab_jsonl(dataset_path):
        if limit is not None and len(out) >= limit:
            break
        doc_id = str(rec.get("doc_id", len(out)))
        text = str(rec.get("text", ""))
        gt = gt_spans_from_tab_record(rec)
        out.append((doc_id, text, gt))
    return out


def build_docs_from_db_bio(dataset_path: str, limit: Optional[int] = None) -> List[Tuple[str, str, List[Span]]]:
    out: List[Tuple[str, str, List[Span]]] = []
    for rec in iter_tab_jsonl(dataset_path):
        if limit is not None and len(out) >= limit:
            break
        doc_id = str(rec.get("wiki_name", rec.get("people", len(out))))
        text = str(rec.get("text", ""))
        gt = gt_spans_from_db_bio_record(rec)
        out.append((doc_id, text, gt))
    return out


# PersonalReddit attribute → canonical PII label mapping
_PERSONALREDDIT_ATTR_LABEL: Dict[str, str] = {
    "age": "AGE",
    "sex": "DEMOGRAPHIC",
    "city_country": "LOCATION",
    "birth_city_country": "LOCATION",
    "education": "OCCUPATION",
    "occupation": "OCCUPATION",
    "income": "AMOUNT",
    "income_level": "DEMOGRAPHIC",
    "relationship_status": "DEMOGRAPHIC",
}


def _gt_spans_from_personalreddit(record: Dict[str, Any]) -> List[Span]:
    """Extract GT spans from a PersonalReddit record.

    Searches for each personality attribute value in the response text.
    Label is mapped to a canonical PII type.
    """
    text = str(record.get("response", ""))
    personality = record.get("personality") or {}
    spans: List[Span] = []
    seen: set[Tuple[int, int]] = set()
    for attr, value in personality.items():
        if not value:
            continue
        label = _PERSONALREDDIT_ATTR_LABEL.get(attr, "SENSITIVE_ATTR")
        needle = str(value).strip()
        if not needle or needle.lower() in ("unknown", "none", "n/a"):
            continue
        for start, end in find_all_occurrences_case_insensitive(text, needle):
            if (start, end) not in seen:
                seen.add((start, end))
                spans.append((start, end, label))
    spans.sort(key=lambda s: (s[0], s[1]))
    return spans


def build_docs_from_personalreddit(dataset_path: str, limit: Optional[int] = None) -> List[Tuple[str, str, List[Span]]]:
    """Load PersonalReddit synthetic dataset (eth-sri/llmprivacy via RUPTA).

    Expected JSONL format: each line has ``response`` (text) and ``personality`` (dict of attributes).
    Ground-truth spans are constructed by fuzzy-matching attribute values in the response.
    """
    out: List[Tuple[str, str, List[Span]]] = []
    with open(dataset_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(rec.get("response", ""))
            if not text:
                continue
            doc_id = f"personalreddit_{i}"
            gt = _gt_spans_from_personalreddit(rec)
            out.append((doc_id, text, gt))
    return out
