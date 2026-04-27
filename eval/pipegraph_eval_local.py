from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from eval.core.bootstrap import (
    ensure_pipegraph_importable as core_ensure_pipegraph_importable,
    load_pipegraph as core_load_pipegraph,
    project_root,
)
from eval.core.config import normalize_runtime_config

logger = logging.getLogger("pipegraph_eval")

Span = Tuple[int, int, str]
ProgressCallback = Callable[[int, int, str], None]
LLM_ENTITY_SOURCES = {"llm", "llm_review", "llm_verified"}


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
    downstream consumers (e.g. ``compute_label_metrics``) can aggregate
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

    # Recall: GT covered by at least one pred (ALL preds, any label)
    tp_gt = 0
    fn_gt = 0
    exact_tp_gt = 0
    exact_fn_gt = 0
    for g in gt:
        g_start, g_end, g_label = g
        overlaps = [
            p for p in all_preds
            if calculate_overlap((g_start, g_end), (p[0], p[1]))
        ]
        covered = bool(overlaps)
        if covered:
            tp_gt += 1
            tp_by_label[g_label] = tp_by_label.get(g_label, 0) + 1
        else:
            fn_gt += 1
            fn_by_label[g_label] = fn_by_label.get(g_label, 0) + 1

        exact_match = any(p[2] == g_label for p in overlaps)
        if exact_match:
            exact_tp_gt += 1
            exact_tp_by_label[g_label] = exact_tp_by_label.get(g_label, 0) + 1
        else:
            exact_fn_gt += 1
            exact_fn_by_label[g_label] = exact_fn_by_label.get(g_label, 0) + 1

    # Precision: only scoped preds overlapping at least one GT
    tp_pred = 0
    fp_pred = 0
    for p in scoped_preds:
        p_start, p_end, p_label = p
        overlaps = any(calculate_overlap((p_start, p_end), (g[0], g[1])) for g in gt)
        if overlaps:
            tp_pred += 1
        else:
            fp_pred += 1
            fp_by_label[p_label] = fp_by_label.get(p_label, 0) + 1

    precision = tp_pred / (tp_pred + fp_pred) if (tp_pred + fp_pred) else 0.0
    recall = tp_gt / (tp_gt + fn_gt) if (tp_gt + fn_gt) else 0.0
    exact_label_recall = (
        exact_tp_gt / (exact_tp_gt + exact_fn_gt)
        if (exact_tp_gt + exact_fn_gt) else 0.0
    )

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
    initial_state = create_initial_state(text, runtime_config)
    return pipeline.invoke(initial_state)


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

    if max_workers is None:
        # Auto-detect: parallel only for OpenRouter
        provider = str(runtime_config.get("llm_provider", "")).lower()
        if not provider:
            try:
                ensure_pipegraph_importable()
                from src.nodes.llm.llm_client import load_full_config  # type: ignore
                _cfg = load_full_config()
                provider = _cfg.get("llm", {}).get("provider", "lmstudio").lower()
            except Exception:
                provider = "lmstudio"
        if provider == "openrouter":
            try:
                ensure_pipegraph_importable()
                from src.nodes.llm.llm_client import load_full_config as _lcfg  # type: ignore
                _or_cfg = _lcfg().get("openrouter", {})
                max_workers = int(_or_cfg.get("max_workers", 8))
            except Exception:
                max_workers = 8
        else:
            max_workers = 1

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
        anonymized_text = final_state.get("text", text)
        metrics = evaluate_spans(text, gt_spans, pred_spans, allowed_labels=allowed_labels)
        return {
            "doc_id": doc_id,
            **metrics,
            "full_text": text,
            "anonymized_text": anonymized_text,
            "text_snippet": text[:200],
            "ground_truth": gt_spans,
            "predictions": pred_spans,
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
                for (s, e, label) in gt_spans
                if not any(calculate_overlap((s, e), (p[0], p[1])) for p in pred_spans)
            ],
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

    def _worker(idx: int) -> Tuple[int, Dict[str, Any]]:
        doc_id, text, gt_spans = docs[idx]
        return idx, _process_doc(idx, doc_id, text, gt_spans)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(docs))) as pool:
        futures = {pool.submit(_worker, i): i for i in range(len(docs))}
        for future in as_completed(futures):
            idx_done = futures[future]
            try:
                idx_res, result = future.result(timeout=300)
                report[idx_res] = result
            except TimeoutError:
                doc_id, text, gt_spans = docs[idx_done]
                logger.warning(f"build_report: document {doc_id} timed out after 300s")
                report[idx_done] = {
                    "doc_id": doc_id,
                    "error": "timeout (300s)",
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": 0.0, "recall": 0.0, "f2": 0.0,
                    "pred_count": 0, "truth_count": len(gt_spans),
                    "leaks_count": len(gt_spans),
                    "full_text": text,
                    "anonymized_text": text,
                    "text_snippet": text[:200],
                    "ground_truth": gt_spans,
                    "predictions": [],
                    "privacy_score": None,
                    "rupta_iterations": 0,
                    "llm_entities": 0,
                    "llm_feedback": {},
                    "leaks": [],
                    "tp_by_label": {},
                    "fp_by_label": {},
                    "fn_by_label": {},
                }
            except Exception as exc:
                doc_id, text, gt_spans = docs[idx_done]
                logger.warning(f"build_report: document {doc_id} failed: {exc}")
                report[idx_done] = {
                    "doc_id": doc_id,
                    "error": str(exc),
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": 0.0, "recall": 0.0, "f2": 0.0,
                    "full_text": text,
                    "anonymized_text": text,
                    "text_snippet": text[:200],
                    "ground_truth": gt_spans,
                    "predictions": [],
                    "privacy_score": None,
                    "rupta_iterations": 0,
                    "llm_entities": 0,
                    "llm_feedback": {},
                    "leaks": [],
                }
            with lock:
                completed += 1
                if progress_cb is not None:
                    try:
                        progress_cb(completed, len(docs), docs[idx_done][0])
                    except Exception:
                        pass

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
