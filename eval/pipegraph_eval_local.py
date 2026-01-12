from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Span = Tuple[int, int, str]
ProgressCallback = Callable[[int, int, str], None]


def _project_root_from_eval_dir() -> str:
    # This file lives in <repo>/eval/
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def ensure_pipegraph_importable() -> str:
    """Ensure `pipegraph/` is importable and returns its absolute path."""
    project_root = _project_root_from_eval_dir()
    pipegraph_dir = os.path.join(project_root, "pipegraph")
    if pipegraph_dir not in sys.path:
        sys.path.insert(0, pipegraph_dir)
    return pipegraph_dir


def load_pipegraph() -> Tuple[Any, Any]:
    """Returns (create_pipeline_graph, create_initial_state)."""
    ensure_pipegraph_importable()
    from src.graph import create_pipeline_graph  # type: ignore
    from src.state import create_initial_state  # type: ignore

    return create_pipeline_graph, create_initial_state


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


def evaluate_spans(text: str, gt_spans: Sequence[Span], pred_spans: Sequence[Span]) -> Dict[str, Any]:
    """Entity-level overlap evaluation (same logic as Streamlit view)."""
    gt = list(gt_spans)
    preds = list(pred_spans)

    # Recall: GT covered by at least one pred
    tp_gt = 0
    fn_gt = 0
    for g in gt:
        g_start, g_end, _ = g
        covered = any(calculate_overlap((g_start, g_end), (p[0], p[1])) for p in preds)
        if covered:
            tp_gt += 1
        else:
            fn_gt += 1

    # Precision: preds overlapping at least one GT
    tp_pred = 0
    fp_pred = 0
    for p in preds:
        p_start, p_end, _ = p
        overlaps = any(calculate_overlap((p_start, p_end), (g[0], g[1])) for g in gt)
        if overlaps:
            tp_pred += 1
        else:
            fp_pred += 1

    precision = tp_pred / (tp_pred + fp_pred) if (tp_pred + fp_pred) else 0.0
    recall = tp_gt / (tp_gt + fn_gt) if (tp_gt + fn_gt) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f2": _f2(precision, recall, beta=2.0),
        "pred_count": len(preds),
        "truth_count": len(gt),
        "leaks_count": fn_gt,
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
                        best = min(occ, key=lambda p: abs(p[0] - start))
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
    initial_state = create_initial_state(text, config or {})
    return pipeline.invoke(initial_state)


def build_report(
    docs: Sequence[Tuple[str, str, List[Span]]],
    pipeline: Any,
    create_initial_state: Any,
    config: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    for idx, (doc_id, text, gt_spans) in enumerate(docs):
        if progress_cb is not None:
            try:
                progress_cb(idx, len(docs), doc_id)
            except Exception:
                pass
        final_state = run_pipegraph_on_text(pipeline, create_initial_state, text, config=config)
        pred_spans = spans_from_pipegraph_entities(final_state.get("entities", []))

        metrics = evaluate_spans(text, gt_spans, pred_spans)
        report.append(
            {
                "doc_id": doc_id,
                **metrics,
                "full_text": text,
                "text_snippet": text[:200],
                "ground_truth": gt_spans,
                "predictions": pred_spans,
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
        )
        if progress_cb is not None:
            try:
                progress_cb(idx + 1, len(docs), doc_id)
            except Exception:
                pass
    return report


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
