from __future__ import annotations

from typing import Any, Dict, List, Tuple


Span = Tuple[int, int, str]


def calculate_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    start1, end1 = span1
    start2, end2 = span2
    return max(0, min(end1, end2) - max(start1, start2)) > 0


def classify_spans(text: str, gt_spans: List[Span], pred_spans: List[Span]) -> List[Dict[str, Any]]:
    # 1. Identify TPs and FNs (GT coverage)
    tp_spans: List[Dict[str, Any]] = []
    fn_spans: List[Dict[str, Any]] = []

    used_preds: set[int] = set()

    for gt_start, gt_end, gt_label in gt_spans:
        is_covered = False
        for i, (pred_start, pred_end, _) in enumerate(pred_spans):
            if calculate_overlap((gt_start, gt_end), (pred_start, pred_end)):
                is_covered = True
                used_preds.add(i)

        if is_covered:
            tp_spans.append({"start": gt_start, "end": gt_end, "label": gt_label, "category": "TP"})
        else:
            fn_spans.append({"start": gt_start, "end": gt_end, "label": gt_label, "category": "FN"})

    # 2. Identify FPs (preds not used by any TP)
    fp_spans: List[Dict[str, Any]] = []
    for i, (s, e, label) in enumerate(pred_spans):
        if i not in used_preds:
            fp_spans.append({"start": s, "end": e, "label": label, "category": "FP"})

    all_spans = tp_spans + fn_spans + fp_spans
    all_spans.sort(key=lambda x: x["start"])
    return all_spans


def render_text_with_spans(text: str, spans: List[Dict[str, Any]]) -> str:
    colors = {
        "TP": "#d4edda",
        "FN": "#f8d7da",
        "FP": "#fff3cd",
    }
    border_colors = {
        "TP": "#28a745",
        "FN": "#dc3545",
        "FP": "#ffc107",
    }

    mask = [0] * len(text)
    span_info: List[Dict[str, Any] | None] = [None] * len(text)

    for span in spans:
        s, e = int(span["start"]), int(span["end"])
        cat = str(span["category"])

        val = 0
        if cat == "TP":
            val = 1
        elif cat == "FN":
            val = 2
        elif cat == "FP":
            val = 3

        s = max(0, s)
        e = min(len(text), e)

        for i in range(s, e):
            if mask[i] == 2:
                continue
            mask[i] = val
            span_info[i] = span

    html = ""
    i = 0
    while i < len(text):
        current_type = mask[i]
        current_span = span_info[i]

        j = i + 1
        while j < len(text) and mask[j] == current_type and span_info[j] == current_span:
            j += 1

        segment_text = text[i:j]

        if current_type == 0 or current_span is None:
            html += segment_text
        else:
            cat = str(current_span["category"])
            label = str(current_span["label"])
            color = colors[cat]
            border = border_colors[cat]
            tooltip = f"{cat}: {label}"

            html += (
                f'<span style="background-color: {color}; border: 1px solid {border}; '
                f'border-radius: 3px; padding: 0 2px;" title="{tooltip}">' 
                f"{segment_text} <span style=\"font-size: 0.7em; font-weight: bold; opacity: 0.7;\">{label}</span></span>"
            )

        i = j

    return html
