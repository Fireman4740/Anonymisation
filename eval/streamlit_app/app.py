import streamlit as st
import json
import os
import glob
import pandas as pd
from typing import List, Dict, Any, Tuple

st.set_page_config(layout="wide", page_title="Anonymization Error Analysis")

# Adjust path to point to the reports directory relative to this script
# Script is in eval/streamlit_app/app.py
# Reports are in eval/evaluation/reports/
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "reports")

def load_report(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_overlap(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    return max(0, min(end1, end2) - max(start1, start2)) > 0

def classify_spans(text: str, gt_spans: List[Tuple[int, int, str]], pred_spans: List[Tuple[int, int, str]]):
    # Convert to list of dicts for easier handling
    # Each span: {start, end, type, category: TP/FP/FN}
    
    # 1. Identify TPs and FNs
    # TP: GT that overlaps with at least one Pred
    # FN: GT that overlaps with NO Pred
    
    tp_spans = []
    fn_spans = []
    
    # We also need to track which preds are used for TPs to find FPs
    used_preds = set()
    
    for gt in gt_spans:
        gt_start, gt_end, gt_label = gt
        is_covered = False
        for i, pred in enumerate(pred_spans):
            pred_start, pred_end, pred_label = pred
            if calculate_overlap((gt_start, gt_end), (pred_start, pred_end)):
                is_covered = True
                used_preds.add(i)
        
        if is_covered:
            tp_spans.append({"start": gt_start, "end": gt_end, "label": gt_label, "category": "TP"})
        else:
            fn_spans.append({"start": gt_start, "end": gt_end, "label": gt_label, "category": "FN"})
            
    # 2. Identify FPs
    # FP: Pred that was not used for any TP
    fp_spans = []
    for i, pred in enumerate(pred_spans):
        if i not in used_preds:
            fp_spans.append({"start": pred[0], "end": pred[1], "label": pred[2], "category": "FP"})
            
    # Combine all spans
    all_spans = tp_spans + fn_spans + fp_spans
    
    # Sort by start position
    all_spans.sort(key=lambda x: x["start"])
    
    return all_spans

def render_text_with_spans(text: str, spans: List[Dict]):
    # Colors
    colors = {
        "TP": "#d4edda", # Greenish
        "FN": "#f8d7da", # Reddish
        "FP": "#fff3cd", # Yellowish
    }
    border_colors = {
        "TP": "#28a745",
        "FN": "#dc3545",
        "FP": "#ffc107",
    }
    
    mask = [0] * len(text)
    span_info = [None] * len(text)
    
    for span in spans:
        s, e = span["start"], span["end"]
        cat = span["category"]
        val = 0
        if cat == "TP": val = 1
        elif cat == "FN": val = 2
        elif cat == "FP": val = 3
        
        # Clamp to text length
        s = max(0, s)
        e = min(len(text), e)
        
        for i in range(s, e):
            # Simple priority: FN > FP > TP
            # If a character is already marked as FN (2), don't overwrite it.
            if mask[i] == 2: continue
            # If a character is FP (3) and we want to write TP (1), keep FP? 
            # Actually, if we have overlap, it's tricky. 
            # But based on our classification, TP and FN are from GT, FP is from Pred.
            # TP and FP shouldn't overlap (because if Pred overlaps GT, it's TP).
            # So only TP/FN overlap (impossible, same GT list) or FP/FN overlap?
            # FP (Pred) and FN (GT) overlap?
            # If Pred overlaps GT, it's a TP. So FP cannot overlap FN.
            # So actually, these sets should be disjoint in terms of character coverage?
            # Wait. 
            # GT: [0, 10]
            # Pred: [5, 15]
            # Overlap: [5, 10].
            # GT is TP (covered). Pred is "used" -> not FP.
            # So we have one TP span [0, 10].
            # But what about [10, 15]? That part of Pred is not covering GT.
            # Our simple classification says: "This whole GT is TP", "This whole Pred is Used".
            # So we visualize the GT as TP [0, 10].
            # We don't visualize the Pred [5, 15] at all because it's "used".
            # This means we miss the fact that [10, 15] was redacted but not needed?
            # Yes, this is a limitation of "Entity-level" metrics vs "Token-level".
            # But for this tool, showing the GT as TP is good.
            # Showing the extra redaction would be better.
            # But let's stick to the simple classification for now.
            
            mask[i] = val
            span_info[i] = span
            
    # Now generate HTML
    html = ""
    i = 0
    while i < len(text):
        current_type = mask[i]
        current_span = span_info[i]
        
        # Find end of this segment
        j = i + 1
        while j < len(text) and mask[j] == current_type and span_info[j] == current_span:
            j += 1
            
        segment_text = text[i:j]
        
        if current_type == 0:
            html += segment_text
        else:
            cat = current_span["category"]
            label = current_span["label"]
            color = colors[cat]
            border = border_colors[cat]
            tooltip = f"{cat}: {label}"
            
            html += f'<span style="background-color: {color}; border: 1px solid {border}; border-radius: 3px; padding: 0 2px;" title="{tooltip}">{segment_text} <span style="font-size: 0.7em; font-weight: bold; opacity: 0.7;">{label}</span></span>'
            
        i = j
        
    return html

def main():
    st.sidebar.title("Configuration")
    
    # Find reports
    if not os.path.exists(REPORTS_DIR):
        st.error(f"Reports directory not found: {REPORTS_DIR}")
        return

    report_files = glob.glob(os.path.join(REPORTS_DIR, "*_details.json"))
    if not report_files:
        st.error(f"No reports found in {REPORTS_DIR}")
        return
        
    report_names = [os.path.basename(f).replace("report_", "").replace("_details.json", "") for f in report_files]
    selected_report_name = st.sidebar.selectbox("Select Dataset Report", report_names)
    
    selected_file = os.path.join(REPORTS_DIR, f"report_{selected_report_name}_details.json")
    
    try:
        data = load_report(selected_file)
    except Exception as e:
        st.error(f"Error loading report: {e}")
        return
        
    st.title(f"Analysis: {selected_report_name}")
    
    # Metrics
    total_docs = len(data)
    avg_prec = sum(d["precision"] for d in data) / total_docs if total_docs else 0
    avg_rec = sum(d["recall"] for d in data) / total_docs if total_docs else 0
    avg_f2 = sum(d["f2"] for d in data) / total_docs if total_docs else 0
    leaky_docs = sum(1 for d in data if d["leaks_count"] > 0)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Precision", f"{avg_prec:.2%}")
    c2.metric("Avg Recall", f"{avg_rec:.2%}")
    c3.metric("Avg F2", f"{avg_f2:.2%}")
    c4.metric("Leaky Docs", f"{leaky_docs} / {total_docs}")
    
    # Filters
    st.subheader("Filter Documents")
    col1, col2 = st.columns(2)
    with col1:
        recall_range = st.slider("Recall Range", 0.0, 1.0, (0.0, 1.0))
    with col2:
        show_leaks_only = st.checkbox("Show only documents with leaks")
        
    # Filter data
    filtered_data = [
        d for d in data 
        if recall_range[0] <= d["recall"] <= recall_range[1]
        and (not show_leaks_only or d["leaks_count"] > 0)
    ]
    
    st.write(f"Showing {len(filtered_data)} documents")
    
    if not filtered_data:
        st.warning("No documents match filters.")
        return
        
    # Dataframe view
    df = pd.DataFrame(filtered_data)
    # Ensure columns exist
    cols = ["doc_id", "recall", "precision", "leaks_count", "text_snippet"]
    display_cols = [c for c in cols if c in df.columns]
    
    st.dataframe(df[display_cols], use_container_width=True)
    
    # Document Inspector
    st.subheader("Document Inspector")
    
    # Select doc ID
    doc_ids = [d["doc_id"] for d in filtered_data]
    selected_doc_id = st.selectbox("Select Document ID to inspect", doc_ids)
    
    # Get doc data
    doc = next((d for d in filtered_data if d["doc_id"] == selected_doc_id), None)
    
    if doc:
        # Check if full_text is available
        text = doc.get("full_text", doc.get("text_snippet", ""))
        if "full_text" not in doc:
            st.warning("⚠️ 'full_text' not found in report. Showing truncated snippet. Re-run evaluation pipeline to get full text.")
            
        gt = doc.get("ground_truth", [])
        preds = doc.get("predictions", [])
        
        # Convert lists to tuples if needed
        gt = [tuple(x) if isinstance(x, list) else x for x in gt]
        preds = [tuple(x) if isinstance(x, list) else x for x in preds]
        
        # Classify spans
        spans = classify_spans(text, gt, preds)
        
        # Render
        html = render_text_with_spans(text, spans)
        
        st.markdown(f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6; font-family: monospace; white-space: pre-wrap;">
            {html}
        </div>
        """, unsafe_allow_html=True)
        
        # Legend
        st.markdown("""
        **Legend:** 
        <span style="background-color: #d4edda; border: 1px solid #28a745; padding: 2px 5px; border-radius: 3px;">True Positive (Correct)</span>
        <span style="background-color: #f8d7da; border: 1px solid #dc3545; padding: 2px 5px; border-radius: 3px;">False Negative (Missed)</span>
        <span style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 2px 5px; border-radius: 3px;">False Positive (Over-redacted)</span>
        """, unsafe_allow_html=True)
        
        # Details
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Ground Truth")
            st.json(gt)
        with c2:
            st.write("### Predictions")
            st.json(preds)
            
        if doc.get("leaks"):
            st.error(f"### Leaks Detected: {len(doc['leaks'])}")
            st.write(doc["leaks"])

if __name__ == "__main__":
    main()
