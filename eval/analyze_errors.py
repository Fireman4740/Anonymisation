import json
import os
import sys
import glob
from typing import List, Dict, Any

OUTPUT_DIR = "evaluation/reports"

def load_report(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_html_report(dataset_name: str, data: List[Dict[str, Any]], output_file: str):
    if not data:
        print(f"No data for {dataset_name}")
        return

    # Global metrics
    total_docs = len(data)
    avg_precision = sum(d["precision"] for d in data) / total_docs
    avg_recall = sum(d["recall"] for d in data) / total_docs
    avg_f2 = sum(d["f2"] for d in data) / total_docs
    leaky_docs = [d for d in data if d["leaks_count"] > 0]
    
    # Worst performers (low recall)
    # Sort by recall ascending
    sorted_by_recall = sorted(data, key=lambda x: x["recall"])
    worst_recall = sorted_by_recall[:10]
    
    # False Positives (high pred_count but low precision)
    # Approximation: if precision is low and pred_count > 0
    false_positives = [d for d in data if d["precision"] < 0.5 and d["pred_count"] > 0]
    false_positives = sorted(false_positives, key=lambda x: x["precision"])[:10]

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Anonymization Error Analysis - {dataset_name}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .metric-box {{ display: inline-block; padding: 15px; background: #f0f0f0; margin: 10px; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f2f2f2; }}
            .snippet {{ font-family: monospace; background: #f9f9f9; padding: 5px; white-space: pre-wrap; }}
            .tag {{ display: inline-block; padding: 2px 5px; border-radius: 3px; font-size: 12px; margin-right: 5px; color: white; }}
            .tag-truth {{ background-color: #28a745; }}
            .tag-pred {{ background-color: #007bff; }}
            .tag-leak {{ background-color: #dc3545; }}
        </style>
    </head>
    <body>
        <h1>Analysis Report: {dataset_name}</h1>
        
        <div>
            <div class="metric-box">
                <div>Documents</div>
                <div class="metric-value">{total_docs}</div>
            </div>
            <div class="metric-box">
                <div>Avg Precision</div>
                <div class="metric-value">{avg_precision:.2%}</div>
            </div>
            <div class="metric-box">
                <div>Avg Recall</div>
                <div class="metric-value">{avg_recall:.2%}</div>
            </div>
            <div class="metric-box">
                <div>Avg F2</div>
                <div class="metric-value">{avg_f2:.2%}</div>
            </div>
            <div class="metric-box">
                <div>Leaky Docs</div>
                <div class="metric-value">{len(leaky_docs)}</div>
            </div>
        </div>

        <h2>Worst Recall (Missed Entities)</h2>
        <table>
            <tr>
                <th>Doc ID</th>
                <th>Recall</th>
                <th>Snippet</th>
                <th>Ground Truth (Missed?)</th>
                <th>Predictions</th>
            </tr>
            {_rows_to_html(worst_recall)}
        </table>

        <h2>Potential False Positives (Low Precision)</h2>
        <table>
            <tr>
                <th>Doc ID</th>
                <th>Precision</th>
                <th>Snippet</th>
                <th>Ground Truth</th>
                <th>Predictions (Wrong?)</th>
            </tr>
            {_rows_to_html(false_positives, metric="precision")}
        </table>
        
        <h2>Leaky Documents</h2>
        <table>
            <tr>
                <th>Doc ID</th>
                <th>Leaks Count</th>
                <th>Snippet</th>
                <th>Leaks Detected</th>
            </tr>
            {_leak_rows_to_html(leaky_docs)}
        </table>

    </body>
    </html>
    """
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report generated: {output_file}")

def _rows_to_html(rows, metric="recall"):
    html_rows = ""
    for row in rows:
        truth_html = "".join([f"<div class='tag tag-truth'>{t}</div>" for t in row['ground_truth']])
        pred_html = "".join([f"<div class='tag tag-pred'>{p}</div>" for p in row['predictions']])
        
        html_rows += f"""
        <tr>
            <td>{row['doc_id']}</td>
            <td>{row[metric]:.2f}</td>
            <td><div class="snippet">{row['text_snippet']}...</div></td>
            <td>{truth_html}</td>
            <td>{pred_html}</td>
        </tr>
        """
    return html_rows

def _leak_rows_to_html(rows):
    html_rows = ""
    for row in rows:
        leaks_html = "".join([f"<div class='tag tag-leak'>{l}</div>" for l in row['leaks']])
        
        html_rows += f"""
        <tr>
            <td>{row['doc_id']}</td>
            <td>{row['leaks_count']}</td>
            <td><div class="snippet">{row['text_snippet']}...</div></td>
            <td>{leaks_html}</td>
        </tr>
        """
    return html_rows

if __name__ == "__main__":
    report_files = glob.glob(os.path.join(OUTPUT_DIR, "*_details.json"))
    
    if not report_files:
        print(f"No detailed JSON reports found in {OUTPUT_DIR}. Run evaluate_pipeline.py first.")
        sys.exit(1)
        
    print("Found reports:")
    for i, f in enumerate(report_files):
        print(f"{i+1}. {os.path.basename(f)}")
        
    for f in report_files:
        dataset_name = os.path.basename(f).replace("report_", "").replace("_details.json", "")
        output_html = f.replace(".json", ".html")
        print(f"Generating report for {dataset_name}...")
        data = load_report(f)
        generate_html_report(dataset_name, data, output_html)
