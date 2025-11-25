import os
import json
import requests
import time
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from metrics import compute_anonymization_metrics, check_leakage
from evaluate_pipeline import load_dataset

# Configuration
API_URL = "http://localhost:8000/anonymize"
DATASET_PATH = "datasets/TAB/test.jsonl"
OUTPUT_FILE = "evaluation/reports/ablation_results.csv"

# Regex keys from patterns_config.yaml
REGEX_KEYS = [
    "stripe_secret", "aws_key_id", "google_api", "generic_secret",
    "iban", "bic", "ipv6", "url_nonhttp", "credit_card",
    "email_strict", "phone_international"
]

# Define Ablation Experiments
EXPERIMENTS = [
    {
        "name": "Baseline (L0)",
        "description": "Standard NER + Regex (No LLM)",
        "level": "L0",
        "overrides": {}
    },
    {
        "name": "L0 (No Regex)",
        "description": "NER Only (Regex disabled)",
        "level": "L0",
        "overrides": {"skip_regex_tags": REGEX_KEYS}
    },
    {
        "name": "L0 (No NER)",
        "description": "Regex Only (NER disabled)",
        "level": "L0",
        "overrides": {"use_gliner": False, "advanced_anonymizer_enable_ner": False}
    },
    {
        "name": "L1 (LLM Detect Only)",
        "description": "L0 + LLM Detection (No Paraphrase/RUPTA)",
        "level": "L1",
        "overrides": {
            "llm_paraphrase": False,
            "rupta_enabled": False,
            "llm_audit": False
        }
    },
    {
        "name": "L1 (Full)",
        "description": "Full Pipeline (LLM Detect + Paraphrase + RUPTA)",
        "level": "L1",
        "overrides": {}
    }
]

def call_api_custom(text: str, level: str, overrides: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = {
        "text": text,
        "level": level,
        "scope_id": "ablation_test",
        "overrides": overrides
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response: {e.response.text}")
        return None

def run_experiment(name: str, level: str, overrides: Dict[str, Any], data: List[Dict[str, Any]]) -> Dict[str, float]:
    print(f"\n🧪 Running Experiment: {name}")
    print(f"   Level: {level}")
    print(f"   Overrides: {overrides}")
    
    results = []
    success_count = 0
    
    # Use a subset for faster ablation if needed, or full dataset
    # For now, full dataset (it's small, ~127 docs)
    eval_data = data 

    for item in tqdm(eval_data, desc=name):
        text = item["text"]
        ground_truth = item["ground_truth"]
        
        api_res = call_api_custom(text, level, overrides)
        
        if not api_res:
            continue
            
        success_count += 1
        
        pred_entities = [
            (ent["start"], ent["end"], ent["etype"])
            for ent in api_res.get("audit", {}).get("entities", [])
        ]
        
        metrics = compute_anonymization_metrics(pred_entities, ground_truth, strict=False)
        
        anonymized_text = api_res.get("anonymized_text", "")
        leaks = check_leakage(anonymized_text, text, ground_truth)
        
        results.append({
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f2": metrics["f2"],
            "leaks_count": len(leaks)
        })
        
    if not results:
        return {
            "Experiment": name,
            "Precision": 0.0,
            "Recall": 0.0,
            "F2 Score": 0.0,
            "Leaks/Doc": 0.0,
            "Success Rate": 0.0
        }
        
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_f2 = sum(r["f2"] for r in results) / len(results)
    avg_leaks = sum(r["leaks_count"] for r in results) / len(results)
    
    return {
        "Experiment": name,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F2 Score": avg_f2,
        "Leaks/Doc": avg_leaks,
        "Success Rate": success_count / len(eval_data)
    }

def main():
    print("🚀 Starting Ablation Study on TAB Dataset")
    
    # Load Dataset
    try:
        data = load_dataset(DATASET_PATH, "TAB")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if not data:
        print("Dataset is empty.")
        return

    all_results = []

    for exp in EXPERIMENTS:
        metrics = run_experiment(exp["name"], exp["level"], exp["overrides"], data)
        all_results.append(metrics)
        
        # Print immediate result
        print(f"   -> F2: {metrics['F2 Score']:.2%}, Recall: {metrics['Recall']:.2%}, Precision: {metrics['Precision']:.2%}")

    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Display Table
    print("\n\n📊 ABLATION STUDY RESULTS")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: "{:.2%}".format(x) if x < 1 else "{:.2f}".format(x)))
    print("=" * 80)
    
    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
