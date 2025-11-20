import json
import os
import time

import pandas as pd
import requests
from requests import RequestException
from tqdm import tqdm

from metrics import compute_anonymization_metrics, check_leakage

# CONFIGURATION
API_URL = "http://localhost:8000/anonymize"
OUTPUT_DIR = "evaluation/reports"
REQUEST_TIMEOUT = 60
MAX_API_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2
CONSECUTIVE_FAILURE_LIMIT = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)
session = requests.Session()

def call_api(text):
    """Appelle l'API d'anonymisation avec retries et backoff."""
    payload = {
        "text": text,
        "level": "L1",  # Ou L0 selon ce que tu veux tester
        "scope_id": "eval_run"
    }

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = session.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Erreur API (tentative {attempt}/{MAX_API_RETRIES}): {e}")
            if attempt < MAX_API_RETRIES:
                sleep_time = RETRY_BACKOFF_SECONDS * attempt
                time.sleep(sleep_time)
            else:
                return None

def load_dataset(file_path, dataset_type):
    """
    Charge et normalise les données.
    Adapte cette fonction selon la structure exacte de tes JSONs TAB/Reddit.
    """
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Si c'est un JSONL (une ligne par JSON)
        if file_path.endswith('.jsonl'):
            for line in f:
                items.append(json.loads(line))
        # Si c'est un gros fichier JSON
        else:
            data = json.load(f)
            # Si TAB est sous forme de liste dans une clé 'documents' ou direct liste
            items = data if isinstance(data, list) else data.get('documents', [])

    normalized = []
    for item in items:
        # ADAPTATION DES CHAMPS SELON LE DATASET
        text = item.get('text') or item.get('content') or item.get('doc_text')
        
        # Récupération de la vérité terrain (spans)
        # On attend une liste de [start, end, label] ou dictionnaires
        labels = item.get('spans') or item.get('entities') or item.get('labels') or []
        
        # Normalisation des labels en format (start, end, type)
        norm_labels = []
        for l in labels:
            if isinstance(l, list): # Format [start, end, type]
                norm_labels.append(tuple(l))
            elif isinstance(l, dict): # Format {"start": 0, "end": 10, "label": "PER"}
                norm_labels.append((l['start'], l['end'], l['label']))
        
        if text:
            normalized.append({"text": text, "ground_truth": norm_labels})
            
    return normalized

def run_evaluation(dataset_name, file_path):
    print(f"\n🚀 Démarrage de l'évaluation pour : {dataset_name}")
    data = load_dataset(file_path, dataset_name)
    
    global_metrics = {"tp": 0, "fp": 0, "fn": 0}
    results = []
    success_count = 0
    skipped_due_to_api = 0
    failure_streak = 0
    
    for i, item in enumerate(tqdm(data)):
        text = item['text']
        ground_truth = item['ground_truth']
        
        # 1. Appel API
        api_res = call_api(text)
        if not api_res:
            skipped_due_to_api += 1
            failure_streak += 1
            if failure_streak >= CONSECUTIVE_FAILURE_LIMIT:
                print(
                    f"❌ Arrêt de l'évaluation {dataset_name}: "
                    f"{CONSECUTIVE_FAILURE_LIMIT} échecs API consécutifs."
                )
                break
            continue
        failure_streak = 0
        success_count += 1
            
        # 2. Récupération des prédictions de l'API
        # L'API renvoie result["audit"]["entities"] avec start/end
        pred_entities = []
        for ent in api_res.get("audit", {}).get("entities", []):
            pred_entities.append((ent['start'], ent['end'], ent['etype']))
            
        # 3. Calcul des métriques pour ce document
        # On utilise strict=False car si l'API masque "Jean Dupont" et la vérité est "Dupont", c'est bon.
        metrics = compute_anonymization_metrics(pred_entities, ground_truth, strict=False)
        
        # 4. Vérification de fuite (Leakage)
        anonymized_text = api_res.get("anonymized_text", "")
        leaks = check_leakage(anonymized_text, text, ground_truth)
        
        # Agrégation
        results.append({
            "doc_id": i,
            "text_preview": text[:50],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f2": metrics['f2'],
            "leaks_count": len(leaks),
            "leaks_details": str(leaks) if leaks else ""
        })
    
    # Création du rapport DataFrame
    if not results:
        print(
            f"⚠️ Aucun document n'a pu être évalué pour {dataset_name}. "
            f"Skips API: {skipped_due_to_api}/{len(data)}"
        )
        return

    df = pd.DataFrame(results)
    print(
        f"✅ Documents traités avec succès: {success_count}/{len(data)} | "
        f"Skippés (API down): {skipped_due_to_api}"
    )
    
    # Moyennes globales
    print(f"\n📊 RÉSULTATS GLOBAUX POUR {dataset_name.upper()}")
    print(f"Rappel Moyen (Recall): {df['recall'].mean():.2%} (Objectif > 95%)")
    print(f"Précision Moyenne:     {df['precision'].mean():.2%}")
    print(f"F2-Score Moyen:        {df['f2'].mean():.2%}")
    print(f"Documents avec fuites: {len(df[df['leaks_count'] > 0])} / {len(df)}")
    
    # Sauvegarde
    report_path = os.path.join(OUTPUT_DIR, f"report_{dataset_name}.csv")
    df.to_csv(report_path, index=False)
    print(f"Rapport détaillé sauvegardé : {report_path}")

if __name__ == "__main__":
    # Liste tes datasets ici
    datasets = [
        ("TAB", "datasets/TAB/dev.jsonl"),
        ("PersonalReddit", "datasets/PersonalReddit/Reddit_synthetic/test.jsonl"),
        ("DB-Bio", "datasets/DB-bio/test.jsonl")
    ]
    
    # Vérification que l'API tourne
    try:
        requests.get("http://localhost:8000/docs", timeout=2)
    except:
        print("❌ ERREUR: L'API ne semble pas tourner sur localhost:8000.")
        print("Lance-la avec : uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000")
        exit(1)

    for name, path in datasets:
        if os.path.exists(path):
            run_evaluation(name, path)
        else:
            print(f"⚠️ Fichier introuvable : {path}")