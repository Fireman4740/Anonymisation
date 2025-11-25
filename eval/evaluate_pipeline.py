import csv
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import requests
from requests import RequestException
from tqdm import tqdm

from metrics import compute_anonymization_metrics, check_leakage

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/anonymize"
OUTPUT_DIR = "evaluation/reports"
REQUEST_TIMEOUT = 300
MAX_API_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2
CONSECUTIVE_FAILURE_LIMIT = 15
MAX_DEBUG_DOCS = 15  # Nombre max de documents pour lesquels on affiche les FP/FN

os.makedirs(OUTPUT_DIR, exist_ok=True)
session = requests.Session()

TEXT_FIELDS = [
    "text",
    "content",
    "doc_text",
    "document",
    "body",
    "article",
    "raw_text",
    "source",
    "source_text",
    "response",
]
LABEL_FIELDS_BASE = [
    "spans",
    "entities",
    "entity_spans",
    "labels",
    "ground_truth",
    "gold_spans",
    "annotations",
    "mentions",
    "targets",
    "ner",
    "ner_spans",
    "sensitive_entities",
]
NESTED_LABEL_FIELDS = [
    "metadata",
    "meta",
    "extra",
    "gt",
    "groundtruth",
    "labelled_data",
    "details",
    "info",
]

DATASET_SPECIFIC_LABEL_KEYS = {
    "TAB": ["labels", "sensitive_entities", "gt_entities", "masked_entities"],
    "PERSONALREDDIT": ["personality", "entities", "ground_truth_entities"],
    "DB-BIO": ["entities", "ground_truth_entities"],
}

EntitySpan = Tuple[int, int, str]
LabelContainer = Union[List[Any], Dict[str, Any], str, None]


# ---------------------------------------------------------------------------
# API CALL
# ---------------------------------------------------------------------------
def call_api(text: str) -> Optional[Dict[str, Any]]:
    """Appelle l'API d'anonymisation avec retries et backoff."""
    payload = {"text": text, "level": "L1", "scope_id": "eval_run"}

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = session.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except RequestException as exc:
            print(f"Erreur API (tentative {attempt}/{MAX_API_RETRIES}): {exc}")
            if attempt < MAX_API_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return None


# ---------------------------------------------------------------------------
# DATA LOADING / NORMALISATION
# ---------------------------------------------------------------------------
def load_dataset(file_path: str, dataset_name: str) -> List[Dict[str, Any]]:
    records = _read_records(file_path)
    normalized: List[Dict[str, Any]] = []
    docs_without_labels = 0

    label_keys = (
        DATASET_SPECIFIC_LABEL_KEYS.get(dataset_name.upper(), []) + LABEL_FIELDS_BASE
    )

    for idx, item in enumerate(records):
        text = _extract_text(item)
        if not text:
            continue

        raw_labels = _find_label_container(item, label_keys)
        spans = _normalize_labels(raw_labels, text)

        if not spans:
            docs_without_labels += 1

        normalized.append({"text": text, "ground_truth": spans})

    print(f"ℹ️ {len(normalized)} documents chargés depuis {dataset_name}")
    if docs_without_labels:
        print(
            f"   ⚠️ {docs_without_labels} document(s) sans vérité terrain détectée "
            "(ils compteront comme 0)."
        )
    return normalized


def _read_records(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]

        data = json.load(handle)

    if isinstance(data, list):
        return data

    for key in ("documents", "data", "items", "entries", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return value

    return [data]


def _extract_text(item: Dict[str, Any]) -> Optional[str]:
    for key in TEXT_FIELDS:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val

    tokens = item.get("tokens") or item.get("words")
    if isinstance(tokens, list) and all(isinstance(tok, str) for tok in tokens):
        return " ".join(tokens)

    return None


def _find_label_container(
    item: Dict[str, Any],
    candidate_keys: Sequence[str],
) -> LabelContainer:
    for key in candidate_keys:
        if key in item and item[key]:
            return item[key]

    for nested_key in NESTED_LABEL_FIELDS:
        nested = item.get(nested_key)
        if isinstance(nested, dict):
            candidate = _find_label_container(nested, candidate_keys)
            if candidate:
                return candidate

    return None


def _normalize_labels(raw_labels: LabelContainer, text: str) -> List[EntitySpan]:
    if not raw_labels:
        return []

    spans: List[EntitySpan] = []

    if isinstance(raw_labels, dict):
        for label, value in raw_labels.items():
            spans.extend(_normalize_label_values(value, str(label), text))
    elif isinstance(raw_labels, list):
        for entry in raw_labels:
            spans.extend(_normalize_label_entry(entry, text))
    else:
        spans.extend(_normalize_label_entry(raw_labels, text))

    return _deduplicate_spans(spans)


def _normalize_label_values(value: Any, label: str, text: str) -> List[EntitySpan]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        spans: List[EntitySpan] = []
        for entry in value:
            spans.extend(_normalize_label_entry(entry, text, label))
        return spans

    return _normalize_label_entry(value, text, label)


def _normalize_label_entry(
    entry: Any,
    text: str,
    label_hint: Optional[str] = None,
) -> List[EntitySpan]:
    spans: List[EntitySpan] = []

    if entry is None:
        return spans

    if isinstance(entry, dict):
        start = entry.get("start") or entry.get("begin") or entry.get("offset")
        end = entry.get("end") or entry.get("stop") or entry.get("limit")
        label = (
            entry.get("label")
            or entry.get("type")
            or entry.get("entity")
            or entry.get("category")
            or label_hint
        )

        if start is not None and end is not None and label:
            spans.append((int(start), int(end), str(label).upper()))
            return spans

        surface = (
            entry.get("text")
            or entry.get("value")
            or entry.get("surface")
            or entry.get("mention")
        )
        if surface and label:
            spans.extend(_spans_from_surface(text, surface, label))
        return spans

    if isinstance(entry, (list, tuple)):
        if len(entry) >= 3 and all(_is_number(val) for val in entry[:2]):
            start, end = int(entry[0]), int(entry[1])
            label = str(entry[2] or label_hint or "ENT")
            spans.append((start, end, label.upper()))
        elif len(entry) == 2 and _is_number(entry[0]) and _is_number(entry[1]) and label_hint:
            spans.append((int(entry[0]), int(entry[1]), label_hint.upper()))
        elif len(entry) == 2 and isinstance(entry[0], str):
            surface, lbl = entry
            label = lbl or label_hint
            if label:
                spans.extend(_spans_from_surface(text, surface, label))
        return spans

    if isinstance(entry, str):
        # Fallback for string-only lists (like TAB masked_entities)
        label = label_hint or "SENSITIVE"
        spans.extend(_spans_from_surface(text, entry, label))
        return spans

    if _is_number(entry):
        label = label_hint or "SENSITIVE"
        spans.extend(_spans_from_surface(text, str(entry), label))
        return spans

    return spans


def _spans_from_surface(text: str, surface: str, label: str) -> List[EntitySpan]:
    clean = (surface or "").strip()
    if not clean:
        return []

    spans: List[EntitySpan] = []
    haystack = text.lower()
    needle = clean.lower()

    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(clean), label.upper()))
        start = idx + len(clean)
    return spans


def _deduplicate_spans(spans: Iterable[EntitySpan]) -> List[EntitySpan]:
    unique: List[EntitySpan] = []
    seen = set()
    for span in spans:
        key = (span[0], span[1], span[2])
        if key in seen:
            continue
        seen.add(key)
        unique.append(span)
    return unique


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def run_evaluation(dataset_name: str, file_path: str) -> None:
    print(f"\n🚀 Démarrage de l'évaluation pour : {dataset_name}")
    data = load_dataset(file_path, dataset_name)

    if not data:
        print("⚠️ Dataset vide ou illisible.")
        return

    results: List[Dict[str, Any]] = []
    success_count = 0
    skipped_due_to_api = 0
    failure_streak = 0
    empty_truth_docs = 0
    zero_pred_docs = 0
    debugged_docs = 0

    for doc_id, item in enumerate(tqdm(data)):
        text = item["text"]
        ground_truth = item["ground_truth"]

        if not ground_truth:
            empty_truth_docs += 1

        api_res = call_api(text)
        if not api_res:
            skipped_due_to_api += 1
            failure_streak += 1
            if failure_streak >= CONSECUTIVE_FAILURE_LIMIT:
                print(
                    f"❌ Arrêt de {dataset_name}: "
                    f"{CONSECUTIVE_FAILURE_LIMIT} échecs API consécutifs."
                )
                break
            continue

        failure_streak = 0
        success_count += 1

        pred_entities: List[EntitySpan] = [
            (ent["start"], ent["end"], ent["etype"])
            for ent in api_res.get("audit", {}).get("entities", [])
        ]

        if not pred_entities:
            zero_pred_docs += 1

        metrics = compute_anonymization_metrics(
            pred_entities,
            ground_truth,
            strict=False,
        )

        anonymized_text = api_res.get("anonymized_text", "")
        leaks = check_leakage(anonymized_text, text, ground_truth)

        results.append(
            {
                "doc_id": doc_id,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f2": metrics["f2"],
                "pred_count": len(pred_entities),
                "truth_count": len(ground_truth),
                "leaks_count": len(leaks),
                # Added for detailed analysis
                "full_text": text,
                "text_snippet": text[:200],
                "ground_truth": ground_truth,
                "predictions": pred_entities,
                "leaks": leaks,
            }
        )

        if debugged_docs < MAX_DEBUG_DOCS:
            need_debug = False
            if ground_truth and metrics["recall"] == 0.0:
                need_debug = True
            if pred_entities and metrics["precision"] == 0.0:
                need_debug = True
            if not ground_truth and pred_entities:
                need_debug = True

            if need_debug:
                _print_doc_debug(doc_id, text, pred_entities, ground_truth)
                debugged_docs += 1

    if not results:
        print("⚠️ Aucun document évalué (API indisponible ?).")
        return

    # Save CSV report using standard library
    report_path = os.path.join(OUTPUT_DIR, f"report_{dataset_name}.csv")
    if results:
        keys = results[0].keys()
        # Filter out complex keys for CSV
        csv_keys = [k for k in keys if k not in ["ground_truth", "predictions", "leaks"]]
        
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_keys)
            writer.writeheader()
            for row in results:
                csv_row = {k: row[k] for k in csv_keys}
                writer.writerow(csv_row)

    # Save detailed JSON for visualization tool
    json_path = os.path.join(OUTPUT_DIR, f"report_{dataset_name}_details.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Calculate metrics manually without pandas
    avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0
    avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
    avg_f2 = sum(r["f2"] for r in results) / len(results) if results else 0
    leaky_docs_count = sum(1 for r in results if r["leaks_count"] > 0)

    print(f"\n✅ Documents traités avec succès: {success_count}/{len(data)}")
    print(f"   Skippés (API down): {skipped_due_to_api}")
    print(f"   Docs sans ground truth: {empty_truth_docs}")
    print(f"   Docs sans prédiction API: {zero_pred_docs}")
    print(f"\n📊 RÉSULTATS GLOBAUX POUR {dataset_name.upper()}")
    print(f"   Rappel moyen:    {avg_recall:.2%}")
    print(f"   Précision moyenne: {avg_precision:.2%}")
    print(f"   F2 moyen:        {avg_f2:.2%}")
    print(f"   Documents avec fuites: {leaky_docs_count} / {len(results)}")
    print(f"   Rapport sauvegardé : {report_path}")


def _print_doc_debug(
    doc_id: int,
    text: str,
    preds: List[EntitySpan],
    truth: List[EntitySpan],
) -> None:
    truth_set = set(truth)
    pred_set = set(preds)

    false_pos = pred_set - truth_set
    false_neg = truth_set - pred_set

    def _materialize(spans: Iterable[EntitySpan]) -> List[str]:
        out = []
        for start, end, label in spans:
            snippet = text[max(0, start - 20) : min(len(text), end + 20)].replace("\n", " ")
            surface = text[start:end]
            out.append(f"{label} -> '{surface}' (contexte: …{snippet}…)")
        return out

    print(f"\n⚠️ Doc {doc_id}: precision={0 if not preds else '%.2f' % 0}, recall={0 if not truth else '%.2f' % 0}")
    if false_pos:
        print("   Faux positifs:")
        for fp in _materialize(list(false_pos)[:5]):
            print(f"     - {fp}")
    if false_neg:
        print("   Manqués (FN):")
        for fn in _materialize(list(false_neg)[:5]):
            print(f"     - {fn}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = [
        # ("TAB", "datasets/TAB/test.jsonl"),
        # ("PersonalReddit", "datasets/PersonalReddit/Reddit_synthetic/test.jsonl"),
        ("DB-Bio", "datasets/DB-bio/test.jsonl"),
    ]

    try:
        requests.get("http://localhost:8000/docs", timeout=2)
    except RequestException:
        print("❌ ERREUR: L'API ne semble pas tourner sur localhost:8000.")
        print("Lance-la avec : uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000")
        raise SystemExit(1)

    for name, path in datasets:
        if os.path.exists(path):
            run_evaluation(name, path)
        else:
            print(f"⚠️ Fichier introuvable : {path}")