from __future__ import annotations

from typing import Dict

from atlas_anno.evaluation.privacy import evaluate_privacy
from atlas_anno.evaluation.reid import evaluate_reidentification
from atlas_anno.evaluation.spans import evaluate_span_metrics
from atlas_anno.evaluation.utility import evaluate_utility
from atlas_anno.storage import (
    load_anonymization_results,
    load_attack_results,
    load_documents,
    save_report,
)


def _load_attack_map(strategy: str):
    attacks = load_attack_results(strategy, "llm")
    if not attacks:
        attacks = load_attack_results(strategy, "structured")
    return {attack.doc_id: attack for attack in attacks}


def _load_attack_rows(strategy: str):
    attacks = load_attack_results(strategy, "llm")
    if not attacks:
        attacks = load_attack_results(strategy, "structured")
    return attacks


def run_eval_spans_command(strategy: str) -> None:
    documents = load_documents(annotated=True)
    report = evaluate_span_metrics(documents)
    save_report(strategy, "spans", report)


def run_eval_privacy_command(strategy: str) -> None:
    documents = load_documents(annotated=True)
    anonymized = {item.doc_id: item for item in load_anonymization_results(strategy)}
    attacks = _load_attack_map(strategy)
    attack_success = {doc_id: attack.best_person_id == next(doc.author_id for doc in documents if doc.doc_id == doc_id) for doc_id, attack in attacks.items()}
    report = evaluate_privacy(documents, anonymized, attack_success=attack_success)
    save_report(strategy, "privacy", report)


def run_eval_reid_command(strategy: str) -> None:
    documents = load_documents(annotated=True)
    attacks = _load_attack_rows(strategy)
    report = evaluate_reidentification(documents, attacks)
    save_report(strategy, "reid", report)


def run_eval_utility_command(strategy: str) -> None:
    documents = load_documents(annotated=True)
    anonymized = {item.doc_id: item for item in load_anonymization_results(strategy)}
    report = evaluate_utility(documents, anonymized)
    save_report(strategy, "utility", report)

