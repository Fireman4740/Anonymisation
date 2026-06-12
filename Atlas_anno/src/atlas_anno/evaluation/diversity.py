from __future__ import annotations

import hashlib
import math
import os
import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from atlas_anno.config import load_config
from atlas_anno.console import log
from atlas_anno.schemas import DocumentRecord
from atlas_anno.storage import load_documents, save_report

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Métriques lexicales (pur Python)
# ---------------------------------------------------------------------------

def distinct_n(texts: Sequence[str], n: int) -> Dict[str, float]:
    """Ratio de n-grammes uniques, au niveau corpus et en moyenne par document."""
    corpus_ngrams: Counter = Counter()
    per_doc: List[float] = []
    for text in texts:
        tokens = _tokenize(text)
        grams = _ngrams(tokens, n)
        if grams:
            per_doc.append(len(set(grams)) / len(grams))
        corpus_ngrams.update(grams)
    total = sum(corpus_ngrams.values())
    return {
        "corpus": (len(corpus_ngrams) / total) if total else 0.0,
        "mean_per_doc": (sum(per_doc) / len(per_doc)) if per_doc else 0.0,
    }


def _bleu4(hypothesis: List[str], references: List[List[str]]) -> float:
    """BLEU-4 (précisions modifiées + brevity penalty), pur Python."""
    if not hypothesis or not references:
        return 0.0
    log_precision_sum = 0.0
    for n in range(1, 5):
        hyp_grams = Counter(_ngrams(hypothesis, n))
        if not hyp_grams:
            return 0.0
        max_ref: Counter = Counter()
        for reference in references:
            ref_grams = Counter(_ngrams(reference, n))
            for gram, count in ref_grams.items():
                if count > max_ref[gram]:
                    max_ref[gram] = count
        clipped = sum(min(count, max_ref[gram]) for gram, count in hyp_grams.items())
        total = sum(hyp_grams.values())
        # Lissage +1 pour éviter log(0) sur les ordres élevés.
        log_precision_sum += math.log((clipped + 1) / (total + 1))
    closest_ref_len = min(
        (abs(len(reference) - len(hypothesis)), len(reference)) for reference in references
    )[1]
    brevity = 1.0 if len(hypothesis) > closest_ref_len else math.exp(1 - closest_ref_len / max(len(hypothesis), 1))
    return brevity * math.exp(log_precision_sum / 4)


def self_bleu(texts: Sequence[str], *, sample: int = 200, seed: int = 17) -> float:
    """Self-BLEU moyen : chaque hypothèse échantillonnée vs le reste du corpus.

    Plus bas = plus divers. Sous-échantillonnage déterministe pour borner O(N²).
    """
    if len(texts) < 2:
        return 0.0
    tokenized = [_tokenize(text) for text in texts]
    indexes = list(range(len(tokenized)))
    if len(indexes) > sample:
        indexes = sorted(random.Random(seed).sample(indexes, sample))
    scores: List[float] = []
    for index in indexes:
        references = [tokens for other, tokens in enumerate(tokenized) if other != index]
        scores.append(_bleu4(tokenized[index], references))
    return sum(scores) / len(scores) if scores else 0.0


def length_stats(texts: Sequence[str]) -> Dict[str, float]:
    lengths = [len(_tokenize(text)) for text in texts]
    if not lengths:
        return {"mean_tokens": 0.0, "std_tokens": 0.0}
    mean = sum(lengths) / len(lengths)
    variance = sum((value - mean) ** 2 for value in lengths) / len(lengths)
    return {"mean_tokens": mean, "std_tokens": math.sqrt(variance)}


def length_controlled_distinct_2(texts: Sequence[str]) -> Dict[str, Any]:
    """distinct-2 par quartile de longueur (contrôle du confound de longueur)."""
    lengths = sorted(len(_tokenize(text)) for text in texts)
    if not lengths:
        return {"per_quartile": {}, "mean": 0.0}
    quartile_bounds = [
        lengths[max(0, (len(lengths) * q) // 4 - 1)] for q in range(1, 4)
    ]

    def _bin(token_count: int) -> str:
        for index, bound in enumerate(quartile_bounds):
            if token_count <= bound:
                return f"q{index + 1}"
        return "q4"

    bins: Dict[str, List[str]] = {}
    for text in texts:
        bins.setdefault(_bin(len(_tokenize(text))), []).append(text)
    per_quartile = {
        name: distinct_n(bin_texts, 2)["corpus"] for name, bin_texts in sorted(bins.items())
    }
    mean = sum(per_quartile.values()) / len(per_quartile) if per_quartile else 0.0
    return {"per_quartile": per_quartile, "mean": mean}


# ---------------------------------------------------------------------------
# Couverture des cellules factorielles
# ---------------------------------------------------------------------------

def _cell_key(document: DocumentRecord) -> Tuple[str, str, str, str]:
    return (
        document.domain,
        str(document.metadata.get("difficulty", "unknown")),
        str(document.metadata.get("register") or "unknown"),
        document.scenario.document_goal,
    )


def cell_coverage(documents: Sequence[DocumentRecord]) -> Dict[str, Any]:
    cells: Counter = Counter(_cell_key(document) for document in documents)
    domains = {key[0] for key in cells}
    difficulties = {key[1] for key in cells}
    registers = {key[2] for key in cells}
    goals = {key[3] for key in cells}
    theoretical = len(domains) * len(difficulties) * len(registers) * len(goals)
    total = sum(cells.values())
    entropy = 0.0
    if total and len(cells) > 1:
        probabilities = [count / total for count in cells.values()]
        entropy = -sum(p * math.log(p) for p in probabilities) / math.log(len(cells))
    return {
        "filled_cells": len(cells),
        "theoretical_cells": theoretical,
        "coverage": (len(cells) / theoretical) if theoretical else 0.0,
        "normalized_entropy": entropy,
        "top_cells": [
            {"cell": " × ".join(key), "count": count} for key, count in cells.most_common(5)
        ],
    }


# ---------------------------------------------------------------------------
# Dédup MinHash (pur Python, hash md5 salé — déterministe inter-process)
# ---------------------------------------------------------------------------

def _shingles(tokens: Sequence[str], size: int) -> set:
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return set(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _minhash_signature(shingles: Iterable[Tuple[str, ...]], num_hashes: int) -> List[int]:
    signature = [2 ** 64] * num_hashes
    for shingle in shingles:
        base = " ".join(shingle)
        for salt in range(num_hashes):
            digest = hashlib.md5(f"{salt}:{base}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:8], "big")
            if value < signature[salt]:
                signature[salt] = value
    return signature


def find_near_duplicates(
    texts: Sequence[str],
    *,
    num_hashes: int = 64,
    shingle_size: int = 3,
    jaccard_threshold: float = 0.80,
) -> List[Tuple[int, int, float]]:
    """Paires (i, j, jaccard_estimé) de quasi-doublons via signatures MinHash."""
    signatures = [
        _minhash_signature(_shingles(_tokenize(text), shingle_size), num_hashes) for text in texts
    ]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            matches = sum(1 for a, b in zip(signatures[i], signatures[j]) if a == b)
            estimate = matches / num_hashes
            if estimate >= jaccard_threshold:
                pairs.append((i, j, round(estimate, 4)))
    return pairs


# ---------------------------------------------------------------------------
# Métriques d'embeddings (optionnelles, import paresseux)
# ---------------------------------------------------------------------------

def embedding_metrics(texts: Sequence[str], *, sample: int = 300, seed: int = 17) -> Dict[str, Any]:
    """Dispersion cosinus + Vendi Score si sentence-transformers/numpy sont
    disponibles ; sinon skip gracieux (l'interpréteur de test ne les a pas)."""
    if os.environ.get("ATLAS_ENABLE_EMBEDDINGS", "").lower() not in {"1", "true", "yes"}:
        return {"skipped": True, "reason": "désactivé par défaut (définir ATLAS_ENABLE_EMBEDDINGS=1)"}
    try:
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        return {"skipped": True, "reason": f"dépendance absente: {exc.name}"}
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    subset = list(texts)
    if len(subset) > sample:
        subset = random.Random(seed).sample(subset, sample)
    try:
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True,
        )
        embeddings = model.encode(subset, normalize_embeddings=True)
    except Exception as exc:
        return {"skipped": True, "reason": f"modèle indisponible hors ligne: {exc}"}
    similarity = embeddings @ embeddings.T
    n = similarity.shape[0]
    off_diagonal = (similarity.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    eigenvalues = np.linalg.eigvalsh(similarity / n)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    eigenvalues = eigenvalues / eigenvalues.sum()
    vendi = float(np.exp(-(eigenvalues * np.log(eigenvalues)).sum()))
    return {
        "skipped": False,
        "mean_pairwise_cosine": float(off_diagonal),
        "dispersion": float(1.0 - off_diagonal),
        "vendi_score": vendi,
        "sampled_documents": n,
    }


# ---------------------------------------------------------------------------
# Rapport consolidé + porte
# ---------------------------------------------------------------------------

def evaluate_diversity(
    documents: Sequence[DocumentRecord],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    config = config or {}
    texts = [document.text for document in documents]
    minhash_config = dict(config.get("minhash", {}) or {})
    duplicates = find_near_duplicates(
        texts,
        num_hashes=int(minhash_config.get("num_hashes", 64)),
        shingle_size=int(minhash_config.get("shingle_size", 3)),
        jaccard_threshold=float(minhash_config.get("jaccard_threshold", 0.80)),
    )
    duplicated_docs = {index for pair in duplicates for index in pair[:2]}
    duplicate_rate = (len(duplicated_docs) / len(texts)) if texts else 0.0

    distinct_1 = distinct_n(texts, 1)
    distinct_2 = distinct_n(texts, 2)
    distinct_3 = distinct_n(texts, 3)
    self_bleu_score = self_bleu(
        texts,
        sample=int(config.get("self_bleu_sample", 200)),
        seed=int(config.get("self_bleu_seed", 17)),
    )
    coverage = cell_coverage(documents)

    thresholds = {
        "self_bleu_max": float(config.get("self_bleu_max", 0.90)),
        "distinct_2_min": float(config.get("distinct_2_min", 0.15)),
        "max_duplicate_rate": float(config.get("max_duplicate_rate", 0.02)),
        "min_cell_coverage": float(config.get("min_cell_coverage", 0.60)),
    }
    failures: List[str] = []
    if self_bleu_score > thresholds["self_bleu_max"]:
        failures.append(f"self_bleu {self_bleu_score:.3f} > {thresholds['self_bleu_max']}")
    if distinct_2["corpus"] < thresholds["distinct_2_min"]:
        failures.append(f"distinct_2 {distinct_2['corpus']:.3f} < {thresholds['distinct_2_min']}")
    if duplicate_rate > thresholds["max_duplicate_rate"]:
        failures.append(f"duplicate_rate {duplicate_rate:.3f} > {thresholds['max_duplicate_rate']}")
    if coverage["coverage"] < thresholds["min_cell_coverage"]:
        failures.append(f"cell_coverage {coverage['coverage']:.3f} < {thresholds['min_cell_coverage']}")

    return {
        "summary": {
            "documents": len(texts),
            "distinct_1": round(distinct_1["corpus"], 4),
            "distinct_2": round(distinct_2["corpus"], 4),
            "distinct_3": round(distinct_3["corpus"], 4),
            "self_bleu": round(self_bleu_score, 4),
            "duplicate_rate": round(duplicate_rate, 4),
            "cell_coverage": round(coverage["coverage"], 4),
            "cell_entropy": round(coverage["normalized_entropy"], 4),
            "passed": not failures,
            "failures": failures,
            "thresholds": thresholds,
        },
        "details": {
            "distinct": {"1": distinct_1, "2": distinct_2, "3": distinct_3},
            "length": length_stats(texts),
            "length_controlled_distinct_2": length_controlled_distinct_2(texts),
            "cells": coverage,
            "duplicate_pairs": [
                {
                    "doc_a": documents[i].doc_id,
                    "doc_b": documents[j].doc_id,
                    "jaccard_estimate": score,
                }
                for i, j, score in duplicates[:50]
            ],
            "embeddings": embedding_metrics(texts),
        },
    }


def run_eval_diversity_command() -> None:
    documents = load_documents(annotated=False)
    if not documents:
        documents = load_documents(annotated=True)
    if not documents:
        raise RuntimeError("Aucun document à évaluer : lancez generate-dataset d'abord.")
    config = load_config().defaults.get("diversity", {}) or {}
    report = evaluate_diversity(documents, config)
    save_report("raw", "diversity", report)
    summary = report["summary"]
    log(
        "Diversity: distinct2={distinct_2} selfBLEU={self_bleu} dup={duplicate_rate} "
        "coverage={cell_coverage} passed={passed}".format(**summary)
    )
    if summary["failures"]:
        log("Diversity failures: " + "; ".join(summary["failures"]))
