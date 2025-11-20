"""NER benchmark helpers."""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from src.services.ner import (
    GPU_OPTIMIZER_AVAILABLE,
    create_optimized_pipeline,
    load_gpu_config,
    run_gliner,
    warm_up_models,
)

logger = logging.getLogger(__name__)

TEST_TEXTS: Dict[str, str] = {
    "short": (
        "Jean Dupont travaille chez Acme Corporation a Paris. "
        "Son email est jean.dupont@acme.fr et son telephone +33 1 42 68 53 00."
    ),
    "medium": (
        "Jean Dupont, ne le 15/03/1985 a Lyon, est ingenieur logiciel chez Acme Corporation. "
        "Adresse: 123 Avenue des Champs-Elysees, 75008 Paris. "
        "Il collabore avec Marie Martin et Pierre Dubois."
    ),
    "long": (
        "Rapport d'incident confidentiel\n\nDate : 24/10/2025\n"
        "Auteur : Jean Dupont (jean.dupont@acme.fr, +33 1 42 68 53 00)\n"
        "Societe : Acme Corporation SAS, 123 Avenue des Champs-Elysees, 75008 Paris\n"
        "Details : tentative d'acces non autorise depuis l'adresse IP 192.168.45.123."
    ),
}


@dataclass
class BenchmarkResult:
    """Store durations and metrics for a single benchmark mode."""

    mode: str
    durations: List[float]
    entity_count: int

    @property
    def runs(self) -> int:
        return len(self.durations)

    def summary(self) -> Dict[str, Any]:
        if not self.durations:
            return {
                "mode": self.mode,
                "runs": 0,
                "avg_seconds": 0.0,
                "min_seconds": 0.0,
                "max_seconds": 0.0,
                "entity_count": self.entity_count,
            }
        avg = statistics.mean(self.durations)
        return {
            "mode": self.mode,
            "runs": self.runs,
            "avg_seconds": avg,
            "min_seconds": min(self.durations),
            "max_seconds": max(self.durations),
            "entity_count": self.entity_count,
        }


def _benchmark_standard(text: str, runs: int, preset: str, threshold: float) -> BenchmarkResult:
    logger.debug("Starting standard NER benchmark")
    warm_up_models(gliner_preset=preset)
    durations: List[float] = []
    entities: List[Dict[str, Any]] = []
    for _ in range(runs):
        start = time.time()
        entities = run_gliner(text, preset=preset, threshold=threshold)
        durations.append(time.time() - start)
    return BenchmarkResult(mode="standard", durations=durations, entity_count=len(entities))


def _benchmark_gpu(text: str, runs: int, preset: str, threshold: float) -> BenchmarkResult:
    if not GPU_OPTIMIZER_AVAILABLE:
        raise RuntimeError("GPU optimiser unavailable. Install optional GPU dependencies.")
    config = load_gpu_config()
    config["enabled"] = True
    config["gliner_preset"] = preset
    pipeline = create_optimized_pipeline(config)
    if pipeline is None:
        raise RuntimeError("Unable to create GPU pipeline. Check CUDA availability and config.")

    durations: List[float] = []
    entities: List[Dict[str, Any]] = []
    for _ in range(runs):
        start = time.time()
        entities = pipeline.predict(text)
        durations.append(time.time() - start)
    if threshold is not None and threshold > 0:
        entities = [ent for ent in entities if float(ent.get("votes", 1.0)) >= threshold]
    return BenchmarkResult(mode="gpu", durations=durations, entity_count=len(entities))


def run_benchmarks(mode: str, text_size: str, runs: int, preset: str, threshold: float) -> List[BenchmarkResult]:
    text = TEST_TEXTS[text_size]
    results: List[BenchmarkResult] = []

    if mode in {"standard", "both"}:
        results.append(_benchmark_standard(text, runs, preset, threshold))
    if mode in {"gpu", "both"}:
        results.append(_benchmark_gpu(text, runs, preset, threshold))

    for result in results:
        summary = result.summary()
        print(f"\n[{summary['mode'].upper()}] {summary['runs']} runs")
        print(f"  avg : {summary['avg_seconds']:.3f}s")
        print(f"  min : {summary['min_seconds']:.3f}s")
        print(f"  max : {summary['max_seconds']:.3f}s")
        print(f"  entities detected : {summary['entity_count']}")

    return results


def handle_cli(args) -> None:
    """Entrypoint used by the command line parser."""

    run_benchmarks(
        mode=args.mode,
        text_size=args.text_size,
        runs=args.runs,
        preset=args.preset,
        threshold=args.threshold,
    )
