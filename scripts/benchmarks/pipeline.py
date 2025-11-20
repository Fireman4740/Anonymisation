"""End-to-end anonymisation pipeline benchmarks."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.core.orchestrator import anonymize_text, get_ner_mode
from src.llm.openrouter_client import OpenRouterClient
from src.rupta.privacy_evaluator import evaluate_reidentification_risk
from src.rupta.utility_evaluator import evaluate_utility_preservation
from src.services.ner import (
    GPU_OPTIMIZER_AVAILABLE,
    create_optimized_pipeline,
    load_gpu_config,
    run_gliner,
    warm_up_models,
)

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

_ENV_DATA_ROOT = os.getenv("ANONYMISATION_DATA_ROOT") or os.getenv("ANONYMIZATION_DATA_ROOT")
_DATASET_BASE_DIRS: List[Path] = []

if _ENV_DATA_ROOT:
    try:
        env_path = Path(_ENV_DATA_ROOT).expanduser().resolve()
        _DATASET_BASE_DIRS.append(env_path)
    except OSError:
        logger.warning("Unable to resolve ANONYMISATION_DATA_ROOT='%s'", _ENV_DATA_ROOT)

_DATASET_BASE_DIRS.extend(
    [
        WORKSPACE_ROOT / "Dataset" / "evaluation",
        WORKSPACE_ROOT / "Dataset",
        WORKSPACE_ROOT / "datasets" / "evaluation",
        WORKSPACE_ROOT / "datasets",
    ]
)

# Drop duplicates while keeping first occurrence
SEEN_BASES: Set[Path] = set()
DATASET_BASE_DIRS: List[Path] = []
for base in _DATASET_BASE_DIRS:
    try:
        resolved = base.resolve()
    except OSError:
        continue
    if resolved in SEEN_BASES:
        continue
    SEEN_BASES.add(resolved)
    DATASET_BASE_DIRS.append(resolved)


_ENV_RESULTS_ROOT = os.getenv("ANONYMISATION_RESULTS_DIR") or os.getenv("ANONYMIZATION_RESULTS_DIR")
_RESULTS_BASE_DIRS: List[Path] = []

if _ENV_RESULTS_ROOT:
    try:
        env_results = Path(_ENV_RESULTS_ROOT).expanduser().resolve()
        _RESULTS_BASE_DIRS.append(env_results)
    except OSError:
        logger.warning("Unable to resolve ANONYMISATION_RESULTS_DIR='%s'", _ENV_RESULTS_ROOT)

_RESULTS_BASE_DIRS.extend(
    [
        WORKSPACE_ROOT / "results",
        WORKSPACE_ROOT / "Results",
    ]
)

SEEN_RESULTS: Set[Path] = set()
RESULTS_BASE_DIRS: List[Path] = []
for base in _RESULTS_BASE_DIRS:
    try:
        resolved = base.resolve()
    except OSError:
        continue
    if resolved in SEEN_RESULTS:
        continue
    SEEN_RESULTS.add(resolved)
    RESULTS_BASE_DIRS.append(resolved)


_GPU_PIPELINE_CACHE: Dict[str, Any] = {"pipeline": None, "config": None}


@dataclass
class PipelineConfig:
    """Configuration holder for pipeline benchmarks.

    Attributes:
        dataset: Dataset name or 'all'.
        split: Split to evaluate.
        samples: Number of samples to evaluate (<= 0 to process the entire split).
    """

    dataset: str
    split: str = "test"
    samples: int = 10
    policy: str = "L1"
    baseline_only: bool = False
    rupta_only: bool = False
    output: Optional[Path] = None
    rate_limit: float = 0.0
    print_summary: bool = True


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list payload in {path}, got {type(data)}")
    return list(data)


def resolve_dataset_file(
    rel_variants: Sequence[Sequence[str]],
    split: str,
    suffixes: Sequence[str],
    dataset_label: str,
) -> Path:
    candidates: List[Path] = []
    for base in DATASET_BASE_DIRS:
        if not base.exists():
            continue
        for rel in rel_variants:
            folder = base.joinpath(*rel)
            if not folder.exists():
                continue
            for suffix in suffixes:
                candidate = folder / f"{split}{suffix}"
                if candidate.exists():
                    return candidate
                candidates.append(candidate)
    hint = candidates[-1] if candidates else None
    hint_msg = f" (last checked: {hint})" if hint else ""
    raise FileNotFoundError(f"Dataset {dataset_label} missing{hint_msg}")


def resolve_results_base() -> Path:
    for base in RESULTS_BASE_DIRS:
        if base.exists() and base.is_dir():
            return base
    # default to first candidate, create if necessary
    target = RESULTS_BASE_DIRS[0] if RESULTS_BASE_DIRS else WORKSPACE_ROOT / "results"
    target.mkdir(parents=True, exist_ok=True)
    return target


def default_output_path(config: PipelineConfig, timestamp_slug: str) -> Path:
    base = resolve_results_base()
    subdir = base / "pipeline"
    if config.dataset != "all":
        subdir = subdir / config.dataset
    subdir.mkdir(parents=True, exist_ok=True)
    name = f"{config.dataset}-{config.split}_{config.policy}_{timestamp_slug}.json"
    return subdir / name


def persist_suite(suite: Dict[str, Any], output_path: Path) -> None:
    """Write the current suite payload atomically to disk."""

    suite["output_path"] = str(output_path)
    suite["last_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(suite, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def precompute_ner_batches(entries: Sequence[Dict[str, Any]], dataset: str) -> List[List[Dict[str, Any]]]:
    """Precompute NER entities for a list of entries.

    Returns a list aligned with ``entries`` where each item is the list of
    detected entities. When acceleration backends are unavailable, falls back
    to a threaded GLiNER ensemble to avoid re-running NER inside each
    ``anonymize_text`` invocation.
    """

    if not entries:
        return []

    texts = [entry.get("text", "") for entry in entries]
    if not any(texts):
        return [[] for _ in entries]

    start_ts = time.time()
    pipeline = _GPU_PIPELINE_CACHE.get("pipeline")
    config = _GPU_PIPELINE_CACHE.get("config")

    if pipeline is None and GPU_OPTIMIZER_AVAILABLE:
        try:
            config = load_gpu_config()
            if config.get("enabled"):
                pipeline = create_optimized_pipeline(config)
                _GPU_PIPELINE_CACHE["pipeline"] = pipeline
                _GPU_PIPELINE_CACHE["config"] = config
        except Exception as exc:
            logger.warning("Failed to initialise GPU NER pipeline: %s", exc)
            pipeline = None

    results: List[List[Dict[str, Any]]] = []

    if pipeline is not None:
        logger.info(
            "Precomputing NER for %d %s samples using GPU pipeline (batch=%s, parallel_models=%s)",
            len(entries),
            dataset,
            config.get("batch_size") if isinstance(config, dict) else "?",
            config.get("max_parallel_models") if isinstance(config, dict) else "?",
        )
        for idx, text in enumerate(texts):
            if not text:
                ner = []
            else:
                try:
                    ner = pipeline.predict(text)
                except Exception as exc:
                    logger.warning(
                        "GPU NER failed for %s sample %d (%s); falling back to GLiNER",
                        dataset,
                        idx,
                        exc,
                    )
                    ner = run_gliner(text)
            results.append(ner)
            if (idx + 1) % 20 == 0:
                logger.debug("NER precompute %s: %d/%d", dataset, idx + 1, len(entries))
    else:
        max_workers = min(8, max(1, (os.cpu_count() or 4)))
        logger.info(
            "Precomputing NER for %d %s samples using GLiNER ensemble with %d workers",
            len(entries),
            dataset,
            max_workers,
        )

        def worker(text: str) -> List[Dict[str, Any]]:
            return run_gliner(text) if text else []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(worker, texts))

    elapsed = time.time() - start_ts
    logger.info("NER precompute finished for %s in %.2fs", dataset, elapsed)
    return results


def load_dataset(dataset: str, split: str) -> List[Dict[str, Any]]:
    if dataset == "dbbio":
        file = resolve_dataset_file(
            rel_variants=[("DB-Bio",), ("DB-bio",), ("dbbio",)],
            split=split,
            suffixes=(".jsonl", ".json"),
            dataset_label="DB-Bio",
        )
        return load_jsonl(file) if file.suffix == ".jsonl" else load_json_array(file)
    if dataset == "reddit":
        file = resolve_dataset_file(
            rel_variants=[
                ("PersonalReddit",),
                ("PersonalReddit", "Reddit_synthetic"),
            ],
            split=split,
            suffixes=(".jsonl", ".json"),
            dataset_label="PersonalReddit",
        )
        return load_jsonl(file) if file.suffix == ".jsonl" else load_json_array(file)
    if dataset == "tab":
        file = resolve_dataset_file(
            rel_variants=[("TAB",), ("tab",)],
            split=split,
            suffixes=(".jsonl", ".json"),
            dataset_label="TAB",
        )
        return load_jsonl(file) if file.suffix == ".jsonl" else load_json_array(file)
    raise ValueError(f"Unsupported dataset '{dataset}'")


def ensure_client() -> OpenRouterClient:
    client = OpenRouterClient.from_config()
    if client.requires_api_key and not client.api_key:
        missing = client.api_key_env or "OPENROUTER_API_KEY"
        raise RuntimeError(
            "No API key found for LLM provider. "
            f"Set the environment variable {missing}."
        )
    return client


def extract_ground_truth(entry: Dict[str, Any]) -> Tuple[str, str]:
    people = entry.get("people")
    if isinstance(people, list):
        ground_truth_people = people[0] if people else "Unknown"
    else:
        ground_truth_people = people or entry.get("person") or "Unknown"
    label = (
        entry.get("label")
        or entry.get("occupation")
        or entry.get("meta", {}).get("label")
        or "Unknown"
    )
    return str(ground_truth_people), str(label)


def summarise_metrics(records: Iterable[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    runtimes: List[float] = []
    for record in records:
        block = record.get(key)
        if not block:
            continue
        metrics.append(block)
        runtimes.append(float(block.get("runtime_seconds", 0.0)))
    if not metrics:
        return None

    def collect(path: List[str]) -> List[float]:
        values: List[float] = []
        for item in metrics:
            ref = item
            for part in path:
                ref = ref.get(part)
                if ref is None:
                    break
            if ref is None:
                continue
            try:
                values.append(float(ref))
            except (TypeError, ValueError):
                continue
        return values

    ranks = collect(["privacy", "rank"])
    confidences = collect(["utility", "confidence_score"])
    non_identified = [
        1.0 if block.get("privacy", {}).get("non_identified") else 0.0
        for block in metrics
    ]
    utility_preserved = [
        1.0 if block.get("utility", {}).get("utility_preserved") else 0.0
        for block in metrics
    ]

    return {
        "samples": len(metrics),
        "avg_privacy_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "privacy_non_identified_rate": (sum(non_identified) / len(non_identified)) if non_identified else None,
        "avg_utility_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "utility_preserved_rate": (sum(utility_preserved) / len(utility_preserved)) if utility_preserved else None,
        "avg_runtime_seconds": (sum(runtimes) / len(runtimes)) if runtimes else None,
    }


def evaluate_single(
    dataset: str,
    index: int,
    entry: Dict[str, Any],
    client: OpenRouterClient,
    policy: str,
    run_baseline: bool,
    run_rupta: bool,
    precomputed_ner: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    original_text = entry.get("text", "")
    ground_truth_people, ground_truth_label = extract_ground_truth(entry)

    record: Dict[str, Any] = {
        "id": entry.get("doc_id", f"{dataset}_{index}"),
        "ground_truth_people": ground_truth_people,
        "ground_truth_label": ground_truth_label,
    }

    ner_cache: List[Dict[str, Any]] = [dict(ent) for ent in precomputed_ner or []]

    if run_baseline:
        start = time.time()
        baseline_overrides = {
            "rupta_enabled": False,
            "llm_detection": policy != "L0",
        }
        if ner_cache:
            baseline_overrides["disable_internal_ner"] = True
        baseline_result = anonymize_text(
            value=original_text,
            scope_id=f"{dataset}_baseline_{index}",
            secret_salt="benchmark_secret",
            level=policy,
            overrides=baseline_overrides,
            ner_results=ner_cache or None,
        )
        runtime = time.time() - start
        anonymized = baseline_result.get("anonymized_text", original_text)
        if not ner_cache:
            ner_cache = [
                dict(ent)
                for ent in (
                    (baseline_result.get("ner") or {}).get("entities")
                    or baseline_result.get("ner_entities")
                    or []
                )
            ]
        privacy = evaluate_reidentification_risk(
            client=client,
            anonymized_text=anonymized,
            ground_truth_people=ground_truth_people,
            p_threshold=10,
        )
        utility = evaluate_utility_preservation(
            client=client,
            anonymized_text=anonymized,
            ground_truth_label=ground_truth_label,
        )
        record["baseline"] = {
            "anonymized_text": anonymized,
            "privacy": privacy,
            "utility": utility,
            "runtime_seconds": runtime,
        }

    if run_rupta:
        start = time.time()
        rupta_overrides = {
            "rupta_enabled": True,
            "rupta_ground_truth_people": ground_truth_people,
            "rupta_ground_truth_label": ground_truth_label,
        }
        if ner_cache:
            rupta_overrides.setdefault("disable_internal_ner", True)
        rupta_result = anonymize_text(
            value=original_text,
            scope_id=f"{dataset}_rupta_{index}",
            secret_salt="benchmark_secret",
            level=policy,
            overrides=rupta_overrides,
            ner_results=ner_cache or None,
        )
        runtime = time.time() - start
        anonymized = rupta_result.get("anonymized_text", original_text)
        metrics = rupta_result.get("rupta_metrics") or {}
        if metrics:
            privacy = metrics.get("privacy", {})
            utility = metrics.get("utility", {})
        else:
            privacy = evaluate_reidentification_risk(
                client=client,
                anonymized_text=anonymized,
                ground_truth_people=ground_truth_people,
                p_threshold=10,
            )
            utility = evaluate_utility_preservation(
                client=client,
                anonymized_text=anonymized,
                ground_truth_label=ground_truth_label,
            )
        record["rupta"] = {
            "anonymized_text": anonymized,
            "privacy": privacy,
            "utility": utility,
            "runtime_seconds": runtime,
        }

    return record


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    datasets = [config.dataset] if config.dataset != "all" else ["dbbio", "reddit", "tab"]
    run_baseline = not config.rupta_only
    run_rupta = not config.baseline_only
    if not run_baseline and not run_rupta:
        raise ValueError("Nothing to run: disable either baseline or rupta execution")

    client = ensure_client()

    try:
        warm_up_models()
    except Exception as exc:
        logger.debug("GLiNER warm-up skipped: %s", exc)

    try:
        ner_mode = get_ner_mode()
    except Exception:
        ner_mode = "unknown"
    else:
        logger.info("NER warm-up ready (mode: %s)", ner_mode)

    timestamp_slug = time.strftime("%Y%m%d-%H%M%S")
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    requested_samples = config.samples if config.samples > 0 else "all"
    suite: Dict[str, Any] = {
        "policy": config.policy,
        "split": config.split,
        "datasets": {},
        "generated_at": generated_at,
        "ner_mode": ner_mode,
        "samples_requested": requested_samples,
        "samples_requested_raw": config.samples,
    }

    for dataset in datasets:
        entries = load_dataset(dataset, config.split)
        total_entries = len(entries)
        if config.samples <= 0:
            selected = entries
        else:
            selected = entries[: min(total_entries, config.samples)]
        if not selected:
            logger.warning("No samples selected for dataset %s", dataset)
            suite["datasets"][dataset] = {
                "samples": [],
                "summary": {"baseline": None, "rupta": None},
                "requested_samples": requested_samples,
                "requested_samples_raw": config.samples,
                "processed_samples": 0,
                "total_entries": total_entries,
            }
            continue
        records: List[Dict[str, Any]] = []
        try:
            precomputed = precompute_ner_batches(selected, dataset)
        except Exception as exc:
            logger.warning("NER precompute failed for %s: %s", dataset, exc)
            precomputed = [[] for _ in selected]

        dataset_start = time.time()
        progress_begin = dataset_start

        def render_progress(current: int, total: int) -> None:
            width = 28
            frac = current / total if total else 1.0
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            elapsed = time.time() - progress_begin
            avg = elapsed / current if current else 0.0
            eta = avg * (total - current) if current else 0.0
            print(
                f"\r[{dataset.upper()}] [{bar}] {current}/{total} | elapsed {elapsed:.1f}s | avg {avg:.2f}s | eta {eta:.1f}s",
                end="",
                flush=True,
            )
            if current >= total:
                print()

        for idx, entry in enumerate(selected):
            ner_snapshot = precomputed[idx] if idx < len(precomputed) else None
            record = evaluate_single(
                dataset,
                idx,
                entry,
                client,
                config.policy,
                run_baseline,
                run_rupta,
                precomputed_ner=ner_snapshot,
            )
            records.append(record)
            if config.rate_limit > 0:
                time.sleep(config.rate_limit)
            render_progress(idx + 1, len(selected))

        dataset_elapsed = time.time() - dataset_start
        avg_sample_time = dataset_elapsed / len(selected) if selected else 0.0
        throughput = (len(selected) / dataset_elapsed) if dataset_elapsed else 0.0
        summary = {
            "baseline": summarise_metrics(records, "baseline"),
            "rupta": summarise_metrics(records, "rupta"),
        }
        error_count = 0
        for record in records:
            for variant in ("baseline", "rupta"):
                block = record.get(variant) or {}
                privacy_block = block.get("privacy") or {}
                if isinstance(privacy_block, dict) and privacy_block.get("error"):
                    error_count += 1
        suite["datasets"][dataset] = {
            "samples": records,
            "summary": summary,
            "requested_samples": requested_samples,
            "requested_samples_raw": config.samples,
            "processed_samples": len(selected),
            "total_entries": total_entries,
            "timing": {
                "elapsed_seconds": dataset_elapsed,
                "avg_seconds_per_sample": avg_sample_time,
                "samples_per_minute": throughput * 60.0,
            },
            "errors": error_count,
        }

        if config.print_summary:
            print(f"\n=== {dataset.upper()} ({len(records)} / {total_entries} samples) ===")
            base = summary.get("baseline") if summary else None
            if base:
                print("Baseline:")
                print(f"  avg privacy rank: {base['avg_privacy_rank']}")
                print(f"  non identified rate: {base['privacy_non_identified_rate']}")
                print(f"  avg utility confidence: {base['avg_utility_confidence']}")
                print(f"  utility preserved rate: {base['utility_preserved_rate']}")
                print(f"  avg runtime (s): {base['avg_runtime_seconds']}")
            rup = summary.get("rupta") if summary else None
            if rup:
                print("RUPTA:")
                print(f"  avg privacy rank: {rup['avg_privacy_rank']}")
                print(f"  non identified rate: {rup['privacy_non_identified_rate']}")
                print(f"  avg utility confidence: {rup['avg_utility_confidence']}")
                print(f"  utility preserved rate: {rup['utility_preserved_rate']}")
                print(f"  avg runtime (s): {rup['avg_runtime_seconds']}")
            timing = suite["datasets"][dataset]["timing"]
            print("Timing:")
            print(f"  total elapsed (s): {timing['elapsed_seconds']:.2f}")
            print(f"  avg per sample (s): {timing['avg_seconds_per_sample']:.2f}")
            print(f"  throughput (samples/min): {timing['samples_per_minute']:.2f}")
            if error_count:
                print(f"  privacy evaluator errors: {error_count}")

    output_path = config.output
    if output_path is None:
        output_path = default_output_path(config, timestamp_slug)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    persist_suite(suite, output_path)
    print(f"\nResults written to {output_path}")

    return suite


def handle_cli(args) -> Dict[str, Any]:
    config = PipelineConfig(
        dataset=args.dataset,
        split=args.split,
        samples=args.samples,
        policy=args.policy,
        baseline_only=args.baseline_only,
        rupta_only=args.rupta_only,
        output=args.output,
        rate_limit=args.rate_limit,
    )
    return run_pipeline(config)
