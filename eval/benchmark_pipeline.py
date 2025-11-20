#!/usr/bin/env python3
"""Benchmark le pipeline d'anonymisation via l'API FastAPI."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_API_URL = os.getenv("PIPELINE_API_URL", "http://localhost:8000")
DEFAULT_API_TIMEOUT = float(os.getenv("PIPELINE_API_TIMEOUT", "120"))
DEFAULT_SECRET_SALT = os.getenv("PIPELINE_SECRET_SALT", "change_me")
DEFAULT_SCOPE_PREFIX = os.getenv("PIPELINE_SCOPE_PREFIX", "benchmark")
DEFAULT_LEVEL = os.getenv("PIPELINE_DEFAULT_LEVEL", "L0")

TEXT_KEYS = ("text", "original_text", "prompt", "content", "input", "message", "body")
LEVEL_PATTERN = re.compile(r"(L[0-9])", re.IGNORECASE)

thread_local = threading.local()


@dataclass
class ExampleTask:
    dataset: str
    example_id: str
    text: str
    level: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    secret_salt: Optional[str] = None
    ner_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class ExampleResult:
    dataset: str
    example_id: str
    latency_seconds: Optional[float]
    success: bool
    status_code: Optional[int]
    error: Optional[str]
    response_bytes: Optional[int]
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmarke l'API /anonymize sur un ou plusieurs jeux de données.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        nargs="+",
        required=True,
        help="Chemin(s) vers un fichier JSON/JSONL ou un dossier contenant des datasets.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Nombre total d'exemples à mesurer (hors warmup).")
    parser.add_argument(
        "--per-dataset-limit",
        type=int,
        default=None,
        help="Limite d'exemples chargés par fichier de dataset avant mélange.",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Nombre de requêtes de chauffe ignorées dans les métriques.")
    parser.add_argument("--concurrency", type=int, default=1, help="Nombre de requêtes parallèles (>=1).")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL de base de l'API (défaut: %(default)s).")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_API_TIMEOUT,
        help="Timeout HTTP en secondes (défaut: %(default)s).",
    )
    parser.add_argument(
        "--secret-salt",
        default=DEFAULT_SECRET_SALT,
        help="Secret HMAC utilisé si l'exemple n'en fournit pas (défaut: %(default)s).",
    )
    parser.add_argument(
        "--scope-prefix",
        default=DEFAULT_SCOPE_PREFIX,
        help="Préfixe appliqué aux scope_id générés (défaut: %(default)s).",
    )
    parser.add_argument(
        "--default-level",
        default=DEFAULT_LEVEL,
        help="Niveau L0/L1/L2 utilisé si l'exemple n'en précise pas (défaut: %(default)s).",
    )
    parser.add_argument(
        "--global-overrides",
        default=None,
        help="JSON ou chemin vers un fichier JSON d'overrides appliqué à tous les exemples.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Mélange les exemples avant exécution.")
    parser.add_argument("--seed", type=int, default=None, help="Seed utilisé avec --shuffle pour rendre l'ordre déterministe.")
    parser.add_argument("--fail-fast", action="store_true", help="Arrête dès la première erreur API.")
    parser.add_argument("--output", type=Path, default=None, help="Chemin du rapport JSON à écrire.")
    parser.add_argument("--verbose", action="store_true", help="Affiche un log détaillé par exemple.")
    return parser.parse_args()


def parse_global_overrides(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        candidate = Path(raw)
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"Impossible d'interpréter --global-overrides ({raw}).")


def iter_dataset_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"Format non supporté: {path}")
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Chemin introuvable: {path}")
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}:
            yield candidate


def load_examples(paths: List[str], *, default_level: str, per_dataset_limit: Optional[int]) -> List[ExampleTask]:
    tasks: List[ExampleTask] = []
    for raw_path in paths:
        base_path = Path(raw_path).expanduser().resolve()
        for file_path in iter_dataset_files(base_path):
            dataset_label = "/".join(file_path.parts[-2:]) if file_path.parent != base_path else file_path.stem
            dataset_tasks = load_examples_from_file(
                file_path,
                dataset_label=dataset_label,
                default_level=default_level,
                limit=per_dataset_limit,
            )
            tasks.extend(dataset_tasks)
    return tasks


def load_examples_from_file(
    file_path: Path,
    *,
    dataset_label: str,
    default_level: str,
    limit: Optional[int] = None,
) -> List[ExampleTask]:
    if file_path.suffix.lower() == ".jsonl":
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "examples" in payload:
                records = payload["examples"]
            elif "cases" in payload:
                records = payload["cases"]
            else:
                records = list(payload.values()) if isinstance(payload, dict) else []
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Structure JSON non supportée dans {file_path}")

    tasks: List[ExampleTask] = []
    for idx, record in enumerate(records):
        if limit is not None and len(tasks) >= limit:
            break
        try:
            text = extract_text(record)
        except ValueError:
            continue
        example_id = extract_example_id(record, idx)
        level = extract_level(record, default_level)
        overrides = dict(record.get("overrides") or record.get("policy_overrides") or {})
        secret = record.get("secret_salt")
        ner_results = record.get("ner_results")
        tasks.append(
            ExampleTask(
                dataset=str(dataset_label),
                example_id=example_id,
                text=text,
                level=level,
                overrides=overrides,
                secret_salt=secret,
                ner_results=ner_results if isinstance(ner_results, list) else None,
            )
        )
    return tasks


def extract_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise ValueError("Impossible d'extraire le texte")
    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Aucun champ texte valide trouvé")


def extract_example_id(record: Any, fallback_idx: int) -> str:
    if isinstance(record, dict):
        for key in ("id", "example_id", "uuid", "ticket_id", "name"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"sample_{fallback_idx:05d}"


def extract_level(record: Any, default_level: str) -> str:
    candidate: Optional[str] = None
    if isinstance(record, dict):
        candidate = record.get("level") or record.get("niveau_anonymisation") or record.get("anonymization_level")
    if isinstance(candidate, str):
        match = LEVEL_PATTERN.search(candidate)
        if match:
            return match.group(1).upper()
        return candidate.strip().upper()
    return default_level


def slugify(value: str, fallback: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    return sanitized or fallback


def build_scope_id(prefix: str, dataset: str, example_id: str, index: int) -> str:
    parts = [part for part in [prefix, slugify(dataset, "dataset"), slugify(example_id, "sample"), str(index)] if part]
    scope = "-".join(parts)
    return scope[:120]


def get_session() -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        session = requests.Session()
        thread_local.session = session
    return session


def call_api(
    *,
    example: ExampleTask,
    index: int,
    args: argparse.Namespace,
    global_overrides: Dict[str, Any],
) -> ExampleResult:
    session = get_session()
    endpoint = f"{args.api_url.rstrip('/')}/anonymize"
    scope_id = build_scope_id(args.scope_prefix, example.dataset, example.example_id, index)
    overrides = dict(global_overrides)
    overrides.update(example.overrides or {})
    payload = {
        "text": example.text,
        "scope_id": scope_id,
        "level": example.level or args.default_level,
        "secret_salt": example.secret_salt or args.secret_salt,
        "overrides": overrides,
        "ner_results": example.ner_results,
    }
    start = time.perf_counter()
    try:
        response = session.post(endpoint, json=payload, timeout=args.timeout)
        latency = time.perf_counter() - start
        status_code = response.status_code
        response.raise_for_status()
        data = response.json()
        metrics = data.get("evaluation", {}).get("metrics", {}) if isinstance(data, dict) else {}
        return ExampleResult(
            dataset=example.dataset,
            example_id=example.example_id,
            latency_seconds=latency,
            success=True,
            status_code=status_code,
            error=None,
            response_bytes=len(response.content),
            evaluation_metrics=metrics if isinstance(metrics, dict) else {},
        )
    except requests.RequestException as exc:
        return ExampleResult(
            dataset=example.dataset,
            example_id=example.example_id,
            latency_seconds=None,
            success=False,
            status_code=getattr(exc.response, "status_code", None),
            error=str(exc),
            response_bytes=getattr(exc.response, "content", None) and len(exc.response.content),
        )
    except ValueError as exc:  # JSON invalide
        latency = time.perf_counter() - start
        return ExampleResult(
            dataset=example.dataset,
            example_id=example.example_id,
            latency_seconds=latency,
            success=False,
            status_code=None,
            error=f"Réponse non JSON: {exc}",
            response_bytes=None,
        )


def run_warmup(examples: List[ExampleTask], args: argparse.Namespace, global_overrides: Dict[str, Any]) -> None:
    if not examples:
        return
    print(f"[INFO] Warmup: {len(examples)} requêtes (séquentiel)...")
    for idx, example in enumerate(examples):
        result = call_api(example=example, index=idx, args=args, global_overrides=global_overrides)
        status = "OK" if result.success else f"KO ({result.error})"
        print(f"  - {example.dataset}::{example.example_id}: {status}")


def run_benchmark(
    examples: List[ExampleTask],
    args: argparse.Namespace,
    global_overrides: Dict[str, Any],
) -> tuple[List[ExampleResult], float]:
    if not examples:
        return [], 0.0

    measurement_start = time.perf_counter()
    results: List[ExampleResult] = []
    total = len(examples)
    print(f"[INFO] Lancement du benchmark sur {total} exemples (concurrency={args.concurrency})...")

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(call_api, example=example, index=idx, args=args, global_overrides=global_overrides): example
            for idx, example in enumerate(examples)
        }
        for completed_idx, future in enumerate(iter_futures(futures), 1):
            example = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - protection ultime
                result = ExampleResult(
                    dataset=example.dataset,
                    example_id=example.example_id,
                    latency_seconds=None,
                    success=False,
                    status_code=None,
                    error=str(exc),
                    response_bytes=None,
                )
            results.append(result)
            if args.verbose:
                status = "OK" if result.success else f"KO ({result.error})"
                latency_ms = f"{result.latency_seconds * 1000:.1f} ms" if result.latency_seconds else "-"
                print(f"[RUN] {completed_idx}/{total} {example.dataset}::{example.example_id} → {status} | {latency_ms}")
            elif completed_idx % max(1, total // 10 or 1) == 0:
                print(f"[PROGRESS] {completed_idx}/{total} exemples traités...")
            if not result.success and args.fail_fast:
                print("[ERROR] Mode fail-fast: arrêt suite à une erreur.")
                break

    measurement_end = time.perf_counter()
    return results, measurement_end - measurement_start


def iter_futures(futures):
    for future in as_completed(futures):
        yield future


def compute_stats(results: List[ExampleResult]) -> Dict[str, Any]:
    datasets: Dict[str, List[ExampleResult]] = defaultdict(list)
    for result in results:
        datasets[result.dataset].append(result)

    summary = {}
    for dataset, dataset_results in datasets.items():
        summary[dataset] = summarize_results(dataset_results)
    return summary


def summarize_results(dataset_results: List[ExampleResult]) -> Dict[str, Any]:
    latencies = sorted([r.latency_seconds for r in dataset_results if r.success and r.latency_seconds is not None])
    success = sum(1 for r in dataset_results if r.success)
    failure = len(dataset_results) - success
    return {
        "count": len(dataset_results),
        "success": success,
        "failure": failure,
        "latency_ms": build_latency_stats(latencies),
    }


def build_latency_stats(latencies: List[float]) -> Optional[Dict[str, float]]:
    if not latencies:
        return None
    return {
        "avg": 1000 * (sum(latencies) / len(latencies)),
        "p50": 1000 * percentile(latencies, 50),
        "p90": 1000 * percentile(latencies, 90),
        "p95": 1000 * percentile(latencies, 95),
        "max": 1000 * max(latencies),
        "min": 1000 * min(latencies),
    }


def percentile(data: List[float], percent: float) -> float:
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * (percent / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1


def print_summary(
    *,
    dataset_stats: Dict[str, Any],
    total_results: List[ExampleResult],
    wall_time: float,
) -> None:
    header = f"{'Dataset':35} | {'Total':>5} | {'OK':>5} | {'KO':>5} | {'avg (ms)':>10} | {'p95 (ms)':>10} | {'max (ms)':>10}"
    print("\n" + header)
    print("-" * len(header))

    for dataset, stats in sorted(dataset_stats.items()):
        lat = stats.get("latency_ms") or {}
        print(
            f"{dataset[:35]:35} | {stats['count']:5d} | {stats['success']:5d} | {stats['failure']:5d} | "
            f"{lat.get('avg', 0):10.1f} | {lat.get('p95', 0):10.1f} | {lat.get('max', 0):10.1f}"
        )

    total_success = sum(1 for r in total_results if r.success)
    total_failure = len(total_results) - total_success
    latencies = sorted([r.latency_seconds for r in total_results if r.success and r.latency_seconds is not None])
    overall_latency = build_latency_stats(latencies) or {}
    throughput = total_success / wall_time if wall_time > 0 else 0
    print("-" * len(header))
    print(
        f"{'GLOBAL':35} | {len(total_results):5d} | {total_success:5d} | {total_failure:5d} | "
        f"{overall_latency.get('avg', 0):10.1f} | {overall_latency.get('p95', 0):10.1f} | {overall_latency.get('max', 0):10.1f}"
    )
    print(f"[INFO] Temps total: {wall_time:.2f} s | Débit moyen: {throughput:.2f} req/s")


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    dataset_stats: Dict[str, Any],
    results: List[ExampleResult],
    wall_time: float,
) -> None:
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "api_url": args.api_url,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "warmup": args.warmup,
        "limit": args.limit,
        "per_dataset_limit": args.per_dataset_limit,
        "wall_time_seconds": wall_time,
        "dataset_stats": dataset_stats,
        "results": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Rapport écrit dans {path}")


def main() -> int:
    args = parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency doit être >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup doit être >= 0")

    global_overrides = parse_global_overrides(args.global_overrides)

    examples = load_examples(
        args.datasets,
        default_level=args.default_level,
        per_dataset_limit=args.per_dataset_limit,
    )
    if not examples:
        print("[WARN] Aucun exemple trouvé dans les datasets fournis.")
        return 1

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(examples)

    total_loaded = len(examples)
    warmup_examples = examples[: args.warmup]
    measured_examples = examples[args.warmup :]
    if args.limit is not None:
        measured_examples = measured_examples[: args.limit]

    print(
        f"[INFO] Exemples chargés: {total_loaded} | Warmup: {len(warmup_examples)} | Mesure: {len(measured_examples)}"
    )

    if warmup_examples:
        run_warmup(warmup_examples, args, global_overrides)

    results, wall_time = run_benchmark(measured_examples, args, global_overrides)

    dataset_stats = compute_stats(results)
    print_summary(dataset_stats=dataset_stats, total_results=results, wall_time=wall_time)

    if args.output:
        write_report(args.output, args=args, dataset_stats=dataset_stats, results=results, wall_time=wall_time)

    failed = any(not r.success for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
