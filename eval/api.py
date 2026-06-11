"""High-level evaluation API — single entry point for CLI, UI and scripts.

Wraps the official engine (``eval.run_pipeline_evaluation.run_evaluation``):
all metric/scoring/artifact logic stays there; this module only translates
config files into runner arguments and organises run directories.

Python usage::

    from eval.api import EvaluationRunner

    runner = EvaluationRunner.from_config("configs/evaluation/no_llm.json")
    payload = runner.run(dataset="tab")
    payload = runner.run(dataset="all")
    summary = runner.run_ablation(dataset="tab",
                                  ablation_config="configs/evaluation/ablations/default.json")

Every run directory contains ``run_config.json``, ``summary.json``,
``summary.md``, ``candidate_effective_config.json`` and
``datasets/<key>/{documents.jsonl,metrics.json}`` (written by the engine).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from eval.core.config import build_runtime_config
from eval.registry import get_registry

ConfigLike = Union[Dict[str, Any], str, os.PathLike, None]

DEFAULT_OUTPUT_ROOT = os.path.join("runs", "evaluation")

# Keys of an evaluation config file and their runner-argument defaults.
_RUNNER_ARG_DEFAULTS: Dict[str, Any] = {
    "datasets": None,            # None → all registry datasets
    "split": "test",
    "limit": 20,
    "doc_workers": 1,
    "profile": "auto",
    "eval_mode": "both",
    "masking_mode": "benchmark",
    "language": "english",
    "ratbench_languages": None,
    "ratbench_levels": [1],
    "llm_provider": None,
    "llm_model": None,
    "llm_attacker_model": None,
    "no_llm": False,
    "skip_risk": False,
    "require_risk": False,
    "risk_limit": None,
    "save_runs": False,
    "candidate": None,
}


def build_eval_config(
    *,
    dataset: str,
    profile: str = "auto",
    eval_mode: str = "both",
    masking_mode: str = "benchmark",
    no_llm: bool = False,
    detection_mode: str = "parallel",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    detection_threshold: Optional[float] = None,
    paraphrase_intensity: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a PipeGraph runtime config for evaluation UI/services."""
    cfg = build_runtime_config(
        enable_detection=True,
        enable_deterministic=True,
        enable_ai=True,
        enable_anonymization=True,
        detection_mode=detection_mode,
        dataset_key=dataset,
        profile=profile,
        eval_mode=eval_mode,
        masking_mode=masking_mode,
    )
    if no_llm:
        cfg.update(
            {
                "disable_llm": True,
                "llm_detection": False,
                "llm_verification": False,
                "llm_audit": False,
                "llm_paraphrase": False,
                "rupta_enabled": False,
            }
        )
    if llm_provider:
        cfg["llm_provider"] = llm_provider
    if llm_model:
        cfg["llm_model"] = llm_model
    if detection_threshold is not None:
        cfg["detection_threshold"] = detection_threshold
    if paraphrase_intensity is not None:
        cfg["paraphrase_intensity"] = paraphrase_intensity
    if extra:
        cfg.update(extra)
    return cfg


def load_eval_config(config: ConfigLike) -> Dict[str, Any]:
    """Dict passthrough or JSON/YAML file. ``_``-prefixed keys are comments."""
    if config is None:
        return {}
    if isinstance(config, dict):
        raw = config
    else:
        path = os.fspath(config)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Evaluation config not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            if path.endswith((".yaml", ".yml")):
                import yaml

                raw = yaml.safe_load(handle)
            else:
                raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Evaluation config must be a mapping")
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


class EvaluationRunner:
    """Config-driven facade over the official evaluation engine."""

    def __init__(self, config: ConfigLike = None, config_name: str = "default"):
        self.config = load_eval_config(config)
        self.config_name = str(self.config.get("run_name") or config_name)
        self.output_root = str(self.config.get("output_root") or DEFAULT_OUTPUT_ROOT)

    @classmethod
    def from_config(cls, path: Union[str, os.PathLike]) -> "EvaluationRunner":
        name = os.path.splitext(os.path.basename(os.fspath(path)))[0]
        return cls(path, config_name=name)

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def _resolve_datasets(self, dataset: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(dataset, str):
            if dataset.lower() == "all":
                configured = self.config.get("datasets")
                if configured:
                    return list(configured)
                # default to what the engine supports end-to-end
                from eval.run_pipeline_evaluation import DEFAULT_DATASETS

                return list(DEFAULT_DATASETS)
            return [dataset]
        return list(dataset)

    def _build_args(
        self,
        datasets: List[str],
        output_dir: str,
        overrides: Dict[str, Any],
    ) -> argparse.Namespace:
        values = dict(_RUNNER_ARG_DEFAULTS)
        for key in values:
            if key in self.config:
                values[key] = self.config[key]
        values.update({k: v for k, v in overrides.items() if k in _RUNNER_ARG_DEFAULTS})
        values["datasets"] = datasets
        values["output"] = output_dir
        return argparse.Namespace(**values)

    def _write_candidate(
        self, output_dir: Path, pipeline_overrides: Dict[str, Any], name: str
    ) -> str:
        """Pipeline config overrides travel as a candidate JSON (engine contract)."""
        candidate_path = output_dir / "candidate.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps({"candidate_id": name, "config": pipeline_overrides}, indent=2),
            encoding="utf-8",
        )
        return str(candidate_path)

    def run(
        self,
        dataset: Union[str, Sequence[str]] = "all",
        output: Optional[str] = None,
        pipeline_overrides: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Evaluate one dataset, a list, or ``"all"``. Returns the engine payload."""
        from eval.run_pipeline_evaluation import run_evaluation

        datasets = self._resolve_datasets(dataset)
        label = datasets[0] if len(datasets) == 1 else "all"
        output_dir = Path(
            output or os.path.join(self.output_root, f"{_utc_stamp()}_{label}_{self.config_name}")
        )

        effective_overrides = dict(overrides)
        merged_pipeline_overrides = {
            **(self.config.get("pipeline_overrides") or {}),
            **(pipeline_overrides or {}),
        }
        if merged_pipeline_overrides and not effective_overrides.get("candidate"):
            effective_overrides["candidate"] = self._write_candidate(
                output_dir, merged_pipeline_overrides, name=self.config_name
            )

        args = self._build_args(datasets, str(output_dir), effective_overrides)
        started = time.time()
        payload = run_evaluation(args)
        payload["wall_time_s"] = round(time.time() - started, 3)
        payload["config_name"] = self.config_name

        # errors.jsonl at the run root: one line per dataset-level error,
        # so failed documents/datasets are inspectable without parsing summary.json
        errors = payload.get("errors") or []
        errors_path = output_dir / "errors.jsonl"
        with errors_path.open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, ensure_ascii=False, default=str) + "\n")
        return payload

    # ------------------------------------------------------------------
    # Ablation
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        dataset: Union[str, Sequence[str]],
        ablation_config: ConfigLike,
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run every ablation variant, then write a comparative summary.

        Ablation config format::

            {"ablations": [{"name": "no_llm", "overrides": {...state.config keys...},
                            "runner_overrides": {"no_llm": true}}, ...]}
        """
        plan = load_ablation_plan(ablation_config)
        datasets = self._resolve_datasets(dataset)
        label = datasets[0] if len(datasets) == 1 else "all"
        root_dir = Path(
            output or os.path.join("runs", "ablations", f"{_utc_stamp()}_{label}_{self.config_name}")
        )
        root_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for variant in plan:
            variant_dir = root_dir / variant["name"]
            payload = self.run(
                dataset=datasets,
                output=str(variant_dir),
                pipeline_overrides=variant.get("overrides") or {},
                **(variant.get("runner_overrides") or {}),
            )
            results.append(
                {
                    "name": variant["name"],
                    "primary_metric": payload.get("primary_metric"),
                    "status": payload.get("status"),
                    "wall_time_s": payload.get("wall_time_s"),
                    "run_dir": str(variant_dir),
                    "datasets": {
                        key: value.get("score")
                        for key, value in (payload.get("datasets") or {}).items()
                    },
                }
            )

        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": datasets,
            "config_name": self.config_name,
            "variants": results,
        }
        (root_dir / "ablation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_ablation_csv(root_dir / "ablation_summary.csv", results)
        (root_dir / "ablation_report.md").write_text(
            _render_ablation_report(summary), encoding="utf-8"
        )
        summary["output_dir"] = str(root_dir)
        return summary


# ----------------------------------------------------------------------
# Ablation plan
# ----------------------------------------------------------------------

def load_ablation_plan(config: ConfigLike) -> List[Dict[str, Any]]:
    raw = load_eval_config(config)
    variants = raw.get("ablations")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Ablation config must contain a non-empty 'ablations' list")
    plan: List[Dict[str, Any]] = []
    seen = set()
    for item in variants:
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError(f"Ablation variant without a name: {item}")
        if name in seen:
            raise ValueError(f"Duplicate ablation variant name: {name}")
        seen.add(name)
        plan.append(
            {
                "name": name,
                "overrides": dict(item.get("overrides") or {}),
                "runner_overrides": dict(item.get("runner_overrides") or {}),
            }
        )
    return plan


def _write_ablation_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    dataset_keys = sorted({key for row in results for key in row.get("datasets", {})})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "primary_metric", "status", "wall_time_s", *dataset_keys])
        for row in results:
            writer.writerow(
                [
                    row["name"],
                    row.get("primary_metric"),
                    row.get("status"),
                    row.get("wall_time_s"),
                    *[row.get("datasets", {}).get(key) for key in dataset_keys],
                ]
            )


def _render_ablation_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Ablation Report",
        "",
        f"- Date: {summary['created_at']}",
        f"- Datasets: {', '.join(summary['dataset'])}",
        f"- Base config: {summary['config_name']}",
        "",
        "| Variant | Primary metric | Status | Wall time (s) |",
        "| --- | --- | --- | --- |",
    ]
    ranked = sorted(
        summary["variants"], key=lambda r: (r.get("primary_metric") or 0.0), reverse=True
    )
    for row in ranked:
        metric = row.get("primary_metric")
        metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "n/a"
        lines.append(
            f"| {row['name']} | {metric_str} | {row.get('status')} | {row.get('wall_time_s')} |"
        )
    lines += ["", "Per-variant artifacts live in the sub-directories named after each variant."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# High-level helpers (UI / scripts)
# ----------------------------------------------------------------------

def evaluate_dataset(
    dataset_name: str, config_path: ConfigLike = None, output_dir: Optional[str] = None, **overrides: Any
) -> Dict[str, Any]:
    return EvaluationRunner(config_path).run(dataset=dataset_name, output=output_dir, **overrides)


def evaluate_all(
    config_path: ConfigLike = None, output_dir: Optional[str] = None, **overrides: Any
) -> Dict[str, Any]:
    return EvaluationRunner(config_path).run(dataset="all", output=output_dir, **overrides)


def run_ablation_study(
    dataset_name: str,
    config_path: ConfigLike,
    ablation_config_path: ConfigLike,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    return EvaluationRunner(config_path).run_ablation(
        dataset=dataset_name, ablation_config=ablation_config_path, output=output_dir
    )


def compare_runs(run_dirs: Sequence[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Compare summary.json of several run directories. Writes comparison artifacts."""
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = load_metrics(run_dir)
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_id": summary.get("run_id"),
                "primary_metric": summary.get("primary_metric"),
                "status": summary.get("status"),
                "datasets": {
                    key: value.get("score")
                    for key, value in (summary.get("datasets") or {}).items()
                },
            }
        )

    baseline = rows[0] if rows else None
    for row in rows:
        if baseline and isinstance(row.get("primary_metric"), (int, float)) and isinstance(
            baseline.get("primary_metric"), (int, float)
        ):
            row["delta_vs_first"] = round(row["primary_metric"] - baseline["primary_metric"], 6)

    comparison = {"created_at": datetime.now(timezone.utc).isoformat(), "runs": rows}

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        dataset_keys = sorted({key for row in rows for key in row.get("datasets", {})})
        with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run_dir", "run_id", "primary_metric", "delta_vs_first", *dataset_keys])
            for row in rows:
                writer.writerow(
                    [
                        row["run_dir"],
                        row.get("run_id"),
                        row.get("primary_metric"),
                        row.get("delta_vs_first"),
                        *[row.get("datasets", {}).get(key) for key in dataset_keys],
                    ]
                )
        lines = [
            "# Run Comparison",
            "",
            "| Run | Primary metric | Δ vs first |",
            "| --- | --- | --- |",
        ]
        for row in rows:
            metric = row.get("primary_metric")
            metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "n/a"
            delta = row.get("delta_vs_first")
            delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "—"
            lines.append(f"| {os.path.basename(str(row['run_dir']))} | {metric_str} | {delta_str} |")
        (out / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        comparison["output_dir"] = str(out)

    return comparison


# ----------------------------------------------------------------------
# Run artifact accessors (UI)
# ----------------------------------------------------------------------

def list_available_datasets() -> List[Dict[str, Any]]:
    return get_registry().describe_all()


def list_available_configs(config_root: str = "configs/evaluation") -> List[str]:
    root = Path(config_root)
    if not root.exists():
        return []
    return sorted(
        str(path)
        for path in root.rglob("*")
        if path.suffix in (".json", ".yaml", ".yml") and path.is_file()
    )


def load_metrics(run_dir: Union[str, os.PathLike]) -> Dict[str, Any]:
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary.json in run dir: {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation_report(run_dir: Union[str, os.PathLike]) -> str:
    path = Path(run_dir) / "summary.md"
    if not path.exists():
        raise FileNotFoundError(f"No summary.md in run dir: {run_dir}")
    return path.read_text(encoding="utf-8")


def load_predictions(
    run_dir: Union[str, os.PathLike], dataset: Optional[str] = None
) -> List[Dict[str, Any]]:
    base = Path(run_dir) / "datasets"
    if not base.exists():
        raise FileNotFoundError(f"No datasets/ directory in run dir: {run_dir}")
    rows: List[Dict[str, Any]] = []
    for documents in sorted(base.glob("*/documents.jsonl")):
        if dataset and documents.parent.name != dataset:
            continue
        with documents.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows
