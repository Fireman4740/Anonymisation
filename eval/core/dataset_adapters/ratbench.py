from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from eval.core.dataset_adapters.base import (
    DatasetAdapter,
    DatasetEvaluationContext,
    DatasetRunRequest,
    DocumentReport,
)
from eval.core.io import safe_key, write_jsonl
from eval.core.metrics import gold_text_leakage, runtime_metrics, score_dataset_axes, span_detection_axes
from eval.core.ratbench import compute_leak_summary, direct_id_detection_rate, metrics_by_difficulty, metrics_by_scenario
from eval.core.reporting import aggregate_document_metrics
from eval.core.loaders.ratbench import evaluate_text_leaks, get_ratbench_metadata


def _arg_list(args: Any, name: str, default: list[Any]) -> list[Any]:
    value = getattr(args, name, None)
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _serialize_risk_details(details: Any) -> List[Dict[str, Any]]:
    if details is None:
        return []
    if hasattr(details, "empty") and hasattr(details, "to_dict"):
        if bool(getattr(details, "empty")):
            return []
        return list(details.to_dict(orient="records"))
    if isinstance(details, list):
        return [dict(item) for item in details if isinstance(item, Mapping)]
    return []


class RatbenchDatasetAdapter(DatasetAdapter):
    def expand_run_requests(self, args: Any) -> List[DatasetRunRequest]:
        languages = [str(item) for item in _arg_list(args, "ratbench_languages", [getattr(args, "language", "english")])]
        levels = [int(item) for item in _arg_list(args, "ratbench_levels", [1])]
        split = str(getattr(args, "split", self.default_split) or self.default_split)
        return [
            DatasetRunRequest(
                dataset=self.name,
                dataset_key=f"ratbench/{language}/L{level}",
                split=split,
                language=language,
                level=level,
                limit=getattr(args, "limit", None),
            )
            for language in languages
            for level in levels
        ]

    def protocol_metadata(
        self,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
    ) -> Dict[str, Any]:
        return {
            "name": "RAT-Bench",
            "protocol": "profile_value_search_plus_reidentification_risk",
            "annotation_status": "value_search_no_offsets",
            "language": request.language,
            "level": request.level,
            "warning": "RAT-Bench profiles provide attribute values; char offsets are derived by value matching.",
        }

    def enrich_report(
        self,
        report: DocumentReport,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
        runtime_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        profiles = context.load_ratbench_profiles(
            language=request.language,
            level=request.level,
            limit=request.limit,
        )
        profile_by_doc_id = {
            f"ratbench_{profile.get('id', '')}_L{profile.get('difficulty', '?')}": profile
            for profile in profiles
        }
        for document in report:
            profile = profile_by_doc_id.get(str(document.get("doc_id", "")))
            if not profile:
                continue
            original_text = str(document.get("full_text") or "")
            anonymized_text = str(document.get("anonymized_text") or original_text)
            document["ratbench_metadata"] = get_ratbench_metadata(profile)
            document["text_leak_analysis"] = evaluate_text_leaks(
                original_text=original_text,
                anonymized_text=anonymized_text,
                profile=profile,
            )
        return {"profiles": profiles}

    def _risk_axis(
        self,
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
        runtime_config: Mapping[str, Any],
        report: DocumentReport,
        profiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        args = context.args
        if getattr(args, "skip_risk", False):
            return {
                "status": "risk_skipped",
                "protocol": "ratbench_reidentification_risk",
                "reason": "disabled_by_cli",
            }
        if not context.openrouter_key_available(context.root):
            message = "OPENROUTER_API_KEY not set; RAT-Bench LLM attacker skipped."
            if getattr(args, "require_risk", False):
                raise RuntimeError(message)
            return {
                "status": "risk_degraded",
                "protocol": "ratbench_reidentification_risk",
                "provider": "openrouter",
                "error": message,
            }

        from eval.cli.evaluate_ratbench_risk import evaluate_ratbench_risk_from_pipeline

        payload = evaluate_ratbench_risk_from_pipeline(
            context.pipeline,
            context.create_initial_state,
            profiles,
            config=dict(runtime_config),
            limit=getattr(args, "risk_limit", None) or request.limit,
            report=report,
        )
        if payload.get("error"):
            if getattr(args, "require_risk", False):
                raise RuntimeError(str(payload["error"]))
            return {
                "status": "risk_degraded",
                "protocol": "ratbench_reidentification_risk",
                "provider": "openrouter",
                "error": payload["error"],
            }

        metrics = dict(payload.get("metrics") or {})
        details = _serialize_risk_details(payload.get("detailed_results"))
        details_path = context.output_dir / "ratbench" / f"{safe_key(request.dataset_key)}_risk_details.jsonl"
        if details:
            write_jsonl(details_path, details)
        return {
            "status": "risk_full",
            "protocol": "ratbench_reidentification_risk",
            "provider": "openrouter",
            "attacker_model_policy": "eval.cli.evaluate_ratbench_risk._OPENROUTER_MODELS",
            "details_path": str(details_path) if details else None,
            **metrics,
        }

    def build_axes(
        self,
        report: DocumentReport,
        aggregate: Dict[str, Any],
        request: DatasetRunRequest,
        context: DatasetEvaluationContext,
        runtime_config: Mapping[str, Any],
        elapsed_s: float,
        enrichment: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        aggregate.update(compute_leak_summary(report))
        aggregate["by_difficulty"] = metrics_by_difficulty(report)
        aggregate["by_scenario"] = metrics_by_scenario(report)
        aggregate["direct_id_detection_rates"] = direct_id_detection_rate(report)

        span_axis = span_detection_axes(report, aggregate)
        leakage_axis = gold_text_leakage(report)
        leak_summary = compute_leak_summary(report)
        leakage_axis["ratbench_profile_leakage"] = {
            **leak_summary,
            "avg_protection_rate": round(1.0 - float(leak_summary.get("avg_leak_rate", 0.0)), 4),
            "avg_direct_protection_rate": round(1.0 - float(leak_summary.get("avg_direct_leak_rate", 0.0)), 4),
            "avg_indirect_protection_rate": round(1.0 - float(leak_summary.get("avg_indirect_leak_rate", 0.0)), 4),
        }
        runtime_axis = runtime_metrics(report, elapsed_s)
        risk_axis = self._risk_axis(
            request,
            context,
            runtime_config,
            report,
            list(enrichment.get("profiles") or []),
        )
        utility_axis = {"status": "not_applicable", "protocol": "not_applicable", "score": None}
        score_details = score_dataset_axes(
            dataset=self.name,
            span_detection=span_axis,
            anonymization_leakage=leakage_axis,
            runtime=runtime_axis,
            ratbench_reid_risk=risk_axis,
            utility_preservation=utility_axis,
        )
        return (
            {
                "span_detection": span_axis,
                "anonymization_leakage": leakage_axis,
                "utility_preservation": utility_axis,
                "runtime": runtime_axis,
                "ratbench_reid_risk": risk_axis,
            },
            score_details,
        )
