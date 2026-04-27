from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from eval.core.config import build_runtime_config
from eval.core.reporting import build_report_meta, save_report_payload
from eval.run_store import save_run, utc_now_iso

DocumentReport = List[Dict[str, Any]]


def build_standard_runtime_config(
    *,
    enable_detection: bool,
    enable_deterministic: bool,
    enable_ai: bool,
    enable_anonymization: bool,
    detection_mode: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return build_runtime_config(
        enable_detection=enable_detection,
        enable_deterministic=enable_deterministic,
        enable_ai=enable_ai,
        enable_anonymization=enable_anonymization,
        detection_mode=detection_mode,
        extra=extra,
    )


def save_detailed_report(
    *,
    out_path: str,
    dataset_name: str,
    dataset_path: Optional[str],
    limit: Optional[int],
    config: Mapping[str, Any],
    report: DocumentReport,
    run_name: Optional[str] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    meta = build_report_meta(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        limit=limit,
        config=config,
        run_name=run_name,
        extras=extras,
    )
    save_report_payload(out_path, meta=meta, data=report)
    return meta


def save_optional_run(
    *,
    enabled: bool,
    runs_dir: str,
    report: DocumentReport,
    meta: Mapping[str, Any],
    run_name: Optional[str] = None,
) -> Optional[str]:
    if not enabled:
        return None
    saved = save_run(
        runs_dir,
        meta={**dict(meta), "created_at": utc_now_iso()},
        data=report,
        run_name=run_name,
    )
    return saved
