from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

LLM_CONFIG_ALIASES: Dict[str, str] = {
    "enable_llm_detection": "llm_detection",
    "enable_llm_audit": "llm_audit",
    "enable_llm_paraphrase": "llm_paraphrase",
    "enable_llm_verification": "llm_verification",
}


def normalize_runtime_config(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Normalize runtime config keys used across CLI, UI and ablations.

    The codebase historically used both ``enable_llm_*`` and ``llm_*`` keys.
    PipeGraph reads the canonical ``llm_*`` keys at runtime, so this helper
    keeps backward compatibility while exposing a single config contract.
    """
    runtime: Dict[str, Any] = dict(config or {})

    for legacy_key, canonical_key in LLM_CONFIG_ALIASES.items():
        if canonical_key not in runtime and legacy_key in runtime:
            runtime[canonical_key] = bool(runtime[legacy_key])

    return runtime


def build_runtime_config(
    *,
    enable_detection: bool,
    enable_deterministic: bool,
    enable_ai: bool,
    enable_anonymization: bool,
    detection_mode: str,
    llm_detection: Optional[bool] = None,
    llm_audit: Optional[bool] = None,
    llm_paraphrase: Optional[bool] = None,
    llm_verification: Optional[bool] = None,
    rupta_enabled: Optional[bool] = None,
    rupta_max_iterations: Optional[int] = None,
    rupta_p_threshold: Optional[int] = None,
    disable_llm: Optional[bool] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "enable_detection": bool(enable_detection),
        "enable_deterministic": bool(enable_deterministic),
        "enable_ai": bool(enable_ai),
        "enable_anonymization": bool(enable_anonymization),
        "detection_mode": str(detection_mode),
    }

    optional_values = {
        "llm_detection": llm_detection,
        "llm_audit": llm_audit,
        "llm_paraphrase": llm_paraphrase,
        "llm_verification": llm_verification,
        "rupta_enabled": rupta_enabled,
        "rupta_max_iterations": rupta_max_iterations,
        "rupta_p_threshold": rupta_p_threshold,
        "disable_llm": disable_llm,
    }
    for key, value in optional_values.items():
        if value is not None:
            config[key] = value

    if extra:
        config.update(dict(extra))

    return normalize_runtime_config(config)
