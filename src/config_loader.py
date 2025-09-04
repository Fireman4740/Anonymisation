import json
import os
from typing import Any, Dict, Optional

_CONFIG_CACHE: Optional[Dict[str, Any]] = None

DEFAULT_CONFIG_PATHS = [
    os.path.join(os.getcwd(), "config.json"),
    os.path.join(os.path.dirname(__file__), "..", "config.json"),
]


def load_config(path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """Charge la configuration JSON avec cache.

    Recherche par défaut dans cwd/config.json puis ../config.json par rapport à ce fichier.
    """
    global _CONFIG_CACHE
    if not force_reload and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    candidate_paths = [path] if path else DEFAULT_CONFIG_PATHS
    for p in candidate_paths:
        if not p:
            continue
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            with open(p_abs, "r", encoding="utf-8") as f:
                _CONFIG_CACHE = json.load(f)
                return _CONFIG_CACHE
    # fallback: config minimale
    _CONFIG_CACHE = {
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "retry_count": 1,
            "fallback_model": "openai/gpt-4o-mini",
            "models": {}
        }
    }
    return _CONFIG_CACHE


def get_model_overrides(cfg: Dict[str, Any]) -> Dict[str, str]:
    models = (cfg.get("openrouter") or {}).get("models") or {}
    out = {}
    for k, v in models.items():
        if isinstance(k, str) and isinstance(v, str) and v:
            out[k] = v
    return out


def openrouter_client_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    o = cfg.get("openrouter") or {}
    return {
        "base_url": o.get("base_url", "https://openrouter.ai/api/v1"),
        "retry_count": int(o.get("retry_count", 1) or 1),
        "fallback_model": o.get("fallback_model", "openai/gpt-4o-mini"),
    }
