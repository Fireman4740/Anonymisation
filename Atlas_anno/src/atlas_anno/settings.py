from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from atlas_anno.config import load_config
from atlas_anno.paths import project_root


def parse_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class AtlasSettings:
    openrouter_api_key: str
    openrouter_base_url: str
    atlas_model_reasoning: str
    atlas_model_creative: str
    http_timeout_seconds: int

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openrouter_api_key.strip())


def load_settings(env_path: str | None = None) -> AtlasSettings:
    env_file = Path(env_path) if env_path else project_root() / ".env"
    env_values = parse_dotenv(env_file)
    config = load_config()
    llm_defaults = config.defaults.get("llm", {})
    model_defaults = llm_defaults.get("models", {})

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name, env_values.get(name, default))

    return AtlasSettings(
        openrouter_api_key=get("OPENROUTER_API_KEY"),
        openrouter_base_url=get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"),
        atlas_model_reasoning=get("ATLAS_MODEL_REASONING", str(model_defaults.get("reasoning", "aion-labs/aion-2.0"))),
        atlas_model_creative=get("ATLAS_MODEL_CREATIVE", str(model_defaults.get("creative", "mistralai/mistral-small-creative"))),
        http_timeout_seconds=int(get("ATLAS_HTTP_TIMEOUT_SECONDS", "60") or "60"),
    )
