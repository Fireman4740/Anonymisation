"""Centralized Pydantic settings for PipeGraph."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import os


_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "../../config.json"))


def load_config() -> Dict[str, Any]:
    """Load PipeGraph config.json once per process."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = _config_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = json.load(f)
    else:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def _section(name: str) -> Dict[str, Any]:
    value = load_config().get(name, {})
    return value if isinstance(value, dict) else {}


class SecuritySettings(BaseModel):
    """Security-related settings (secrets, salts, etc)."""

    PSEUDO_SECRET: SecretStr = Field(
        default_factory=lambda: _section("security").get(
            "pseudo_secret", "temp_development_secret_do_not_use_in_prod"
        ),
        description="Master secret for pseudonymization hashing.",
    )
    PSEUDO_SALT: str = Field(
        default_factory=lambda: _section("security").get("pseudo_salt", "default-salt-change-me"),
        description="Salt for pseudonymization.",
    )
    ENV: str = Field(
        default_factory=lambda: _section("security").get("env", "development"),
        description="Environment: development or production",
    )

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    def pseudo_secret_value(self) -> str:
        value = self.PSEUDO_SECRET
        if hasattr(value, "get_secret_value"):
            return value.get_secret_value()
        return str(value)

    def validate_secrets(self):
        """Ensure secrets are strong in production."""
        if self.is_production:
            if (
                self.pseudo_secret_value()
                == "temp_development_secret_do_not_use_in_prod"
            ):
                raise ValueError("Cannot use default development secret in production!")
            if self.PSEUDO_SALT == "default-salt-change-me":
                raise ValueError("Cannot use default salt in production!")


class GPUSettings(BaseModel):
    """GPU and Model optimization settings."""

    ENABLED: bool = Field(default_factory=lambda: bool(_section("ner_gpu").get("enabled", False)), description="Enable GPU acceleration")
    VRAM_GB: int = Field(default_factory=lambda: int(_section("ner_gpu").get("vram_gb", 24)), description="Available VRAM in GB")
    BATCH_SIZE: int = Field(default_factory=lambda: int(_section("ner_gpu").get("batch_size", 32)), description="Inference batch size")
    USE_FP16: bool = Field(default_factory=lambda: bool(_section("ner_gpu").get("use_fp16", True)), description="Use FP16 precision")
    MAX_PARALLEL_MODELS: int = Field(default_factory=lambda: int(_section("ner_gpu").get("max_parallel_models", 3)), description="Max concurrent models")
    COMPILE: bool = Field(default_factory=lambda: bool(_section("ner_gpu").get("use_torch_compile", False)), description="Use torch.compile")


class DetectionSettings(BaseModel):
    """Global detection settings."""

    DEFAULT_THRESHOLD: float = Field(
        default_factory=lambda: float(_section("detection").get("default_threshold", 0.35)),
        description="Default confidence threshold",
    )
    PATTERNS_PATH: str = Field(
        default_factory=lambda: str(_section("detection").get("patterns_path", "config/patterns_config.yaml")),
        description="Path to regex patterns",
    )

    # Path resolution helper
    def resolve_patterns_path(self, root_dir: str) -> str:
        if os.path.isabs(self.PATTERNS_PATH):
            return self.PATTERNS_PATH
        return os.path.abspath(os.path.join(root_dir, self.PATTERNS_PATH))


class LLMEnvSettings(BaseSettings):
    """
    LLM secrets loaded from environment variables / .env file.
    Keeps API keys out of config.json (which may be committed to VCS).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- API keys ---
    OPENROUTER_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="OpenRouter API key. Set OPENROUTER_API_KEY in .env to use OpenRouter as LLM provider.",
    )

    # --- Helpers ---
    @property
    def openrouter_key(self) -> str:
        """Return the raw API key string, or raise if not configured."""
        if self.OPENROUTER_API_KEY is None:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file: OPENROUTER_API_KEY=sk-or-v1-..."
            )
        return self.OPENROUTER_API_KEY.get_secret_value()

    @property
    def has_openrouter_key(self) -> bool:
        return self.OPENROUTER_API_KEY is not None

class PipelineSettings(BaseModel):
    """Main aggregator for settings."""

    security: SecuritySettings = Field(default_factory=lambda: SecuritySettings())
    gpu: GPUSettings = Field(default_factory=lambda: GPUSettings())
    detection: DetectionSettings = Field(default_factory=lambda: DetectionSettings())
    llm_env: LLMEnvSettings = Field(default_factory=lambda: LLMEnvSettings())

    DEBUG: bool = Field(default_factory=lambda: bool(_section("runtime").get("debug", False)))


class LLMSettings:
    """
    LLM configuration loaded from config.json (not env-based to avoid secrets exposure).
    Provides typed access to the llm, features, policy_defaults and rupta sections.
    """

    def __init__(self, config_path: Optional[str] = None):
        import json

        if config_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.abspath(os.path.join(here, "../../config.json"))

        self._raw: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self._raw = json.load(f)

    # --- LLM connection --------------------------------------------------
    @property
    def base_url(self) -> str:
        return self._raw.get("llm", {}).get("base_url", "http://localhost:11434/v1")

    @property
    def timeout(self) -> int:
        return int(self._raw.get("llm", {}).get("timeout_seconds", 90))

    @property
    def retry_count(self) -> int:
        return int(self._raw.get("llm", {}).get("retry_count", 1))

    def model_for(self, role: str) -> str:
        llm = self._raw.get("llm", {})
        return llm.get("models", {}).get(role) or llm.get("model", "openai/gpt-oss-20b")

    @property
    def supports_response_format(self) -> bool:
        return bool(self._raw.get("llm", {}).get("supports_response_format", False))

    # --- Feature flags ---------------------------------------------------
    @property
    def llm_detection_enabled(self) -> bool:
        return bool(self._raw.get("features", {}).get("llm_detection", True))

    @property
    def llm_audit_enabled(self) -> bool:
        return bool(self._raw.get("features", {}).get("llm_audit", True))

    @property
    def llm_paraphrase_enabled(self) -> bool:
        return bool(self._raw.get("features", {}).get("llm_paraphrase", True))

    # --- RUPTA -----------------------------------------------------------
    @property
    def rupta_enabled(self) -> bool:
        return bool(self._raw.get("rupta", {}).get("enabled", False))

    @property
    def rupta_max_iterations(self) -> int:
        return int(self._raw.get("rupta", {}).get("max_iterations", 3))

    @property
    def rupta_p_threshold(self) -> int:
        """Privacy score above which we trigger paraphrase (0-100)."""
        return int(self._raw.get("rupta", {}).get("p_threshold", 15))

    # --- Policy ----------------------------------------------------------
    @property
    def paraphrase_intensity(self) -> int:
        return int(self._raw.get("policy_defaults", {}).get("paraphrase_intensity", 1))



# Singleton instance
settings = PipelineSettings()
