"""
Centralized Pydantic settings for PipeGraph.
Handles environment variables, secrets, and configuration defaults.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class SecuritySettings(BaseSettings):
    """Security-related settings (secrets, salts, etc)."""

    model_config = SettingsConfigDict(env_prefix="PIPEGRAPH_SEC_", env_file=".env", extra="ignore")

    PSEUDO_SECRET: SecretStr = Field(
        default="temp_development_secret_do_not_use_in_prod",
        description="Master secret for pseudonymization hashing. MUST be set in production via PIPEGRAPH_SEC_PSEUDO_SECRET env var.",
    )
    PSEUDO_SALT: str = Field(
        "default-salt-change-me",
        description="Salt for pseudonymization. Should be changed in production.",
    )
    ENV: str = Field("development", description="Environment: development or production")

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    def validate_secrets(self):
        """Ensure secrets are strong in production."""
        if self.is_production:
            if (
                self.PSEUDO_SECRET.get_secret_value()
                == "temp_development_secret_do_not_use_in_prod"
            ):
                raise ValueError("Cannot use default development secret in production!")
            if self.PSEUDO_SALT == "default-salt-change-me":
                raise ValueError("Cannot use default salt in production!")


class GPUSettings(BaseSettings):
    """GPU and Model optimization settings."""

    model_config = SettingsConfigDict(env_prefix="NER_GPU_", env_file=".env", extra="ignore")

    ENABLED: bool = Field(False, description="Enable GPU acceleration")
    VRAM_GB: int = Field(24, description="Available VRAM in GB")
    BATCH_SIZE: int = Field(32, description="Inference batch size")
    USE_FP16: bool = Field(True, description="Use FP16 precision")
    MAX_PARALLEL_MODELS: int = Field(3, description="Max concurrent models")
    COMPILE: bool = Field(False, description="Use torch.compile")


class DetectionSettings(BaseSettings):
    """Global detection settings."""

    model_config = SettingsConfigDict(env_prefix="PIPEGRAPH_DET_", env_file=".env", extra="ignore")

    DEFAULT_THRESHOLD: float = Field(0.35, description="Default confidence threshold")
    PATTERNS_PATH: str = Field("config/patterns_config.yaml", description="Path to regex patterns")

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

    # --- OpenRouter model overrides ---
    # Global (all roles): OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
    OPENROUTER_MODEL: Optional[str] = Field(default=None, description="OpenRouter model for all roles (overrides config.json)")
    OPENROUTER_MODEL_DETECT: Optional[str] = Field(default=None, description="OpenRouter model for detection role")
    OPENROUTER_MODEL_AUDIT: Optional[str] = Field(default=None, description="OpenRouter model for audit role")
    OPENROUTER_MODEL_PARAPHRASE: Optional[str] = Field(default=None, description="OpenRouter model for paraphrase role")

    # --- Ollama / local model overrides ---
    # Global (all roles): LLM_MODEL=openai/gpt-oss-20b
    LLM_MODEL: Optional[str] = Field(default=None, description="Local LLM model for all roles (overrides config.json)")
    LLM_MODEL_DETECT: Optional[str] = Field(default=None, description="Local model for detection role")
    LLM_MODEL_AUDIT: Optional[str] = Field(default=None, description="Local model for audit role")
    LLM_MODEL_PARAPHRASE: Optional[str] = Field(default=None, description="Local model for paraphrase role")

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

    def model_for(self, role: str, provider: str) -> Optional[str]:
        """
        Return the env-configured model for the given role+provider, or None.
        Priority: role-specific > global.
        """
        if provider == "openrouter":
            per_role = {
                "detect": self.OPENROUTER_MODEL_DETECT,
                "audit": self.OPENROUTER_MODEL_AUDIT,
                "paraphrase": self.OPENROUTER_MODEL_PARAPHRASE,
            }.get(role)
            return per_role or self.OPENROUTER_MODEL
        else:
            per_role = {
                "detect": self.LLM_MODEL_DETECT,
                "audit": self.LLM_MODEL_AUDIT,
                "paraphrase": self.LLM_MODEL_PARAPHRASE,
            }.get(role)
            return per_role or self.LLM_MODEL


class PipelineSettings(BaseSettings):
    """Main aggregator for settings."""

    security: SecuritySettings = Field(default_factory=lambda: SecuritySettings())
    gpu: GPUSettings = Field(default_factory=lambda: GPUSettings())
    detection: DetectionSettings = Field(default_factory=lambda: DetectionSettings())
    llm_env: LLMEnvSettings = Field(default_factory=lambda: LLMEnvSettings())

    DEBUG: bool = Field(default=False, validation_alias="PIPEGRAPH_DEBUG")


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
        return self._raw.get("llm", {}).get("models", {}).get(role, "openai/gpt-oss-20b")

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
