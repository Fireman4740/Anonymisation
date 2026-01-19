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


class PipelineSettings(BaseSettings):
    """Main aggregator for settings."""

    security: SecuritySettings = Field(default_factory=lambda: SecuritySettings())
    gpu: GPUSettings = Field(default_factory=lambda: GPUSettings())
    detection: DetectionSettings = Field(default_factory=lambda: DetectionSettings())

    DEBUG: bool = Field(default=False, validation_alias="PIPEGRAPH_DEBUG")


# Singleton instance
settings = PipelineSettings()
