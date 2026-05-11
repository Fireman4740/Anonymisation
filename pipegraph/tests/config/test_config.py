"""
Tests for configuration settings.
"""

import pytest


def test_security_settings_validation():
    """Test that production mode enforces secure secrets."""
    from src.config import SecuritySettings

    settings = SecuritySettings(
        PSEUDO_SECRET="temp_development_secret_do_not_use_in_prod",
        PSEUDO_SALT="default-salt-change-me",
        ENV="production",
    )
    with pytest.raises(ValueError, match="Cannot use default development secret"):
        settings.validate_secrets()


def test_pipeline_settings_load_from_config_json():
    """Non-secret PipeGraph settings come from config.json, not .env."""
    from src.config import settings

    assert settings.security.ENV == "development"
    assert settings.gpu.ENABLED is True
    assert settings.gpu.BATCH_SIZE == 32
    assert settings.detection.DEFAULT_THRESHOLD == 0.35
    assert settings.detection.PATTERNS_PATH == "config/patterns_config.yaml"


def test_pseudo_mapper_determinism():
    """Test that PseudoMapper produces consistent outputs."""
    from src.utils.pseudo import PseudoMapper

    mapper1 = PseudoMapper(secret="test-secret", scope_id="scope1")
    mapper2 = PseudoMapper(secret="test-secret", scope_id="scope1")

    assert mapper1.placeholder("PER", "John Doe") == mapper2.placeholder("PER", "John Doe")
    assert mapper1.placeholder("PER", "John Doe") != mapper1.placeholder("PER", "Jane Doe")
