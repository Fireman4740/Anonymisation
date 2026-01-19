"""
Tests for configuration settings.
"""

import pytest
import os


# Skip all tests if the required env var is not set (avoids import errors)
pytestmark = pytest.mark.skipif(
    not os.environ.get("PIPEGRAPH_SEC_PSEUDO_SECRET"), reason="PIPEGRAPH_SEC_PSEUDO_SECRET not set"
)


def test_security_settings_validation():
    """Test that production mode enforces secure secrets."""
    from src.config import SecuritySettings

    # Simulate production environment
    original_env = os.environ.get("PIPEGRAPH_SEC_ENV")
    original_secret = os.environ.get("PIPEGRAPH_SEC_PSEUDO_SECRET")

    try:
        os.environ["PIPEGRAPH_SEC_ENV"] = "production"
        os.environ["PIPEGRAPH_SEC_PSEUDO_SECRET"] = "temp_development_secret_do_not_use_in_prod"

        with pytest.raises(ValueError, match="Cannot use default development secret"):
            SecuritySettings().validate_secrets()
    finally:
        # Restore original values
        if original_env:
            os.environ["PIPEGRAPH_SEC_ENV"] = original_env
        else:
            os.environ.pop("PIPEGRAPH_SEC_ENV", None)
        if original_secret:
            os.environ["PIPEGRAPH_SEC_PSEUDO_SECRET"] = original_secret


def test_pseudo_mapper_determinism():
    """Test that PseudoMapper produces consistent outputs."""
    from src.utils.pseudo import PseudoMapper

    mapper1 = PseudoMapper(secret="test-secret", scope_id="scope1")
    mapper2 = PseudoMapper(secret="test-secret", scope_id="scope1")

    assert mapper1.placeholder("PER", "John Doe") == mapper2.placeholder("PER", "John Doe")
    assert mapper1.placeholder("PER", "John Doe") != mapper1.placeholder("PER", "Jane Doe")
