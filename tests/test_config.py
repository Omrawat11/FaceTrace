"""Unit tests for configuration management and secret masking."""

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def test_settings_defaults() -> None:
    """Test default values of Settings."""
    settings = Settings()
    assert settings.ENVIRONMENT == "development"
    assert settings.LOG_LEVEL == "INFO"
    assert settings.FACE_SIMILARITY_THRESHOLD == 0.45
    assert settings.FACE_DETECTION_CONFIDENCE == 0.60
    assert settings.SEARCH_PROVIDER == "serpapi"
    assert settings.BLOCKCHAIN_PROVIDER == "mock"


def test_settings_threshold_validation() -> None:
    """Test that similarity and confidence thresholds must fall in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        Settings(FACE_SIMILARITY_THRESHOLD=1.5)

    with pytest.raises(ValidationError):
        Settings(FACE_SIMILARITY_THRESHOLD=-0.1)

    with pytest.raises(ValidationError):
        Settings(FACE_DETECTION_CONFIDENCE=2.0)


def test_settings_safe_dict_redaction() -> None:
    """Test that safe_dict redacts sensitive API keys and blockchain private keys."""
    settings = Settings(
        SERPAPI_API_KEY="secret-serp-key-12345",
        BING_SEARCH_API_KEY="secret-bing-key-67890",
        ETH_PRIVATE_KEY="0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )
    safe = settings.safe_dict()
    assert safe["SERPAPI_API_KEY"] == "***REDACTED***"
    assert safe["BING_SEARCH_API_KEY"] == "***REDACTED***"
    assert safe["ETH_PRIVATE_KEY"] == "***REDACTED***"
    # Verify non-sensitive fields are intact
    assert safe["SEARCH_PROVIDER"] == "serpapi"
