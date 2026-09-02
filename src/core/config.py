"""Configuration management for FaceTrace using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """FaceTrace application and pipeline configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # Face Recognition
    FACE_DETECTOR_PROVIDER: Literal["insightface", "mock"] = "insightface"
    FACE_SIMILARITY_THRESHOLD: float = Field(default=0.45, ge=0.0, le=1.0)
    FACE_DETECTION_CONFIDENCE: float = Field(default=0.60, ge=0.0, le=1.0)

    # Search Provider
    SEARCH_PROVIDER: Literal["serpapi", "bing", "mock"] = "serpapi"
    SERPAPI_API_KEY: str | None = None
    BING_SEARCH_API_KEY: str | None = None
    BING_SEARCH_ENDPOINT: str = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
    MAX_SEARCH_RESULTS: int = Field(default=10, ge=1, le=50)
    SEARCH_REQUEST_TIMEOUT_SECONDS: int = Field(default=15, ge=1, le=60)

    # Blockchain (Sepolia)
    BLOCKCHAIN_PROVIDER: Literal["sepolia", "local", "mock"] = "mock"
    SEPOLIA_RPC_URL: str = "https://rpc.sepolia.org"
    ETH_PRIVATE_KEY: str | None = None
    EVIDENCE_REGISTRY_CONTRACT_ADDRESS: str | None = None
    BLOCKCHAIN_GAS_LIMIT: int = Field(default=200000, ge=21000)

    # Directories
    DATA_DIR: Path = Path("data")
    DEMO_IMAGES_DIR: Path = Path("data/demo_images")
    CANDIDATES_CACHE_DIR: Path = Path("data/candidates")

    @field_validator("FACE_SIMILARITY_THRESHOLD", "FACE_DETECTION_CONFIDENCE")
    @classmethod
    def validate_thresholds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v

    def safe_dict(self) -> dict:
        """Return configuration dictionary with sensitive credentials redacted."""
        data = self.model_dump()
        sensitive_keys = {"SERPAPI_API_KEY", "BING_SEARCH_API_KEY", "ETH_PRIVATE_KEY"}
        for key in sensitive_keys:
            if data.get(key):
                data[key] = "***REDACTED***"
        return data


# Global singleton settings instance
def get_settings() -> Settings:
    """Retrieve settings instance, initialized from environment."""
    return Settings()
