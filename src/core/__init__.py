"""Core module containing configuration, logging, domain types, and base exceptions."""

from src.core.config import Settings, get_settings
from src.core.exceptions import (
    BlockchainError,
    CandidateDownloadError,
    ConfigurationError,
    EvidenceCanonicalizationError,
    FaceDetectionError,
    FaceEmbeddingError,
    FaceTraceError,
    SearchProviderError,
    VerificationError,
)
from src.core.logging import setup_logger
from src.core.types import BlockchainNetwork, PlatformType, VerificationStatus

__all__ = [
    "Settings",
    "get_settings",
    "setup_logger",
    "PlatformType",
    "VerificationStatus",
    "BlockchainNetwork",
    "FaceTraceError",
    "ConfigurationError",
    "FaceDetectionError",
    "FaceEmbeddingError",
    "SearchProviderError",
    "CandidateDownloadError",
    "EvidenceCanonicalizationError",
    "BlockchainError",
    "VerificationError",
]
