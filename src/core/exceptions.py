"""Domain exceptions hierarchy for FaceTrace."""


class FaceTraceError(Exception):
    """Base exception for all FaceTrace domain errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(FaceTraceError):
    """Raised when environment or runtime configuration is invalid or missing."""
    pass


class FaceDetectionError(FaceTraceError):
    """Raised when face detection fails or no valid face is present."""
    pass


class FaceEmbeddingError(FaceTraceError):
    """Raised when biometric embedding extraction fails."""
    pass


class SearchProviderError(FaceTraceError):
    """Raised when genuine search API request fails or returns invalid response."""
    pass


class CandidateDownloadError(FaceTraceError):
    """Raised when fetching candidate image or web page fails."""
    pass


class EvidenceCanonicalizationError(FaceTraceError):
    """Raised when evidence dictionary canonicalization fails."""
    pass


class BlockchainError(FaceTraceError):
    """Raised when blockchain interaction, RPC call, or transaction fails."""
    pass


class BlockchainTransactionReverted(BlockchainError):
    """Raised when an on-chain transaction reverts."""
    pass


class VerificationError(FaceTraceError):
    """Raised during the evidence verification step."""
    pass
