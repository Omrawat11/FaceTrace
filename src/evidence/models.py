"""Deterministic Evidence and EvidenceRecord models and cryptographic fingerprinting."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.candidates.base import CandidateMatch
from src.evidence.canonical import (
    canonicalize_evidence_dict,
    compute_sha256_fingerprint,
    to_bytes32_hex,
)


@dataclass
class EvidenceRecord:
    """Structured, minimal evidence record representing a verified matching face discovery.

    Contains exclusively non-biometric, privacy-preserving provenance metadata:
    NO face embeddings, NO raw face scans, NO biometric vectors.
    """

    post_url: str
    platform: str
    title: str
    image_sha256: str
    similarity_score: float
    threshold: float
    discovered_at: str
    evidence_version: str = "1.0.0"
    caption: str | None = None

    def __post_init__(self) -> None:
        # Guarantee 4 decimal places for floating point consistency across architectures
        self.similarity_score = round(self.similarity_score, 4)
        self.threshold = round(self.threshold, 4)

    @property
    def similarity_threshold(self) -> float:
        """Alias for threshold for semantic compatibility."""
        return self.threshold

    @property
    def evidence_hash(self) -> str:
        """Hexadecimal SHA-256 fingerprint of the canonical evidence representation."""
        return self.compute_fingerprint()

    @classmethod
    def create_from_match(
        cls,
        match: CandidateMatch,
        threshold: float = 0.45,
        similarity_threshold: float | None = None,
        caption: str | None = None,
        version: str = "1.0.0",
        timestamp: datetime | None = None,
    ) -> "EvidenceRecord":
        """Factory creating a structured EvidenceRecord from an evaluated CandidateMatch."""
        ts = timestamp or datetime.now(timezone.utc)
        iso_timestamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        effective_threshold = similarity_threshold if similarity_threshold is not None else threshold

        return cls(
            post_url=match.candidate.url,
            platform=match.candidate.platform.value,
            title=match.candidate.title,
            image_sha256=match.candidate.image_hash,
            similarity_score=match.similarity_score,
            threshold=effective_threshold,
            discovered_at=iso_timestamp,
            evidence_version=version,
            caption=caption,
        )

    def to_dict(self) -> dict:
        """Convert evidence fields to a serializable dictionary."""
        return asdict(self)

    def to_canonical_json(self) -> str:
        """Generate deterministic, RFC 8785-compliant canonical JSON representation."""
        return canonicalize_evidence_dict(self.to_dict())

    def compute_fingerprint(self) -> str:
        """Compute the 64-character hexadecimal SHA-256 fingerprint from the canonical JSON."""
        return compute_sha256_fingerprint(self.to_canonical_json())

    def to_bytes32(self) -> str:
        """Format the fingerprint as an EVM-compatible 0x-prefixed 32-byte hex string."""
        return to_bytes32_hex(self.compute_fingerprint())


@dataclass
class Evidence:
    """Backward-compatible Evidence model from Phase 1."""

    version: str
    source_url: str
    platform: str
    title: str
    candidate_image_hash: str
    similarity_score: float
    discovery_timestamp: str
    caption: str | None = None
    face_box: list[int] | None = None

    def __post_init__(self) -> None:
        self.similarity_score = round(self.similarity_score, 4)

    @classmethod
    def create_from_match(
        cls,
        match: CandidateMatch,
        caption: str | None = None,
        version: str = "1.0.0",
        timestamp: datetime | None = None,
    ) -> "Evidence":
        """Factory creating an Evidence instance from a CandidateMatch."""
        ts = timestamp or datetime.now(timezone.utc)
        iso_timestamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        face_box = (
            match.detected_face.bbox.as_list()
            if match.detected_face and match.detected_face.bbox
            else None
        )

        return cls(
            version=version,
            source_url=match.candidate.url,
            platform=match.candidate.platform.value,
            title=match.candidate.title,
            caption=caption,
            candidate_image_hash=match.candidate.image_hash,
            similarity_score=match.similarity_score,
            discovery_timestamp=iso_timestamp,
            face_box=face_box,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_canonical_json(self) -> str:
        return canonicalize_evidence_dict(self.to_dict())

    def compute_fingerprint(self) -> str:
        return compute_sha256_fingerprint(self.to_canonical_json())

    def to_bytes32(self) -> str:
        return to_bytes32_hex(self.compute_fingerprint())
