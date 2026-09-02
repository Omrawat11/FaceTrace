"""Evidence packaging, canonicalization, and cryptographic hashing module."""

from src.evidence.canonical import (
    canonicalize_evidence_dict,
    compute_image_sha256,
    compute_sha256_fingerprint,
    normalize_evidence_dict,
    to_bytes32_hex,
)
from src.evidence.models import Evidence, EvidenceRecord

__all__ = [
    "Evidence",
    "EvidenceRecord",
    "canonicalize_evidence_dict",
    "compute_image_sha256",
    "compute_sha256_fingerprint",
    "normalize_evidence_dict",
    "to_bytes32_hex",
]
