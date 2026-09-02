"""Deterministic canonicalization utilities and cryptographic hashing for Evidence.

CRYPTOGRAPHIC HASH DISTINCTION:
1. IMAGE HASH:
   SHA256(exact raw image bytes)
   Proves whether the exact candidate image file or pixels have been altered.

2. EVIDENCE HASH:
   SHA256(RFC 8785 canonicalized JSON of EvidenceRecord)
   Cryptographically fingerprints the complete provenance record (URLs, platform,
   similarity scores, discovery timestamp, and image hash) for on-chain anchoring.
"""

import hashlib
import json
from typing import Any
import unicodedata

from src.core.exceptions import EvidenceCanonicalizationError


def compute_image_sha256(image_bytes: bytes) -> str:
    """Calculate the deterministic 64-character hex SHA-256 digest of raw image bytes.

    This fingerprints the candidate image content directly, proving whether
    the exact downloaded bytes have changed.
    """
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray for image hashing, got {type(image_bytes).__name__}")
    return hashlib.sha256(image_bytes).hexdigest()



def normalize_string(val: str | None) -> str | None:
    """Normalize string using Unicode NFC and strip surrounding whitespace."""
    if val is None:
        return None
    return unicodedata.normalize("NFC", val.strip())


def normalize_evidence_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively clean, sort, and normalize evidence dictionary values for deterministic serialization."""
    normalized: dict[str, Any] = {}
    for key, val in sorted(data.items()):
        # Exclude internal or transient keys
        if key.startswith("_"):
            continue

        if isinstance(val, str):
            normalized[key] = normalize_string(val)
        elif isinstance(val, float):
            # Round floats to 4 decimal places to ensure cross-platform precision consistency
            normalized[key] = round(val, 4)
        elif isinstance(val, dict):
            normalized[key] = normalize_evidence_dict(val)
        elif isinstance(val, list):
            # Normalize list elements
            normalized_list = []
            for item in val:
                if isinstance(item, float):
                    normalized_list.append(round(item, 4))
                elif isinstance(item, str):
                    normalized_list.append(normalize_string(item))
                elif isinstance(item, dict):
                    normalized_list.append(normalize_evidence_dict(item))
                else:
                    normalized_list.append(item)
            normalized[key] = normalized_list
        else:
            normalized[key] = val
    return normalized


def canonicalize_evidence_dict(data: dict[str, Any]) -> str:
    """Produce deterministic, RFC 8785-compliant canonical JSON representation.

    Rules:
    - Keys are strictly sorted in lexicographical order.
    - Compact separators (no whitespace after ':' or ',').
    - Unicode NFC normalization.
    - UTF-8 representation without ASCII escaping.
    """
    try:
        clean_dict = normalize_evidence_dict(data)
        return json.dumps(
            clean_dict,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except Exception as exc:
        raise EvidenceCanonicalizationError(
            f"Failed to produce canonical representation: {exc}",
            details={"raw_data": data},
        ) from exc


def compute_sha256_fingerprint(canonical_json: str) -> str:
    """Compute 64-character hexadecimal SHA-256 hash of UTF-8 canonical string."""
    canonical_bytes = canonical_json.encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def to_bytes32_hex(fingerprint_hex: str) -> str:
    """Format 64-character hex string as a 0x-prefixed 32-byte hex for EVM contracts."""
    cleaned = fingerprint_hex.lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    if len(cleaned) != 64:
        raise ValueError(f"Fingerprint must be 64 hex characters (32 bytes), got {len(cleaned)}")
    return f"0x{cleaned}"
