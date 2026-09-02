"""Unit tests for Evidence canonicalization and cryptographic fingerprint generation."""

import pytest

from src.evidence.canonical import (
    canonicalize_evidence_dict,
    compute_sha256_fingerprint,
    to_bytes32_hex,
)
from src.evidence.models import Evidence


def test_canonical_json_key_order_invariance() -> None:
    """Test that two dictionaries with different key insertion order produce identical canonical JSON."""
    dict_a = {
        "version": "1.0.0",
        "source_url": "https://example.com/post/1",
        "title": "Sample Title",
        "similarity_score": 0.8524,
    }
    dict_b = {
        "title": "Sample Title",
        "similarity_score": 0.8524,
        "version": "1.0.0",
        "source_url": "https://example.com/post/1",
    }

    canon_a = canonicalize_evidence_dict(dict_a)
    canon_b = canonicalize_evidence_dict(dict_b)

    assert canon_a == canon_b
    assert compute_sha256_fingerprint(canon_a) == compute_sha256_fingerprint(canon_b)


def test_float_precision_stability() -> None:
    """Test that float values are normalized to 4 decimal places."""
    dict_float_1 = {"score": 0.8524000000000001}
    dict_float_2 = {"score": 0.8524}

    canon_1 = canonicalize_evidence_dict(dict_float_1)
    canon_2 = canonicalize_evidence_dict(dict_float_2)

    assert canon_1 == canon_2


def test_tamper_detection_avalanche_effect(sample_evidence: Evidence) -> None:
    """Test that modifying even one character in an Evidence field completely changes the fingerprint."""
    original_fp = sample_evidence.compute_fingerprint()

    # Tamper with title
    tampered_evidence = Evidence(
        version=sample_evidence.version,
        source_url=sample_evidence.source_url,
        platform=sample_evidence.platform,
        title="Tampered Title",  # Changed!
        caption=sample_evidence.caption,
        candidate_image_hash=sample_evidence.candidate_image_hash,
        similarity_score=sample_evidence.similarity_score,
        discovery_timestamp=sample_evidence.discovery_timestamp,
        face_box=sample_evidence.face_box,
    )
    tampered_fp = tampered_evidence.compute_fingerprint()

    assert original_fp != tampered_fp


def test_bytes32_hex_format(sample_evidence: Evidence) -> None:
    """Test that to_bytes32() produces a valid 0x-prefixed 64-hex-char string (32 bytes)."""
    b32 = sample_evidence.to_bytes32()
    assert b32.startswith("0x")
    # 2 chars for '0x' + 64 hex chars = 66 total chars
    assert len(b32) == 66
    # Valid hexadecimal
    int(b32, 16)


def test_invalid_bytes32_length() -> None:
    """Test that invalid hex lengths raise ValueError."""
    with pytest.raises(ValueError):
        to_bytes32_hex("1234abcd")  # Too short
