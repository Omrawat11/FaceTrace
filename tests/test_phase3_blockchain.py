"""Comprehensive test suite for Phase 3: Evidence Fingerprinting & Blockchain Anchoring.

Covers all 24 required test specifications across 5 core domains:
- Hashing (Tests 1-6)
- Canonicalization (Tests 7-9)
- Blockchain Integration (Tests 10-17)
- Verification & Tamper Detection (Tests 18-21)
- Privacy & Anti-Leakage Guarantees (Tests 22-24)
"""

import hashlib
import json
from pathlib import Path
import pytest
from web3 import Web3, EthereumTesterProvider

from src.blockchain import anchor_evidence, get_blockchain_provider
from src.blockchain.base import AnchorResult, OnChainRecord, TransactionReceipt
from src.blockchain.mock import MockBlockchainProvider
from src.blockchain.service import BlockchainError, EthereumBlockchainService
from src.candidates.base import Candidate, CandidateMatch
from src.core.types import PlatformType, VerificationStatus
from src.evidence.canonical import (
    canonicalize_evidence_dict,
    compute_image_sha256,
    compute_sha256_fingerprint,
    to_bytes32_hex,
)
from src.evidence.models import Evidence, EvidenceRecord
from src.face.base import BoundingBox, DetectedFace
from src.verification.engine import DefaultVerificationEngine, verify_evidence


@pytest.fixture
def local_evm_service() -> EthereumBlockchainService:
    """Fixture providing an in-process local EVM blockchain service with deployed EvidenceRegistry."""
    w3 = Web3(EthereumTesterProvider())
    service = EthereumBlockchainService(w3=w3, network_name="local")
    service.deploy_contract()
    return service


@pytest.fixture
def sample_match() -> CandidateMatch:
    """Fixture providing a realistic CandidateMatch."""
    img_bytes = b"sample_candidate_face_image_bytes_456"
    candidate = Candidate.create(
        url="https://instagram.com/p/sample_anchor_post",
        image_url="https://instagram.com/media/sample.jpg",
        image_bytes=img_bytes,
        title="Sample Verified Instagram Post",
        platform=PlatformType.INSTAGRAM,
    )
    return CandidateMatch(
        candidate=candidate,
        similarity_score=0.8245,
        detected_face=DetectedFace(bbox=BoundingBox(50, 60, 200, 250), confidence=0.98),
        is_match=True,
    )


@pytest.fixture
def sample_evidence_record(sample_match: CandidateMatch) -> EvidenceRecord:
    """Fixture providing a deterministic EvidenceRecord."""
    return EvidenceRecord.create_from_match(
        match=sample_match,
        threshold=0.45,
    )


# ==============================================================================
# Domain 1: Hashing Tests (Tests 1 - 6)
# ==============================================================================

def test_01_same_image_same_image_hash() -> None:
    """Test 1: Same image bytes always generate the exact same SHA-256 hash."""
    img_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRsample_exact_pixels"
    hash_a = compute_image_sha256(img_data)
    hash_b = compute_image_sha256(img_data)
    std_hash = hashlib.sha256(img_data).hexdigest()

    assert hash_a == hash_b
    assert hash_a == std_hash
    assert len(hash_a) == 64


def test_02_different_image_different_image_hash() -> None:
    """Test 2: Different image bytes produce distinct SHA-256 hashes."""
    img_data_1 = b"face_image_dataset_photo_alpha"
    img_data_2 = b"face_image_dataset_photo_beta"

    hash_1 = compute_image_sha256(img_data_1)
    hash_2 = compute_image_sha256(img_data_2)

    assert hash_1 != hash_2
    assert len(hash_1) == 64
    assert len(hash_2) == 64


def test_03_same_evidence_same_evidence_hash(sample_match: CandidateMatch) -> None:
    """Test 3: Identical evidence record contents produce the exact same evidence hash."""
    fixed_timestamp = "2026-09-02T12:00:00Z"
    rec1 = EvidenceRecord(
        post_url=sample_match.candidate.url,
        platform="instagram",
        title="Post Title",
        image_sha256="abcdef1234567890",
        similarity_score=0.8200,
        threshold=0.4500,
        discovered_at=fixed_timestamp,
    )
    rec2 = EvidenceRecord(
        post_url=sample_match.candidate.url,
        platform="instagram",
        title="Post Title",
        image_sha256="abcdef1234567890",
        similarity_score=0.8200,
        threshold=0.4500,
        discovered_at=fixed_timestamp,
    )

    hash1 = rec1.compute_fingerprint()
    hash2 = rec2.compute_fingerprint()

    assert hash1 == hash2
    assert rec1.to_bytes32() == rec2.to_bytes32()
    assert rec1.evidence_hash == hash1


def test_04_changed_evidence_different_evidence_hash(sample_evidence_record: EvidenceRecord) -> None:
    """Test 4: Modifying even one evidence field produces a completely different evidence hash."""
    orig_hash = sample_evidence_record.compute_fingerprint()

    # Alter similarity score by 0.001
    modified_record = EvidenceRecord(
        post_url=sample_evidence_record.post_url,
        platform=sample_evidence_record.platform,
        title=sample_evidence_record.title,
        image_sha256=sample_evidence_record.image_sha256,
        similarity_score=sample_evidence_record.similarity_score + 0.001,
        threshold=sample_evidence_record.threshold,
        discovered_at=sample_evidence_record.discovered_at,
        evidence_version=sample_evidence_record.evidence_version,
    )
    mod_hash = modified_record.compute_fingerprint()

    assert orig_hash != mod_hash
    assert sample_evidence_record.to_bytes32() != modified_record.to_bytes32()


def test_05_correct_bytes32_formatting() -> None:
    """Test 5: to_bytes32_hex formats 64-hex-char fingerprints as 0x-prefixed 32-byte EVM hex."""
    raw_sha256 = "8f31a28892d29486c9fca4cf1d115e5a9528f80bb19c72e259e89d13411b6890"
    b32 = to_bytes32_hex(raw_sha256)

    assert b32.startswith("0x")
    assert len(b32) == 66  # '0x' + 64 hex chars = 32 bytes
    assert b32 == f"0x{raw_sha256}"
    # Verify valid hexadecimal conversion
    as_int = int(b32, 16)
    assert as_int > 0


def test_06_invalid_hash_rejection() -> None:
    """Test 6: Malformed, truncated, or non-hex hashes are rejected by to_bytes32_hex and compute_image_sha256."""
    with pytest.raises(ValueError, match="Fingerprint must be 64 hex characters"):
        to_bytes32_hex("1234abcd")  # Too short

    with pytest.raises(ValueError, match="Fingerprint must be 64 hex characters"):
        to_bytes32_hex("0x" + "aa" * 33)  # Too long

    with pytest.raises(TypeError):
        compute_image_sha256("not_bytes")  # type: ignore


# ==============================================================================
# Domain 2: Canonicalization Tests (Tests 7 - 9)
# ==============================================================================

def test_07_key_ordering_does_not_affect_hash() -> None:
    """Test 7: Dictionaries with different key ordering produce identical canonical JSON and hash."""
    dict_a = {
        "post_url": "https://reddit.com/r/face/1",
        "platform": "reddit",
        "image_sha256": "3fd9c94c5ea2ba4cbf8aa531502f7f1da450901f3f45225ab84fd3b6c95399a6",
        "similarity_score": 0.8123,
        "threshold": 0.4500,
    }
    dict_b = {
        "threshold": 0.4500,
        "similarity_score": 0.8123,
        "image_sha256": "3fd9c94c5ea2ba4cbf8aa531502f7f1da450901f3f45225ab84fd3b6c95399a6",
        "platform": "reddit",
        "post_url": "https://reddit.com/r/face/1",
    }

    canon_a = canonicalize_evidence_dict(dict_a)
    canon_b = canonicalize_evidence_dict(dict_b)

    assert canon_a == canon_b
    assert compute_sha256_fingerprint(canon_a) == compute_sha256_fingerprint(canon_b)


def test_08_relevant_value_changes_affect_hash(sample_evidence_record: EvidenceRecord) -> None:
    """Test 8: Changing individual fields (post_url, platform, image_sha256) changes the evidence hash."""
    base_dict = sample_evidence_record.to_dict()
    base_hash = compute_sha256_fingerprint(canonicalize_evidence_dict(base_dict))

    for key, alt_val in [
        ("post_url", "https://otherplatform.com/post/999"),
        ("platform", "twitter"),
        ("image_sha256", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
        ("title", "Alternative Title"),
    ]:
        tampered_dict = dict(base_dict)
        tampered_dict[key] = alt_val
        tampered_hash = compute_sha256_fingerprint(canonicalize_evidence_dict(tampered_dict))
        assert base_hash != tampered_hash, f"Modifying '{key}' should change evidence hash"


def test_09_existing_canonicalization_tests_remain_passing() -> None:
    """Test 9: Phase 1 Evidence model canonicalization remains fully functional."""
    evidence = Evidence(
        version="1.0.0",
        source_url="https://example.com/photo/123",
        platform="instagram",
        title="Test Photo",
        candidate_image_hash="abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        similarity_score=0.8800,
        discovery_timestamp="2026-09-02T12:00:00Z",
    )
    canon_json = evidence.to_canonical_json()
    fp = evidence.compute_fingerprint()
    b32 = evidence.to_bytes32()

    assert len(canon_json) > 0
    assert len(fp) == 64
    assert b32.startswith("0x")
    assert len(b32) == 66


# ==============================================================================
# Domain 3: Blockchain Integration Tests (Tests 10 - 17)
# ==============================================================================

def test_10_contract_deployment() -> None:
    """Test 10: Contract deployment creates a valid EvidenceRegistry instance at an EVM address."""
    w3 = Web3(EthereumTesterProvider())
    service = EthereumBlockchainService(w3=w3, network_name="local")
    contract_addr = service.deploy_contract()

    assert contract_addr is not None
    assert Web3.is_address(contract_addr)
    assert service.contract_address == contract_addr


def test_11_contract_address_configuration() -> None:
    """Test 11: Blockchain service correctly respects configured contract address."""
    w3 = Web3(EthereumTesterProvider())
    deployer_service = EthereumBlockchainService(w3=w3, network_name="local")
    deployed_address = deployer_service.deploy_contract()

    # Re-instantiate service passing pre-configured contract address
    client_service = EthereumBlockchainService(
        w3=w3,
        contract_address=deployed_address,
        network_name="local",
    )
    assert client_service.contract_address == deployed_address


def test_12_evidence_anchoring_via_anchor_evidence(
    local_evm_service: EthereumBlockchainService, sample_evidence_record: EvidenceRecord
) -> None:
    """Test 12: High-level anchor_evidence() completes full anchoring and returns AnchorResult."""
    b32_fp = sample_evidence_record.to_bytes32()
    result = local_evm_service.anchor_evidence(b32_fp, sample_evidence_record.post_url)

    assert isinstance(result, AnchorResult)
    assert result.success is True
    assert result.is_success is True
    assert result.status == 1
    assert result.block_number >= 1
    assert result.contract_address == local_evm_service.contract_address
    assert result.stored_evidence_hash.lower() == b32_fp.lower()
    assert result.timestamp is not None and result.timestamp > 0
    assert result.error is None


def test_13_successful_transaction_receipt(
    local_evm_service: EthereumBlockchainService, sample_evidence_record: EvidenceRecord
) -> None:
    """Test 13: Submitting evidence creates a confirmed on-chain transaction receipt."""
    b32_fp = sample_evidence_record.to_bytes32()
    receipt = local_evm_service.record_fingerprint(b32_fp, sample_evidence_record.post_url)

    assert isinstance(receipt, TransactionReceipt)
    assert receipt.is_success is True
    assert receipt.status == 1
    assert receipt.block_number >= 1
    assert len(receipt.tx_hash) >= 64
    assert receipt.gas_used > 21000


def test_14_evidence_retrieval(
    local_evm_service: EthereumBlockchainService, sample_evidence_record: EvidenceRecord
) -> None:
    """Test 14: get_evidence() retrieves confirmed on-chain record with timestamp and block number."""
    b32_fp = sample_evidence_record.to_bytes32()
    local_evm_service.record_fingerprint(b32_fp, sample_evidence_record.post_url)

    record = local_evm_service.get_evidence(b32_fp)
    assert isinstance(record, OnChainRecord)
    assert record.exists is True
    assert record.fingerprint.lower() == b32_fp.lower()
    assert record.source_url == sample_evidence_record.post_url
    assert record.timestamp > 0
    assert record.block_number > 0
    assert Web3.is_address(record.recorder)
    assert record.formatted_time != "N/A"


def test_15_stored_hash_matches_submitted_hash(
    local_evm_service: EthereumBlockchainService, sample_evidence_record: EvidenceRecord
) -> None:
    """Test 15: The hash retrieved from the smart contract matches the submitted hash exactly."""
    submitted_hash = sample_evidence_record.to_bytes32()
    local_evm_service.record_fingerprint(submitted_hash, sample_evidence_record.post_url)

    on_chain_record = local_evm_service.get_evidence(submitted_hash)
    assert on_chain_record.fingerprint.lower() == submitted_hash.lower()


def test_16_transaction_failure_handling(
    local_evm_service: EthereumBlockchainService, sample_evidence_record: EvidenceRecord
) -> None:
    """Test 16: Duplicate fingerprint submission reverts with EvidenceAlreadyRecorded."""
    b32_fp = sample_evidence_record.to_bytes32()
    local_evm_service.record_fingerprint(b32_fp, sample_evidence_record.post_url)

    with pytest.raises(BlockchainError):
        local_evm_service.record_fingerprint(b32_fp, sample_evidence_record.post_url)

    # anchor_evidence handles the failure gracefully returning success=False
    fail_result = local_evm_service.anchor_evidence(b32_fp, sample_evidence_record.post_url)
    assert fail_result.success is False
    assert fail_result.is_success is False
    assert fail_result.error is not None


def test_17_missing_evidence_handling(local_evm_service: EthereumBlockchainService) -> None:
    """Test 17: Querying an unrecorded fingerprint returns exists=False and has_evidence=False."""
    unanchored_fp = "0x" + "ee" * 32
    record = local_evm_service.get_evidence(unanchored_fp)

    assert record.exists is False
    assert record.timestamp == 0
    assert record.block_number == 0
    assert local_evm_service.has_evidence(unanchored_fp) is False


# ==============================================================================
# Domain 4: Verification Tests (Tests 18 - 21)
# ==============================================================================

def test_18_correct_evidence_verified(
    local_evm_service: EthereumBlockchainService, sample_evidence_record: EvidenceRecord
) -> None:
    """Test 18: Unmodified, anchored evidence returns VERIFIED status."""
    local_evm_service.anchor_evidence(sample_evidence_record.to_bytes32(), sample_evidence_record.post_url)

    result = verify_evidence(sample_evidence_record, blockchain_provider=local_evm_service)

    assert result.is_verified is True
    assert result.status == VerificationStatus.VERIFIED
    assert result.on_chain_record is not None
    assert result.tamper_reasons == []


def test_19_modified_evidence_tampered(
    local_evm_service: EthereumBlockchainService, sample_evidence_record: EvidenceRecord
) -> None:
    """Test 19: When evidence is modified after anchoring, verifier detects TAMPERED state."""
    original_fp = sample_evidence_record.to_bytes32()
    local_evm_service.anchor_evidence(original_fp, sample_evidence_record.post_url)

    # Tamper with similarity score
    tampered_record = EvidenceRecord(
        post_url=sample_evidence_record.post_url,
        platform=sample_evidence_record.platform,
        title=sample_evidence_record.title,
        image_sha256=sample_evidence_record.image_sha256,
        similarity_score=sample_evidence_record.similarity_score + 0.05,  # Altered!
        threshold=sample_evidence_record.threshold,
        discovered_at=sample_evidence_record.discovered_at,
        evidence_version=sample_evidence_record.evidence_version,
    )

    # Verify tampered record against expected anchored fingerprint
    tamper_result = verify_evidence(
        tampered_record,
        blockchain_provider=local_evm_service,
        expected_fingerprint=original_fp,
    )

    assert tamper_result.is_verified is False
    assert tamper_result.status == VerificationStatus.TAMPERED
    assert len(tamper_result.tamper_reasons) > 0


def test_20_missing_blockchain_record_not_found(
    local_evm_service: EthereumBlockchainService, sample_evidence_record: EvidenceRecord
) -> None:
    """Test 20: Evidence not yet anchored on-chain returns NOT_FOUND."""
    unanchored_evidence = EvidenceRecord(
        post_url="https://example.com/unanchored/post",
        platform="web",
        title="Unanchored Photo",
        image_sha256="7777777777777777777777777777777777777777777777777777777777777777",
        similarity_score=0.75,
        threshold=0.45,
        discovered_at="2026-09-02T10:00:00Z",
    )

    result = verify_evidence(unanchored_evidence, blockchain_provider=local_evm_service)

    assert result.is_verified is False
    assert result.status == VerificationStatus.NOT_FOUND
    assert "not been anchored" in result.tamper_reasons[0].lower()


def test_21_blockchain_failure_blockchain_error(
    sample_evidence_record: EvidenceRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 21: Blockchain communication / RPC failure produces BLOCKCHAIN_ERROR status."""
    service = MockBlockchainProvider()

    def failing_get_evidence(fingerprint: str):
        raise ConnectionError("RPC Node Unreachable: simulated network failure")

    monkeypatch.setattr(service, "get_evidence", failing_get_evidence)

    result = verify_evidence(sample_evidence_record, blockchain_provider=service)

    assert result.is_verified is False
    assert result.status == VerificationStatus.BLOCKCHAIN_ERROR
    assert len(result.tamper_reasons) > 0
    assert "network failure" in result.tamper_reasons[0]


# ==============================================================================
# Domain 5: Privacy & Anti-Leakage Tests (Tests 22 - 24)
# ==============================================================================

def test_22_raw_image_never_sent_to_blockchain(
    local_evm_service: EthereumBlockchainService, sample_match: CandidateMatch
) -> None:
    """Test 22: Verify raw candidate image bytes are never written to the blockchain or contract."""
    evidence = EvidenceRecord.create_from_match(sample_match)
    b32_fp = evidence.to_bytes32()
    raw_img = sample_match.candidate.image_bytes

    # Submit transaction
    receipt = local_evm_service.record_fingerprint(b32_fp, evidence.post_url)
    tx = local_evm_service.w3.eth.get_transaction(receipt.tx_hash)

    # 1. Transaction calldata input must not contain raw image bytes
    tx_input = tx.input.hex() if hasattr(tx.input, "hex") else str(tx.input)
    assert raw_img.hex() not in tx_input

    # 2. On-chain record must not contain raw image bytes
    record = local_evm_service.get_evidence(b32_fp)
    assert not hasattr(record, "image_bytes")
    assert not hasattr(record, "raw_image")


def test_23_face_embedding_never_sent_to_blockchain(
    local_evm_service: EthereumBlockchainService, sample_match: CandidateMatch
) -> None:
    """Test 23: Verify 512-dimensional face biometric embeddings are never anchored on-chain."""
    evidence = EvidenceRecord.create_from_match(sample_match)
    b32_fp = evidence.to_bytes32()

    # EvidenceRecord dictionary must have no embeddings
    evidence_dict = evidence.to_dict()
    assert "embedding" not in evidence_dict
    assert "face_embedding" not in evidence_dict
    assert "biometric_vector" not in evidence_dict

    receipt = local_evm_service.record_fingerprint(b32_fp, evidence.post_url)
    record = local_evm_service.get_evidence(b32_fp)

    assert not hasattr(record, "embedding")
    assert not hasattr(record, "vector")


def test_24_api_keys_never_enter_evidence_records(sample_evidence_record: EvidenceRecord) -> None:
    """Test 24: Verify API keys, secret credentials, or environment tokens never enter evidence records."""
    evidence_dict = sample_evidence_record.to_dict()
    canonical_json = sample_evidence_record.to_canonical_json()

    forbidden_substrings = ["api_key", "secret", "private_key", "password", "token", "serpapi"]
    for key in evidence_dict.keys():
        for forbidden in forbidden_substrings:
            assert forbidden not in key.lower(), f"Forbidden credential key found: {key}"

    for forbidden in forbidden_substrings:
        assert forbidden not in canonical_json.lower(), f"Forbidden credential found in canonical JSON: {forbidden}"
