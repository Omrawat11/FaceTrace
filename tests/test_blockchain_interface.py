"""Unit tests for blockchain models, ABI definition, and verification results."""

from src.blockchain.abi import EVIDENCE_REGISTRY_ABI
from src.blockchain.base import OnChainRecord, TransactionReceipt
from src.core.types import VerificationStatus
from src.verification.base import VerificationResult


def test_evidence_registry_abi_structure() -> None:
    """Test that ABI defines essential functions: recordEvidence, getEvidence, hasEvidence."""
    function_names = {
        item["name"]
        for item in EVIDENCE_REGISTRY_ABI
        if item.get("type") in ("function", "event")
    }
    assert "recordEvidence" in function_names
    assert "getEvidence" in function_names
    assert "hasEvidence" in function_names
    assert "EvidenceRecorded" in function_names


def test_on_chain_record_formatted_time() -> None:
    """Test timestamp conversion on OnChainRecord."""
    record = OnChainRecord(
        exists=True,
        fingerprint="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        recorder="0x90F79bf6EB2c4f870365E785982E1f101E93b906",
        timestamp=1700000000,
        block_number=4500123,
        source_url="https://example.com/evidence",
    )
    assert record.exists is True
    assert record.formatted_time == "2023-11-14 22:13:20Z"


def test_transaction_receipt_status() -> None:
    """Test is_success logic on TransactionReceipt."""
    success_rcpt = TransactionReceipt(
        tx_hash="0xabc123",
        block_number=100,
        block_hash="0xdef456",
        gas_used=65000,
        status=1,
    )
    assert success_rcpt.is_success is True

    failed_rcpt = TransactionReceipt(
        tx_hash="0xabc123",
        block_number=100,
        block_hash="0xdef456",
        gas_used=21000,
        status=0,
    )
    assert failed_rcpt.is_success is False


def test_verification_result_states() -> None:
    """Test VerificationResult helper factories."""
    record = OnChainRecord(
        exists=True,
        fingerprint="0x" + "aa" * 32,
        recorder="0x" + "11" * 20,
        timestamp=1700000000,
        block_number=10,
        source_url="https://example.com",
    )

    success_res = VerificationResult.success(
        fingerprint="0x" + "aa" * 32,
        on_chain_record=record,
        source_url="https://example.com",
    )
    assert success_res.is_verified is True
    assert success_res.status == VerificationStatus.VERIFIED

    not_found_res = VerificationResult.not_found(
        fingerprint="0x" + "bb" * 32,
        source_url="https://example.com",
    )
    assert not_found_res.is_verified is False
    assert not_found_res.status == VerificationStatus.NOT_FOUND

    tampered_res = VerificationResult.tampered(
        fingerprint="0x" + "cc" * 32,
        on_chain_record=record,
        source_url="https://example.com",
        reasons=["Fingerprint mismatch"],
    )
    assert tampered_res.is_verified is False
    assert tampered_res.status == VerificationStatus.TAMPERED
