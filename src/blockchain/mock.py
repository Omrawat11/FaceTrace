"""Deterministic mock blockchain provider for testing without an active EVM node."""

import hashlib
import time
from typing import Dict

from src.blockchain.base import AnchorResult, BlockchainProvider, OnChainRecord, TransactionReceipt
from src.evidence.canonical import to_bytes32_hex


class MockBlockchainProvider(BlockchainProvider):
    """Simulated in-memory blockchain provider storing evidence fingerprints."""

    def __init__(self) -> None:
        self._records: Dict[str, OnChainRecord] = {}
        self._tx_counter: int = 1
        self._block_counter: int = 100

    @property
    def network_name(self) -> str:
        return "mock"

    def record_fingerprint(
        self,
        fingerprint: str,
        source_url: str,
    ) -> TransactionReceipt:
        """Record fingerprint into in-memory dictionary and return deterministic receipt."""
        b32_fp = to_bytes32_hex(fingerprint).lower()

        if b32_fp in self._records:
            raise ValueError(f"Evidence already recorded for fingerprint: {b32_fp}")

        current_time = int(time.time())
        self._block_counter += 1
        tx_hash_bytes = hashlib.sha256(f"tx_{self._tx_counter}_{b32_fp}".encode()).hexdigest()
        tx_hash = f"0x{tx_hash_bytes}"
        block_hash = f"0x{hashlib.sha256(f'block_{self._block_counter}'.encode()).hexdigest()}"

        record = OnChainRecord(
            exists=True,
            fingerprint=b32_fp,
            recorder="0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf",
            timestamp=current_time,
            block_number=self._block_counter,
            source_url=source_url,
        )
        self._records[b32_fp] = record
        self._tx_counter += 1

        return TransactionReceipt(
            tx_hash=tx_hash,
            block_number=self._block_counter,
            block_hash=block_hash,
            gas_used=48500,
            status=1,
        )

    def get_evidence(self, fingerprint: str) -> OnChainRecord:
        """Retrieve simulated on-chain record."""
        b32_fp = to_bytes32_hex(fingerprint).lower()
        if b32_fp in self._records:
            return self._records[b32_fp]

        return OnChainRecord(
            exists=False,
            fingerprint=b32_fp,
            recorder="0x0000000000000000000000000000000000000000",
            timestamp=0,
            block_number=0,
            source_url="",
        )

    def has_evidence(self, fingerprint: str) -> bool:
        """Check if fingerprint exists in simulated store."""
        b32_fp = to_bytes32_hex(fingerprint).lower()
        return b32_fp in self._records

    def anchor_evidence(
        self,
        evidence_hash: str,
        source_url: str = "",
    ) -> AnchorResult:
        """Simulated end-to-end evidence anchoring flow."""
        b32_fp = to_bytes32_hex(evidence_hash).lower()
        mock_contract = "0xF2E246BB76DF876Cef8b38ae84130F4F55De395b"

        try:
            receipt = self.record_fingerprint(b32_fp, source_url)
            on_chain = self.get_evidence(b32_fp)

            return AnchorResult(
                success=True,
                tx_hash=receipt.tx_hash,
                block_number=receipt.block_number,
                contract_address=mock_contract,
                stored_evidence_hash=on_chain.fingerprint,
                timestamp=on_chain.timestamp,
                status=receipt.status,
                error=None,
            )
        except Exception as exc:
            return AnchorResult(
                success=False,
                tx_hash="",
                block_number=0,
                contract_address=mock_contract,
                stored_evidence_hash="",
                timestamp=None,
                status=0,
                error=str(exc),
            )

