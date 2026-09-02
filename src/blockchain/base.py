"""Abstract interfaces and data structures for Ethereum testnet (Sepolia) interactions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class OnChainRecord:
    """Represents an evidence record retrieved from the on-chain EvidenceRegistry contract."""

    exists: bool
    fingerprint: str
    recorder: str
    timestamp: int
    block_number: int
    source_url: str

    @property
    def formatted_time(self) -> str:
        """Format UNIX block timestamp to UTC ISO 8601 string."""
        if self.timestamp == 0:
            return "N/A"
        dt = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%SZ")


@dataclass
class TransactionReceipt:
    """Receipt summary of an on-chain transaction mining confirmation."""

    tx_hash: str
    block_number: int
    block_hash: str
    gas_used: int
    status: int  # 1 for success, 0 for revert

    @property
    def is_success(self) -> bool:
        return self.status == 1


@dataclass
class AnchorResult:
    """Complete summary of an evidence anchoring operation on the blockchain."""

    success: bool
    tx_hash: str
    block_number: int
    contract_address: str
    stored_evidence_hash: str
    timestamp: int | None
    status: int  # 1 for success, 0 for failure
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.success and self.status == 1


class BlockchainProvider(ABC):
    """Abstract interface for anchoring and querying evidence fingerprints on-chain."""

    @property
    @abstractmethod
    def network_name(self) -> str:
        """Name of the connected network (e.g. 'sepolia', 'mock')."""
        pass

    @abstractmethod
    def record_fingerprint(
        self,
        fingerprint: str,
        source_url: str,
    ) -> TransactionReceipt:
        """Submit a transaction recording the 32-byte evidence fingerprint.

        Args:
            fingerprint: 0x-prefixed 32-byte hex string.
            source_url: Source URL where matching evidence was found.

        Returns:
            TransactionReceipt with block confirmation details.
        """
        pass

    @abstractmethod
    def get_evidence(self, fingerprint: str) -> OnChainRecord:
        """Query the smart contract for an existing fingerprint record (view call, zero gas)."""
        pass

    @abstractmethod
    def has_evidence(self, fingerprint: str) -> bool:
        """Check whether a fingerprint exists on-chain."""
        pass

    @abstractmethod
    def anchor_evidence(
        self,
        evidence_hash: str,
        source_url: str = "",
    ) -> AnchorResult:
        """High-level evidence anchoring flow: submits transaction, waits for receipt,
        verifies confirmed mining status, and retrieves on-chain stored hash.

        Args:
            evidence_hash: 64-char hex or 0x-prefixed 32-byte evidence SHA-256 fingerprint.
            source_url: Source URL where matching evidence was discovered.

        Returns:
            AnchorResult with transaction status, mined block, contract address,
            and retrieved on-chain hash.
        """
        pass
