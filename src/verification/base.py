"""Evidence integrity verification interfaces and result models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.blockchain.base import OnChainRecord
from src.core.types import VerificationStatus
from src.evidence.models import Evidence


@dataclass
class VerificationResult:
    """Detailed verification outcome comparing local evidence against the blockchain."""

    is_verified: bool
    status: VerificationStatus
    computed_fingerprint: str
    on_chain_record: OnChainRecord | None = None
    evidence_version: str = "1.0.0"
    source_url: str = ""
    tamper_reasons: list[str] = field(default_factory=list)
    verification_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    @classmethod
    def success(
        cls,
        fingerprint: str,
        on_chain_record: OnChainRecord,
        source_url: str,
        evidence_version: str = "1.0.0",
    ) -> "VerificationResult":
        """Factory for a successful verification match."""
        return cls(
            is_verified=True,
            status=VerificationStatus.VERIFIED,
            computed_fingerprint=fingerprint,
            on_chain_record=on_chain_record,
            source_url=source_url,
            evidence_version=evidence_version,
            tamper_reasons=[],
        )

    @classmethod
    def not_found(
        cls,
        fingerprint: str,
        source_url: str,
        evidence_version: str = "1.0.0",
    ) -> "VerificationResult":
        """Factory when no record exists on-chain for the computed fingerprint."""
        return cls(
            is_verified=False,
            status=VerificationStatus.NOT_FOUND,
            computed_fingerprint=fingerprint,
            on_chain_record=None,
            source_url=source_url,
            evidence_version=evidence_version,
            tamper_reasons=["Fingerprint has not been anchored to the blockchain registry."],
        )

    @classmethod
    def tampered(
        cls,
        fingerprint: str,
        on_chain_record: OnChainRecord | None,
        source_url: str,
        reasons: list[str],
        evidence_version: str = "1.0.0",
    ) -> "VerificationResult":
        """Factory when evidence content mismatches on-chain record."""
        return cls(
            is_verified=False,
            status=VerificationStatus.TAMPERED,
            computed_fingerprint=fingerprint,
            on_chain_record=on_chain_record,
            source_url=source_url,
            evidence_version=evidence_version,
            tamper_reasons=reasons,
        )

    @classmethod
    def blockchain_error(
        cls,
        fingerprint: str,
        source_url: str,
        error_message: str,
        evidence_version: str = "1.0.0",
    ) -> "VerificationResult":
        """Factory when blockchain RPC connection or smart contract interaction fails."""
        return cls(
            is_verified=False,
            status=VerificationStatus.BLOCKCHAIN_ERROR,
            computed_fingerprint=fingerprint,
            on_chain_record=None,
            source_url=source_url,
            evidence_version=evidence_version,
            tamper_reasons=[f"Blockchain communication error: {error_message}"],
        )


class VerificationEngine(ABC):
    """Abstract interface for verifying current evidence integrity against the blockchain."""

    @abstractmethod
    def verify(self, evidence: Evidence) -> VerificationResult:
        """Re-compute evidence fingerprint and verify against the on-chain registry."""
        pass
