"""Default verification engine comparing canonical evidence against on-chain fingerprints."""

import logging
from src.blockchain.base import BlockchainProvider
from src.blockchain import get_blockchain_provider
from src.evidence.models import Evidence, EvidenceRecord
from src.verification.base import VerificationEngine, VerificationResult

logger = logging.getLogger(__name__)


class DefaultVerificationEngine(VerificationEngine):
    """Verifies evidence authenticity and temporal provenance against the blockchain registry."""

    def __init__(self, blockchain_provider: BlockchainProvider | None = None) -> None:
        self.blockchain_provider = blockchain_provider or get_blockchain_provider()

    def verify(
        self,
        evidence: Evidence | EvidenceRecord,
        expected_fingerprint: str | None = None,
    ) -> VerificationResult:
        """Re-compute canonical fingerprint from evidence and verify against on-chain records.

        Args:
            evidence: Evidence or EvidenceRecord instance to verify.
            expected_fingerprint: Optional expected on-chain fingerprint to compare against for tamper detection.

        Returns:
            VerificationResult with status: VERIFIED, TAMPERED, NOT_FOUND, or BLOCKCHAIN_ERROR.
        """
        # 1. Deterministically recompute canonical fingerprint
        computed_fp = evidence.to_bytes32()
        source_url = (
            getattr(evidence, "source_url", None)
            or getattr(evidence, "post_url", "")
        )
        version = getattr(evidence, "version", None) or getattr(evidence, "evidence_version", "1.0.0")

        # 2. Check if an expected anchored fingerprint was provided and mismatches
        if expected_fingerprint is not None:
            norm_expected = expected_fingerprint.lower()
            norm_computed = computed_fp.lower()
            if norm_expected != norm_computed:
                logger.warning(
                    "Evidence tampered: computed fingerprint %s != expected anchored %s",
                    norm_computed,
                    norm_expected,
                )
                # Try to fetch original on-chain record for context
                try:
                    on_chain_orig = self.blockchain_provider.get_evidence(expected_fingerprint)
                except Exception:
                    on_chain_orig = None

                return VerificationResult.tampered(
                    fingerprint=computed_fp,
                    on_chain_record=on_chain_orig,
                    source_url=source_url,
                    reasons=[
                        f"Evidence fingerprint mismatch: computed '{computed_fp}' "
                        f"differs from anchored '{expected_fingerprint}'."
                    ],
                    evidence_version=version,
                )

        logger.info(
            "Verifying evidence fingerprint %s against blockchain (%s)...",
            computed_fp,
            self.blockchain_provider.network_name,
        )

        # 3. Query blockchain registry (zero gas view call)
        try:
            on_chain_record = self.blockchain_provider.get_evidence(computed_fp)
        except Exception as exc:
            logger.error("Blockchain query failed during verification: %s", exc)
            return VerificationResult.blockchain_error(
                fingerprint=computed_fp,
                source_url=source_url,
                error_message=str(exc),
                evidence_version=version,
            )

        # 4. Check existence
        if not on_chain_record.exists:
            logger.warning("Evidence fingerprint %s was NOT found on-chain.", computed_fp)
            return VerificationResult.not_found(
                fingerprint=computed_fp,
                source_url=source_url,
                evidence_version=version,
            )

        # 5. Check on-chain metadata consistency
        tamper_reasons: list[str] = []
        if on_chain_record.source_url and source_url and on_chain_record.source_url != source_url:
            tamper_reasons.append(
                f"Source URL mismatch: on-chain record has '{on_chain_record.source_url}', "
                f"evidence has '{source_url}'."
            )

        if tamper_reasons:
            logger.warning("Evidence tampered: %s", tamper_reasons)
            return VerificationResult.tampered(
                fingerprint=computed_fp,
                on_chain_record=on_chain_record,
                source_url=source_url,
                reasons=tamper_reasons,
                evidence_version=version,
            )

        logger.info(
            "Evidence successfully VERIFIED on-chain! (Recorded at block #%d by %s)",
            on_chain_record.block_number,
            on_chain_record.recorder,
        )
        return VerificationResult.success(
            fingerprint=computed_fp,
            on_chain_record=on_chain_record,
            source_url=source_url,
            evidence_version=version,
        )


def verify_evidence(
    evidence: Evidence | EvidenceRecord,
    blockchain_provider: BlockchainProvider | None = None,
    expected_fingerprint: str | None = None,
) -> VerificationResult:
    """Convenience function verifying an evidence record against the blockchain registry.

    Args:
        evidence: EvidenceRecord or Evidence instance to verify.
        blockchain_provider: Optional BlockchainProvider instance (defaults to configured provider).
        expected_fingerprint: Optional expected on-chain fingerprint for direct tamper detection.

    Returns:
        Structured VerificationResult (VERIFIED, TAMPERED, NOT_FOUND, or BLOCKCHAIN_ERROR).
    """
    engine = DefaultVerificationEngine(blockchain_provider=blockchain_provider)
    return engine.verify(evidence, expected_fingerprint=expected_fingerprint)

