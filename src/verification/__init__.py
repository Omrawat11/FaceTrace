"""Verification module providing tamper detection and blockchain verification."""

from src.verification.base import VerificationEngine, VerificationResult
from src.verification.engine import DefaultVerificationEngine, verify_evidence

__all__ = [
    "VerificationEngine",
    "VerificationResult",
    "DefaultVerificationEngine",
    "verify_evidence",
]
