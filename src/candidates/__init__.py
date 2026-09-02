"""Candidate processing, downloading, face extraction, and ranking module."""

from src.candidates.base import (
    Candidate,
    CandidateMatch,
    CandidateProcessor,
)
from src.candidates.processor import DefaultCandidateProcessor

__all__ = [
    "Candidate",
    "CandidateMatch",
    "CandidateProcessor",
    "DefaultCandidateProcessor",
]
