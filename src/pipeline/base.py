"""End-to-end FaceTrace pipeline orchestration interfaces and execution output."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.blockchain.base import TransactionReceipt
from src.candidates.base import CandidateMatch
from src.evidence.models import Evidence
from src.face.base import DetectedFace, FaceEmbedding
from src.search.base import SearchResult
from src.verification.base import VerificationResult


class PipelineStage(str, Enum):
    """Discrete sequential stages in the FaceTrace pipeline."""

    IDLE = "IDLE"
    FACE_DETECTION = "FACE_DETECTION"
    FACE_ENCODING = "FACE_ENCODING"
    SEARCHING = "SEARCHING"
    CANDIDATE_EVALUATION = "CANDIDATE_EVALUATION"
    MATCHING = "MATCHING"
    EVIDENCE_EXTRACTION = "EVIDENCE_EXTRACTION"
    FINGERPRINTING = "FINGERPRINTING"
    BLOCKCHAIN_RECORDING = "BLOCKCHAIN_RECORDING"
    VERIFICATION = "VERIFICATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class PipelineOutput:
    """Comprehensive artifact bundle produced by an end-to-end FaceTrace execution."""

    query_image_path: Path
    stage: PipelineStage
    query_face: DetectedFace | None = None
    query_embedding: FaceEmbedding | None = None
    search_results: list[SearchResult] | None = None
    best_match: CandidateMatch | None = None
    evidence: Evidence | None = None
    fingerprint: str | None = None
    transaction_receipt: TransactionReceipt | None = None
    verification_result: VerificationResult | None = None
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        return (
            self.stage == PipelineStage.COMPLETED
            and self.verification_result is not None
            and self.verification_result.is_verified
        )


class FaceTracePipeline(ABC):
    """Abstract orchestrator for the complete FaceTrace biometric provenance pipeline."""

    @abstractmethod
    def run(self, query_image: bytes | Path | str) -> PipelineOutput:
        """Execute the full FaceTrace pipeline:

        Input image
        -> Face detection
        -> Face embedding
        -> Genuine web/social search
        -> Candidate retrieval & evaluation
        -> Face similarity ranking & best match
        -> Evidence extraction
        -> Canonicalization & SHA-256 fingerprint
        -> Blockchain recording (Sepolia)
        -> Re-verification against blockchain
        """
        pass
