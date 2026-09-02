"""End-to-end FaceTrace pipeline orchestration interfaces and execution output."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from src.blockchain.base import AnchorResult, TransactionReceipt
from src.candidates.base import CandidateMatch
from src.evidence.models import Evidence, EvidenceRecord
from src.face.base import DetectedFace, FaceEmbedding
from src.search.base import SearchResult
from src.verification.base import VerificationResult

# Optional callback type for UI progress updates: (stage, message) -> None
ProgressCallback = Callable[[str, str], None] | None


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
    all_matches: list[CandidateMatch] = field(default_factory=list)
    best_match: CandidateMatch | None = None
    evidence: Evidence | None = None
    evidence_record: EvidenceRecord | None = None
    fingerprint: str | None = None
    anchor_result: AnchorResult | None = None
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
