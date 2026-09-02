"""Candidate retrieval, feature extraction, similarity scoring, and ranking interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib

from src.core.types import PlatformType
from src.face.base import DetectedFace, FaceEmbedding
from src.search.base import SearchResult


@dataclass
class Candidate:
    """A fetched candidate image with metadata retrieved from a search result."""

    url: str
    image_url: str
    image_bytes: bytes
    image_hash: str
    title: str
    platform: PlatformType = PlatformType.UNKNOWN

    @classmethod
    def create(
        cls,
        url: str,
        image_url: str,
        image_bytes: bytes,
        title: str,
        platform: PlatformType = PlatformType.UNKNOWN,
    ) -> "Candidate":
        """Factory method computing SHA-256 hash of candidate image bytes."""
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        return cls(
            url=url,
            image_url=image_url,
            image_bytes=image_bytes,
            image_hash=image_hash,
            title=title,
            platform=platform,
        )


@dataclass
class CandidateMatch:
    """Evaluation result of comparing a candidate face with the query face embedding."""

    candidate: Candidate
    similarity_score: float
    detected_face: DetectedFace | None = None
    is_match: bool = False
    detected_faces_count: int = 0
    status: str = "PROCESSED"  # MATCH, NO_MATCH, NO_FACE_DETECTED, ERROR
    threshold_used: float = 0.45
    error_message: str | None = None
    local_image_path: str | None = None

    @property
    def formatted_score(self) -> float:
        """Similarity score rounded to 4 decimal places for deterministic precision."""
        return round(self.similarity_score, 4)


class CandidateProcessor(ABC):
    """Abstract interface for downloading, matching, and ranking search candidates."""

    @abstractmethod
    def fetch_and_evaluate(
        self,
        search_results: list[SearchResult],
        query_embedding: FaceEmbedding,
        similarity_threshold: float = 0.45,
    ) -> list[CandidateMatch]:
        """Download candidate images, detect faces, compute similarities, and return ranked matches."""
        pass

    @abstractmethod
    def select_best_match(
        self,
        matches: list[CandidateMatch],
    ) -> CandidateMatch | None:
        """Select the highest-confidence match meeting or exceeding the threshold."""
        pass
