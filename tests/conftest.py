"""Shared pytest fixtures and test data."""

import pytest

from src.candidates.base import Candidate, CandidateMatch
from src.core.config import Settings
from src.core.types import PlatformType
from src.evidence.models import Evidence
from src.face.base import BoundingBox, DetectedFace, FaceEmbedding


@pytest.fixture
def sample_settings() -> Settings:
    """Fixture returning clean settings instance."""
    return Settings(
        ENVIRONMENT="testing",
        FACE_SIMILARITY_THRESHOLD=0.45,
        SEARCH_PROVIDER="mock",
        BLOCKCHAIN_PROVIDER="mock",
    )


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """Fixture returning a sample face bounding box."""
    return BoundingBox(x1=50, y1=60, x2=200, y2=250)


@pytest.fixture
def sample_detected_face(sample_bbox: BoundingBox) -> DetectedFace:
    """Fixture returning a sample detected face."""
    return DetectedFace(
        bbox=sample_bbox,
        confidence=0.985,
        landmarks=[(100.0, 110.0), (150.0, 110.0), (125.0, 140.0)],
    )


@pytest.fixture
def sample_embedding_512d() -> FaceEmbedding:
    """Fixture returning a mock 512D unit-length embedding vector."""
    # Simple normalized vector
    val = 1.0 / (512**0.5)
    return FaceEmbedding(vector=[val] * 512, dimension=512, model_name="arcface")


@pytest.fixture
def sample_candidate() -> Candidate:
    """Fixture returning a valid candidate."""
    return Candidate.create(
        url="https://instagram.com/p/sample_post_123",
        image_url="https://instagram.com/media/sample_post_123.jpg",
        image_bytes=b"fake-jpeg-image-bytes-for-candidate-testing",
        title="Sample Public Post Photo",
        platform=PlatformType.INSTAGRAM,
    )


@pytest.fixture
def sample_candidate_match(
    sample_candidate: Candidate, sample_detected_face: DetectedFace
) -> CandidateMatch:
    """Fixture returning a candidate match above threshold."""
    return CandidateMatch(
        candidate=sample_candidate,
        similarity_score=0.8724,
        detected_face=sample_detected_face,
        is_match=True,
    )


@pytest.fixture
def sample_evidence(sample_candidate_match: CandidateMatch) -> Evidence:
    """Fixture returning a deterministic Evidence instance."""
    return Evidence(
        version="1.0.0",
        source_url="https://instagram.com/p/sample_post_123",
        platform="instagram",
        title="Sample Public Post Photo",
        caption="A public photo from social media",
        candidate_image_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        similarity_score=0.8724,
        discovery_timestamp="2026-09-01T18:00:00Z",
        face_box=[50, 60, 200, 250],
    )
