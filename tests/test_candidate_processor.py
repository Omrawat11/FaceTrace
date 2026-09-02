"""Unit tests for DefaultCandidateProcessor matching and ranking."""

import io
import pytest
from unittest.mock import MagicMock
from PIL import Image

from src.candidates.base import CandidateMatch
from src.candidates.processor import DefaultCandidateProcessor
from src.core.types import PlatformType
from src.face.base import BoundingBox, DetectedFace, FaceEmbedding
from src.face.mock import MockFaceDetector, MockFaceEmbedder
from src.search.base import SearchResult


def create_sample_image_bytes() -> bytes:
    """Create a simple valid image buffer."""
    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dummy_image_bytes() -> bytes:
    return create_sample_image_bytes()


def test_candidate_processor_successful_match(dummy_image_bytes: bytes, sample_embedding_512d: FaceEmbedding) -> None:
    """Test candidate download, face detection, embedding matching above threshold."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = dummy_image_bytes
    mock_session.get.return_value = mock_resp

    # Embedder returns exact same vector as query embedding -> similarity 1.0
    detector = MockFaceDetector()
    embedder = MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector)

    processor = DefaultCandidateProcessor(
        face_detector=detector,
        face_embedder=embedder,
        session=mock_session,
    )

    search_results = [
        SearchResult(
            url="https://instagram.com/p/test_post_1",
            title="Public Keynote Portrait",
            source_domain="instagram.com",
            thumbnail_url="https://images.example.com/photo1.jpg",
            platform=PlatformType.INSTAGRAM,
        )
    ]

    matches = processor.fetch_and_evaluate(
        search_results=search_results,
        query_embedding=sample_embedding_512d,
        similarity_threshold=0.45,
    )

    assert len(matches) == 1
    match = matches[0]
    assert match.is_match is True
    assert pytest.approx(match.similarity_score, rel=1e-4) == 1.0
    assert match.candidate.platform == PlatformType.INSTAGRAM
    assert len(match.candidate.image_hash) == 64  # SHA-256 hex length

    best = processor.select_best_match(matches)
    assert best is not None
    assert best.candidate.url == "https://instagram.com/p/test_post_1"


def test_candidate_processor_below_threshold(dummy_image_bytes: bytes, sample_embedding_512d: FaceEmbedding) -> None:
    """Test candidate with low cosine similarity is marked is_match=False."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = dummy_image_bytes
    mock_session.get.return_value = mock_resp

    # Embedder returns orthogonal vector -> similarity 0.0
    orthogonal_vec = [0.0] * 512
    orthogonal_vec[0] = 1.0  # while sample_embedding has 1/sqrt(512) for all, let's create orthogonal
    # To be strictly orthogonal to uniform vector, sum must be 0:
    # 256 ones, 256 negative ones:
    orthogonal_vec = [1.0] * 256 + [-1.0] * 256

    detector = MockFaceDetector()
    embedder = MockFaceEmbedder(fixed_vector=orthogonal_vec)

    processor = DefaultCandidateProcessor(
        face_detector=detector,
        face_embedder=embedder,
        session=mock_session,
    )

    search_results = [
        SearchResult(
            url="https://x.com/someone/status/123",
            title="Unrelated post",
            source_domain="x.com",
            thumbnail_url="https://images.example.com/unrelated.jpg",
            platform=PlatformType.TWITTER,
        )
    ]

    matches = processor.fetch_and_evaluate(
        search_results=search_results,
        query_embedding=sample_embedding_512d,
        similarity_threshold=0.45,
    )

    assert len(matches) == 1
    assert matches[0].is_match is False
    assert pytest.approx(matches[0].similarity_score, abs=1e-5) == 0.0

    best = processor.select_best_match(matches)
    assert best is None


def test_candidate_processor_no_face_detected(dummy_image_bytes: bytes, sample_embedding_512d: FaceEmbedding) -> None:
    """Test candidate where detector finds no face assigns 0.0 similarity and is_match=False."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = dummy_image_bytes
    mock_session.get.return_value = mock_resp

    detector = MagicMock()
    detector.detect_faces.return_value = []  # No faces found

    processor = DefaultCandidateProcessor(
        face_detector=detector,
        session=mock_session,
    )

    search_results = [
        SearchResult(
            url="https://example.com/scenery.jpg",
            title="Landscape scenery",
            source_domain="example.com",
            thumbnail_url="https://images.example.com/scenery.jpg",
        )
    ]

    matches = processor.fetch_and_evaluate(
        search_results=search_results,
        query_embedding=sample_embedding_512d,
        similarity_threshold=0.45,
    )

    assert len(matches) == 1
    assert matches[0].similarity_score == 0.0
    assert matches[0].is_match is False
    assert matches[0].detected_face is None


def test_candidate_processor_ranking_order(dummy_image_bytes: bytes) -> None:
    """Test that multiple candidates are returned in descending order of similarity score."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = dummy_image_bytes
    mock_session.get.return_value = mock_resp

    # Query vector
    query_vec = [1.0] + [0.0] * 511
    query_embedding = FaceEmbedding(vector=query_vec, dimension=512)

    # Embedder returning different vectors per call:
    # 1st call -> sim = 0.90
    # 2nd call -> sim = 0.95
    # 3rd call -> sim = 0.40
    vec_90 = [0.90] + [0.43588989] + [0.0] * 510  # norm = 1.0, dot = 0.90
    vec_95 = [0.95] + [0.3122499] + [0.0] * 510   # norm = 1.0, dot = 0.95
    vec_40 = [0.40] + [0.916515] + [0.0] * 510    # norm = 1.0, dot = 0.40

    embedder = MagicMock()
    embedder.extract_embedding.side_effect = [
        FaceEmbedding(vector=vec_90, dimension=512),
        FaceEmbedding(vector=vec_95, dimension=512),
        FaceEmbedding(vector=vec_40, dimension=512),
    ]

    processor = DefaultCandidateProcessor(
        face_detector=MockFaceDetector(),
        face_embedder=embedder,
        session=mock_session,
    )

    results = [
        SearchResult(url="https://site.com/1", title="Result 1", source_domain="site.com", thumbnail_url="http://img/1.jpg"),
        SearchResult(url="https://site.com/2", title="Result 2", source_domain="site.com", thumbnail_url="http://img/2.jpg"),
        SearchResult(url="https://site.com/3", title="Result 3", source_domain="site.com", thumbnail_url="http://img/3.jpg"),
    ]

    matches = processor.fetch_and_evaluate(results, query_embedding, similarity_threshold=0.45)

    assert len(matches) == 3
    # Sorted descending: Result 2 (0.95) -> Result 1 (0.90) -> Result 3 (0.40)
    assert matches[0].candidate.url == "https://site.com/2"
    assert pytest.approx(matches[0].similarity_score, abs=1e-2) == 0.95
    assert matches[0].is_match is True

    assert matches[1].candidate.url == "https://site.com/1"
    assert pytest.approx(matches[1].similarity_score, abs=1e-2) == 0.90
    assert matches[1].is_match is True

    assert matches[2].candidate.url == "https://site.com/3"
    assert pytest.approx(matches[2].similarity_score, abs=1e-2) == 0.40
    assert matches[2].is_match is False

    best = processor.select_best_match(matches)
    assert best is not None
    assert best.candidate.url == "https://site.com/2"


def test_candidate_processor_network_failure_handling(sample_embedding_512d: FaceEmbedding) -> None:
    """Test that network errors downloading a candidate image are caught and handled gracefully."""
    mock_session = MagicMock()
    # First image returns 404, second succeeds
    fail_resp = MagicMock()
    fail_resp.status_code = 404
    fail_resp.content = b""

    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.content = create_sample_image_bytes()

    mock_session.get.side_effect = [fail_resp, success_resp]

    processor = DefaultCandidateProcessor(
        face_detector=MockFaceDetector(),
        face_embedder=MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector),
        session=mock_session,
    )

    results = [
        SearchResult(url="https://site.com/broken", title="Broken Link", source_domain="site.com", thumbnail_url="http://img/broken.jpg"),
        SearchResult(url="https://site.com/good", title="Good Link", source_domain="site.com", thumbnail_url="http://img/good.jpg"),
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d, similarity_threshold=0.45)

    assert len(matches) == 2
    # Successful match is ranked first
    assert matches[0].candidate.url == "https://site.com/good"
    assert matches[0].is_match is True
    # Failed download candidate is preserved with status ERROR
    assert matches[1].candidate.url == "https://site.com/broken"
    assert matches[1].is_match is False
    assert matches[1].status == "ERROR"
    assert matches[1].error_message is not None

    best = processor.select_best_match(matches)
    assert best is not None
    assert best.candidate.url == "https://site.com/good"
