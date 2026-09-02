"""Comprehensive test suite for Phase 2B Local Search and Biometric Face Matching Pipeline."""

import io
from pathlib import Path
import pytest
from PIL import Image

from src.candidates.base import Candidate, CandidateMatch
from src.candidates.processor import DefaultCandidateProcessor
from src.core.types import PlatformType
from src.face.base import BoundingBox, DetectedFace, FaceEmbedding
from src.face.insightface_engine import (
    InsightFaceDetector,
    InsightFaceEmbedder,
    load_bgr_image,
)
from src.face.mock import MockFaceDetector, MockFaceEmbedder
from src.face.query_processor import (
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    QueryFaceProcessor,
)
from src.search.base import SearchProvider, SearchResult
from src.search.mock import MockSearchProvider


def make_dummy_jpeg(color: str = "white", width: int = 100, height: int = 100) -> bytes:
    """Helper creating valid in-memory JPEG bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------------------
# 1. Query Face Processing Tests
# ------------------------------------------------------------------------------

def test_query_face_success(sample_embedding_512d: FaceEmbedding) -> None:
    """Test query face processing with exactly one detected face."""
    detector = MockFaceDetector()
    embedder = MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector)
    processor = QueryFaceProcessor(detector=detector, embedder=embedder)

    img_bytes = make_dummy_jpeg()
    detected_face, embedding = processor.process_query_image(img_bytes)

    assert detected_face is not None
    assert detected_face.confidence >= 0.90
    assert embedding.dimension == 512
    assert len(embedding.vector) == 512


def test_query_face_no_face_detected() -> None:
    """Test query face processing when zero faces are found."""
    detector = MockFaceDetector()
    detector.detect_faces = lambda img: []  # No faces
    embedder = MockFaceEmbedder()
    processor = QueryFaceProcessor(detector=detector, embedder=embedder)

    img_bytes = make_dummy_jpeg()
    with pytest.raises(NoFaceDetectedError, match="No face detected in query image"):
        processor.process_query_image(img_bytes)


def test_query_face_multiple_faces_detected() -> None:
    """Test query face processing when multiple faces are found."""
    detector = MockFaceDetector()
    detector.detect_faces = lambda img: [
        DetectedFace(bbox=BoundingBox(10, 10, 50, 50), confidence=0.95),
        DetectedFace(bbox=BoundingBox(60, 60, 100, 100), confidence=0.92),
    ]
    embedder = MockFaceEmbedder()
    processor = QueryFaceProcessor(detector=detector, embedder=embedder)

    img_bytes = make_dummy_jpeg()
    with pytest.raises(MultipleFacesDetectedError, match="Multiple faces detected"):
        processor.process_query_image(img_bytes)


def test_query_face_invalid_image() -> None:
    """Test query face processing with corrupted/non-image bytes."""
    processor = QueryFaceProcessor()
    with pytest.raises(InvalidImageError):
        processor.process_query_image(b"this is not a valid image payload")


def test_query_face_empty_payload() -> None:
    """Test query face processing with 0-byte payload."""
    processor = QueryFaceProcessor()
    with pytest.raises(InvalidImageError, match="empty"):
        processor.process_query_image(b"")


# ------------------------------------------------------------------------------
# 2. Candidate Processing & Match / Non-Match Tests
# ------------------------------------------------------------------------------

def test_candidate_matching_detected(sample_embedding_512d: FaceEmbedding, tmp_path: Path) -> None:
    """Test candidate matching with high similarity (is_match=True)."""
    cand_img = tmp_path / "cand_match.jpg"
    cand_img.write_bytes(make_dummy_jpeg())

    # Same vector -> similarity = 1.0
    detector = MockFaceDetector()
    embedder = MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector)
    processor = DefaultCandidateProcessor(face_detector=detector, face_embedder=embedder)

    results = [
        SearchResult(
            url="https://instagram.com/p/match1",
            title="Matching Post",
            source_domain="instagram.com",
            thumbnail_url=str(cand_img),
            platform=PlatformType.INSTAGRAM,
        )
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d, similarity_threshold=0.45)

    assert len(matches) == 1
    m = matches[0]
    assert m.is_match is True
    assert m.status == "MATCH"
    assert pytest.approx(m.similarity_score, rel=1e-3) == 1.0
    assert m.detected_faces_count == 1
    assert m.detected_face is not None


def test_candidate_non_matching_rejected(sample_embedding_512d: FaceEmbedding, tmp_path: Path) -> None:
    """Test candidate with low similarity is rejected (is_match=False)."""
    cand_img = tmp_path / "cand_non_match.jpg"
    cand_img.write_bytes(make_dummy_jpeg())

    # Orthogonal vector -> similarity = 0.0
    ortho_vec = [1.0] * 256 + [-1.0] * 256
    detector = MockFaceDetector()
    embedder = MockFaceEmbedder(fixed_vector=ortho_vec)
    processor = DefaultCandidateProcessor(face_detector=detector, face_embedder=embedder)

    results = [
        SearchResult(
            url="https://x.com/post/unrelated",
            title="Unrelated Post",
            source_domain="x.com",
            thumbnail_url=str(cand_img),
            platform=PlatformType.TWITTER,
        )
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d, similarity_threshold=0.45)

    assert len(matches) == 1
    m = matches[0]
    assert m.is_match is False
    assert m.status == "NO_MATCH"
    assert pytest.approx(m.similarity_score, abs=1e-4) == 0.0

    best = processor.select_best_match(matches)
    assert best is None


def test_candidate_no_face_detected(sample_embedding_512d: FaceEmbedding, tmp_path: Path) -> None:
    """Test candidate with zero faces is marked NO_FACE_DETECTED."""
    cand_img = tmp_path / "landscape.jpg"
    cand_img.write_bytes(make_dummy_jpeg())

    detector = MockFaceDetector()
    detector.detect_faces = lambda img: []  # No faces in candidate
    embedder = MockFaceEmbedder()
    processor = DefaultCandidateProcessor(face_detector=detector, face_embedder=embedder)

    results = [
        SearchResult(
            url="https://pinterest.com/pin/landscape",
            title="Landscape Pin",
            source_domain="pinterest.com",
            thumbnail_url=str(cand_img),
            platform=PlatformType.PINTEREST,
        )
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d, similarity_threshold=0.45)

    assert len(matches) == 1
    m = matches[0]
    assert m.is_match is False
    assert m.status == "NO_FACE_DETECTED"
    assert m.similarity_score == 0.0
    assert m.detected_faces_count == 0
    assert m.detected_face is None


def test_candidate_multiple_faces_selects_highest_similarity(tmp_path: Path) -> None:
    """Test candidate containing multiple faces compares against all and picks the maximum similarity."""
    cand_img = tmp_path / "multi_face.jpg"
    cand_img.write_bytes(make_dummy_jpeg())

    query_vec = [1.0] + [0.0] * 511
    query_emb = FaceEmbedding(vector=query_vec, dimension=512)

    # Face 1: similarity = 0.20
    # Face 2: similarity = 0.88 (Target person)
    # Face 3: similarity = 0.05
    vec_face1 = [0.20] + [0.97979589] + [0.0] * 510
    vec_face2 = [0.88] + [0.47497368] + [0.0] * 510
    vec_face3 = [0.05] + [0.99874921] + [0.0] * 510

    face1 = DetectedFace(bbox=BoundingBox(10, 10, 40, 40), confidence=0.85)
    face2 = DetectedFace(bbox=BoundingBox(50, 50, 90, 90), confidence=0.92)
    face3 = DetectedFace(bbox=BoundingBox(100, 100, 140, 140), confidence=0.78)

    detector = MockFaceDetector()
    detector.detect_faces = lambda img: [face1, face2, face3]

    embedder = MockFaceEmbedder()
    def mock_extract(img, bbox=None):
        if bbox == face1.bbox:
            return FaceEmbedding(vector=vec_face1, dimension=512)
        elif bbox == face2.bbox:
            return FaceEmbedding(vector=vec_face2, dimension=512)
        else:
            return FaceEmbedding(vector=vec_face3, dimension=512)
    embedder.extract_embedding = mock_extract

    processor = DefaultCandidateProcessor(face_detector=detector, face_embedder=embedder)

    results = [
        SearchResult(
            url="https://x.com/group_photo",
            title="Team Photo",
            source_domain="x.com",
            thumbnail_url=str(cand_img),
            platform=PlatformType.TWITTER,
        )
    ]

    matches = processor.fetch_and_evaluate(results, query_emb, similarity_threshold=0.45)

    assert len(matches) == 1
    m = matches[0]
    assert m.is_match is True
    assert m.status == "MATCH"
    assert pytest.approx(m.similarity_score, abs=1e-3) == 0.88
    assert m.detected_faces_count == 3
    assert m.detected_face == face2  # Picked face 2 with highest score


def test_candidate_ranking_order(tmp_path: Path) -> None:
    """Test candidate list is strictly sorted in descending order of similarity score."""
    p1 = tmp_path / "c1.jpg"
    p2 = tmp_path / "c2.jpg"
    p3 = tmp_path / "c3.jpg"
    for p in (p1, p2, p3):
        p.write_bytes(make_dummy_jpeg())

    query_vec = [1.0] + [0.0] * 511
    query_emb = FaceEmbedding(vector=query_vec, dimension=512)

    vec_70 = [0.70] + [0.71414284] + [0.0] * 510
    vec_92 = [0.92] + [0.39191835] + [0.0] * 510
    vec_35 = [0.35] + [0.93674969] + [0.0] * 510

    embedder = MockFaceEmbedder()
    embedder.extract_embedding = lambda img, bbox=None: (
        FaceEmbedding(vector=vec_70, dimension=512)
        if "c1.jpg" in str(getattr(img, "name", "")) or True
        else FaceEmbedding(vector=vec_70, dimension=512)
    )

    # Use side_effect on extract_embedding
    embedder.extract_embedding = lambda img, bbox=None: next(iter_embeddings)
    iter_embeddings = iter([
        FaceEmbedding(vector=vec_70, dimension=512),
        FaceEmbedding(vector=vec_92, dimension=512),
        FaceEmbedding(vector=vec_35, dimension=512),
    ])

    processor = DefaultCandidateProcessor(
        face_detector=MockFaceDetector(),
        face_embedder=embedder,
    )

    results = [
        SearchResult(url="https://site.com/1", title="Result 1 (0.70)", source_domain="site.com", thumbnail_url=str(p1)),
        SearchResult(url="https://site.com/2", title="Result 2 (0.92)", source_domain="site.com", thumbnail_url=str(p2)),
        SearchResult(url="https://site.com/3", title="Result 3 (0.35)", source_domain="site.com", thumbnail_url=str(p3)),
    ]

    matches = processor.fetch_and_evaluate(results, query_emb, similarity_threshold=0.45)

    assert len(matches) == 3
    # Ordered descending: Result 2 (0.92) -> Result 1 (0.70) -> Result 3 (0.35)
    assert matches[0].candidate.url == "https://site.com/2"
    assert pytest.approx(matches[0].similarity_score, abs=1e-2) == 0.92
    assert matches[0].is_match is True

    assert matches[1].candidate.url == "https://site.com/1"
    assert pytest.approx(matches[1].similarity_score, abs=1e-2) == 0.70
    assert matches[1].is_match is True

    assert matches[2].candidate.url == "https://site.com/3"
    assert pytest.approx(matches[2].similarity_score, abs=1e-2) == 0.35
    assert matches[2].is_match is False

    best = processor.select_best_match(matches)
    assert best is not None
    assert best.candidate.url == "https://site.com/2"


def test_configurable_threshold_behavior(tmp_path: Path) -> None:
    """Test that matches adhere strictly to the supplied similarity threshold."""
    img_path = tmp_path / "c.jpg"
    img_path.write_bytes(make_dummy_jpeg())

    # Similarity = 0.50
    vec_50 = [0.50] + [0.8660254] + [0.0] * 510
    query_vec = [1.0] + [0.0] * 511

    query_emb = FaceEmbedding(vector=query_vec, dimension=512)
    embedder = MockFaceEmbedder(fixed_vector=vec_50)
    processor = DefaultCandidateProcessor(
        face_detector=MockFaceDetector(),
        face_embedder=embedder,
    )

    results = [
        SearchResult(url="https://site.com/c", title="Candidate", source_domain="site.com", thumbnail_url=str(img_path))
    ]

    # Test with threshold 0.45 -> Should match
    m_match = processor.fetch_and_evaluate(results, query_emb, similarity_threshold=0.45)
    assert m_match[0].is_match is True
    assert m_match[0].status == "MATCH"

    # Test with threshold 0.55 -> Should NOT match
    m_no_match = processor.fetch_and_evaluate(results, query_emb, similarity_threshold=0.55)
    assert m_no_match[0].is_match is False
    assert m_no_match[0].status == "NO_MATCH"


def test_candidate_missing_image_resilience() -> None:
    """Test candidate with missing image file records ERROR without crashing."""
    processor = DefaultCandidateProcessor(face_detector=MockFaceDetector(), face_embedder=MockFaceEmbedder())
    emb = FaceEmbedding(vector=[1.0] + [0.0] * 511, dimension=512)

    results = [
        SearchResult(
            url="https://site.com/missing",
            title="Missing File",
            source_domain="site.com",
            thumbnail_url="non_existent_file_path_12345.jpg",
        )
    ]

    matches = processor.fetch_and_evaluate(results, emb)
    assert len(matches) == 1
    assert matches[0].status == "ERROR"
    assert matches[0].is_match is False
    assert matches[0].error_message is not None


def test_empty_search_results(sample_embedding_512d: FaceEmbedding) -> None:
    """Test candidate evaluation with empty search results."""
    processor = DefaultCandidateProcessor(face_detector=MockFaceDetector(), face_embedder=MockFaceEmbedder())
    matches = processor.fetch_and_evaluate([], sample_embedding_512d)
    assert matches == []
    assert processor.select_best_match(matches) is None


def test_sha256_candidate_hashing() -> None:
    """Test that candidate image hashing generates a deterministic 64-char hex SHA-256."""
    img_data = b"deterministic_candidate_image_content_test"
    cand = Candidate.create(
        url="https://example.com/test",
        image_url="https://example.com/test.jpg",
        image_bytes=img_data,
        title="Test Candidate",
    )
    assert len(cand.image_hash) == 64
    import hashlib
    expected_hash = hashlib.sha256(img_data).hexdigest()
    assert cand.image_hash == expected_hash


def test_mock_search_provider_conforms_to_search_provider() -> None:
    """Test that MockSearchProvider satisfies the SearchProvider abstract base interface."""
    provider = MockSearchProvider()
    assert isinstance(provider, SearchProvider)
    assert provider.provider_name == "mock"

    results = provider.search_by_image("data/demo_images/user_primary.jpg", max_results=5)
    assert isinstance(results, list)
    assert len(results) <= 5
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.url.startswith("https://") or r.url.startswith("demo://")
        assert r.source_domain != ""


# ------------------------------------------------------------------------------
# 3. Real Biometric Engine Tests (InsightFace SCRFD + ArcFace)
# ------------------------------------------------------------------------------

def test_insightface_real_detection_and_embedding() -> None:
    """Test real InsightFace detector and embedder on controlled demo images."""
    demo_path = Path("data/demo_images/user_primary.jpg")
    if not demo_path.exists():
        pytest.skip("Demo image user_primary.jpg not found.")

    detector = InsightFaceDetector(confidence_threshold=0.50)
    embedder = InsightFaceEmbedder()

    # 1. Test detection
    faces = detector.detect_faces(demo_path)
    assert len(faces) == 1
    assert faces[0].confidence >= 0.70
    assert faces[0].bbox.width > 50
    assert faces[0].bbox.height > 50

    # 2. Test 512D ArcFace embedding extraction
    emb = embedder.extract_embedding(demo_path, bbox=faces[0].bbox)
    assert emb.dimension == 512
    assert len(emb.vector) == 512

    # Vector should be unit-normalized: norm ~= 1.0
    norm = sum(v * v for v in emb.vector) ** 0.5
    assert pytest.approx(norm, rel=1e-4) == 1.0


def test_insightface_real_end_to_end_match_vs_non_match() -> None:
    """Test real InsightFace comparing user_primary against match and non-match demo images."""
    query_path = Path("data/demo_images/user_primary.jpg")
    match_path = Path("data/demo_images/user_match_01.jpg")
    non_match_path = Path("data/demo_images/non_match_01.jpg")
    no_face_path = Path("data/demo_images/no_face_01.jpg")

    for p in (query_path, match_path, non_match_path, no_face_path):
        if not p.exists():
            pytest.skip(f"Required demo file {p} not found.")

    query_processor = QueryFaceProcessor(
        detector=InsightFaceDetector(),
        embedder=InsightFaceEmbedder(),
    )
    detected_face, query_emb = query_processor.process_query_image(query_path)
    assert detected_face is not None
    assert query_emb.dimension == 512

    candidate_processor = DefaultCandidateProcessor(
        face_detector=InsightFaceDetector(),
        face_embedder=InsightFaceEmbedder(),
    )

    results = [
        SearchResult(url="demo://match", title="Match Candidate", source_domain="demo", thumbnail_url=str(match_path)),
        SearchResult(url="demo://non_match", title="Non-Match Candidate", source_domain="demo", thumbnail_url=str(non_match_path)),
        SearchResult(url="demo://no_face", title="No-Face Candidate", source_domain="demo", thumbnail_url=str(no_face_path)),
    ]

    matches = candidate_processor.fetch_and_evaluate(results, query_emb, similarity_threshold=0.45)

    assert len(matches) == 3
    # Top ranked match should be user_match_01 with high similarity (~1.00)
    top_match = matches[0]
    assert top_match.candidate.url == "demo://match"
    assert top_match.is_match is True
    assert top_match.status == "MATCH"
    assert top_match.similarity_score > 0.90

    # Non-match should have similarity < 0.45
    non_match_eval = next(m for m in matches if m.candidate.url == "demo://non_match")
    assert non_match_eval.is_match is False
    assert non_match_eval.status == "NO_MATCH"
    assert non_match_eval.similarity_score < 0.30

    # No-face should have status NO_FACE_DETECTED and similarity 0.0
    no_face_eval = next(m for m in matches if m.candidate.url == "demo://no_face")
    assert no_face_eval.status == "NO_FACE_DETECTED"
    assert no_face_eval.similarity_score == 0.0
    assert no_face_eval.is_match is False

    best = candidate_processor.select_best_match(matches)
    assert best is not None
    assert best.candidate.url == "demo://match"
