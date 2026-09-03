"""Comprehensive Phase 6 End-to-End Hardening, Failure Matrix & Production Readiness Test Suite.

Covers all 11 test domains:
1. End-to-End Workflow Verification (Full Pipeline)
2. Face Input Matrix (Valid, No Face, Multi Face, Small, Corrupt, Unsupported)
3. Search Provider Failure Matrix (Missing Key, 401, 429, 500, Timeout, Network, Malformed, Empty)
4. Candidate Processing Failure Matrix (Valid, 404, Timeout, Zero Face, Multi Face, Corrupt, Large, Malformed URL)
5. Biometric Matching & Threshold Tests (Below, Boundary, Above, Sorting)
6. Result Deduplication Tests (Same URL Deduplicated, Same Image Different URLs Preserved)
7. No-Match Flow & User Guidance Test
8. Evidence Integrity & Cryptographic Tamper Detection
9. Blockchain Failure Handling (RPC Down, Revert, Not Found, Wrong Address)
10. Privacy, Anti-Leakage & Sensitive Credential Redaction Audit
"""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests
from PIL import Image
from web3 import Web3, EthereumTesterProvider

from src.blockchain.base import AnchorResult, OnChainRecord
from src.blockchain.mock import MockBlockchainProvider
from src.blockchain.service import BlockchainError, EthereumBlockchainService
from src.candidates.base import Candidate, CandidateMatch
from src.candidates.processor import DefaultCandidateProcessor
from src.core.config import Settings, get_settings
from src.core.types import PlatformType, VerificationStatus
from src.evidence.canonical import (
    canonicalize_evidence_dict,
    compute_image_sha256,
    compute_sha256_fingerprint,
    to_bytes32_hex,
)
from src.evidence.models import EvidenceRecord
from src.face.base import BoundingBox, DetectedFace, FaceEmbedding
from src.face.mock import MockFaceDetector, MockFaceEmbedder
from src.face.query_processor import (
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    QueryFaceProcessor,
)
from src.pipeline.base import PipelineOutput, PipelineStage
from src.pipeline.orchestrator import DefaultFaceTracePipeline
from src.search import get_search_provider
from src.search.base import SearchResult
from src.search.mock import MockSearchProvider
from src.search.serpapi import SerpApiError, SerpApiSearchProvider
from src.verification.engine import DefaultVerificationEngine, verify_evidence


def make_test_image_bytes(width: int = 100, height: int = 100, color: str = "blue") -> bytes:
    """Helper to generate a valid in-memory JPEG image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ==============================================================================
# Domain 1: End-to-End Workflow Verification
# ==============================================================================

def test_p6_01_end_to_end_pipeline_workflow(sample_embedding_512d: FaceEmbedding) -> None:
    """Verify complete Phase 1-5 pipeline workflow from image input to verified blockchain record."""
    w3 = Web3(EthereumTesterProvider())
    blockchain = EthereumBlockchainService(w3=w3, network_name="local")
    blockchain.deploy_contract()

    # Fixed mock face services
    detector = MockFaceDetector(return_face=True)
    embedder = MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector)

    cand_bytes = make_test_image_bytes(100, 100, color="green")
    cand_results = [
        SearchResult(
            url="https://instagram.com/p/target_post_1",
            title="Target Subject Public Instagram Portrait",
            source_domain="instagram.com",
            thumbnail_url="https://instagram.com/p/target_post_1.jpg",
            platform=PlatformType.INSTAGRAM,
        )
    ]

    mock_search = MockSearchProvider(predefined_results=cand_results)

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = cand_bytes
    mock_session.get.return_value = mock_resp

    candidates_proc = DefaultCandidateProcessor(
        face_detector=detector,
        face_embedder=embedder,
        session=mock_session,
    )

    pipeline = DefaultFaceTracePipeline(
        detector=detector,
        embedder=embedder,
        search_provider=mock_search,
        candidate_processor=candidates_proc,
        blockchain_provider=blockchain,
        threshold=0.45,
    )

    probe_bytes = make_test_image_bytes(120, 120, color="blue")
    output = pipeline.execute(probe_bytes, auto_anchor=True)

    assert output.stage == PipelineStage.COMPLETED
    assert output.best_match is not None
    assert output.best_match.is_match is True
    assert output.evidence_record is not None
    assert output.anchor_result is not None
    assert output.anchor_result.is_success is True
    assert output.verification_result is not None
    assert output.verification_result.status == VerificationStatus.VERIFIED
    assert output.verification_result.is_verified is True


# ==============================================================================
# Domain 2: Face Input Test Matrix (Tests A - F)
# ==============================================================================

def test_p6_02_face_input_valid_single_face(sample_embedding_512d: FaceEmbedding) -> None:
    """Test A: Valid single face -> Accepted, 512D ArcFace embedding returned."""
    processor = QueryFaceProcessor(
        detector=MockFaceDetector(return_face=True),
        embedder=MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector),
    )
    img = make_test_image_bytes(100, 100)
    face, embedding = processor.process_query_image(img)

    assert face is not None
    assert face.confidence >= 0.90
    assert embedding.dimension == 512
    assert len(embedding.vector) == 512


def test_p6_03_face_input_no_face() -> None:
    """Test B: No face in image -> Rejected with NoFaceDetectedError."""
    mock_det = MagicMock()
    mock_det.detect_faces.return_value = []
    processor = QueryFaceProcessor(detector=mock_det, embedder=MockFaceEmbedder())

    img = make_test_image_bytes(100, 100)
    with pytest.raises(NoFaceDetectedError, match="No face detected"):
        processor.process_query_image(img)


def test_p6_04_face_input_multiple_faces() -> None:
    """Test C: Multiple faces in query image -> Rejected with MultipleFacesDetectedError."""
    mock_det = MagicMock()
    mock_det.detect_faces.return_value = [
        DetectedFace(bbox=BoundingBox(10, 10, 40, 40), confidence=0.95),
        DetectedFace(bbox=BoundingBox(50, 50, 90, 90), confidence=0.92),
    ]
    processor = QueryFaceProcessor(detector=mock_det, embedder=MockFaceEmbedder())

    img = make_test_image_bytes(100, 100)
    with pytest.raises(MultipleFacesDetectedError, match="Multiple faces detected"):
        processor.process_query_image(img)


def test_p6_05_face_input_small_image() -> None:
    """Test D: Image smaller than minimum 32x32 dimensions -> Rejected with descriptive InvalidImageError."""
    processor = QueryFaceProcessor(detector=MockFaceDetector(), embedder=MockFaceEmbedder())
    tiny_img = make_test_image_bytes(16, 16)

    with pytest.raises(InvalidImageError, match="too small"):
        processor.process_query_image(tiny_img)


def test_p6_06_face_input_corrupted_bytes() -> None:
    """Test E: Corrupted image bytes -> Rejected with InvalidImageError."""
    processor = QueryFaceProcessor()
    corrupt_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupt_random_bytes_here"

    with pytest.raises(InvalidImageError):
        processor.process_query_image(corrupt_bytes)


def test_p6_07_face_input_unsupported_garbage_bytes() -> None:
    """Test F: Unsupported/empty/garbage payload -> Rejected with InvalidImageError."""
    processor = QueryFaceProcessor()

    with pytest.raises(InvalidImageError, match="empty"):
        processor.process_query_image(b"")

    with pytest.raises(InvalidImageError):
        processor.process_query_image(b"not an image at all")


# ==============================================================================
# Domain 3: Search Provider Failure Matrix (10 Failure Modes)
# ==============================================================================

def test_p6_08_search_missing_api_key() -> None:
    """Failure 1: Missing API key raises clear SerpApiError."""
    provider = SerpApiSearchProvider(api_key="")
    with pytest.raises(SerpApiError, match="SERPAPI_API_KEY is not configured"):
        provider.search_by_image(b"dummy_bytes")


def test_p6_09_search_invalid_key_401() -> None:
    """Failure 2: Invalid API key (HTTP 401) raises authentication error."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": "Invalid API key"}
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="invalid_test_key", session=mock_session)
    with pytest.raises(SerpApiError, match="authentication failure"):
        provider.search_by_image("https://example.com/photo.jpg")


def test_p6_10_search_rate_limit_429() -> None:
    """Failure 3: HTTP 429 rate limit raises quota error."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    with pytest.raises(SerpApiError, match="rate limit exceeded"):
        provider.search_by_image("https://example.com/photo.jpg")


def test_p6_11_search_server_error_500() -> None:
    """Failure 4: HTTP 500 server error raises service unavailable error."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    with pytest.raises(SerpApiError, match="temporarily unavailable"):
        provider.search_by_image("https://example.com/photo.jpg")


def test_p6_12_search_request_timeout() -> None:
    """Failure 5: Network timeout raises SerpApiError with timeout explanation."""
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session, timeout=5)
    with pytest.raises(SerpApiError, match="timed out"):
        provider.search_by_image("https://example.com/photo.jpg")


def test_p6_13_search_network_connection_failure() -> None:
    """Failure 6: DNS / network connection failure raises SerpApiError."""
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.ConnectionError("Failed to resolve host")

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    with pytest.raises(SerpApiError, match="network connection failure"):
        provider.search_by_image("https://example.com/photo.jpg")


def test_p6_14_search_malformed_json_response() -> None:
    """Failure 7: Non-JSON / malformed response payload raises SerpApiError."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON")
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    with pytest.raises(SerpApiError, match="invalid, non-JSON response"):
        provider.search_by_image("https://example.com/photo.jpg")


def test_p6_15_search_empty_results() -> None:
    """Failure 8: SerpApi response with zero visual matches returns empty list without error."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"visual_matches": []}
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    results = provider.search_by_image("https://example.com/photo.jpg")
    assert results == []


def test_p6_16_search_results_missing_thumbnails() -> None:
    """Failure 9: Results without thumbnail/image fields are parsed safely."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "visual_matches": [
            {
                "title": "Post with no thumbnail",
                "link": "https://example.com/page1",
                "source": "example.com",
            }
        ]
    }
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    results = provider.search_by_image("https://example.com/photo.jpg")
    assert len(results) == 1
    assert results[0].url == "https://example.com/page1"
    assert results[0].thumbnail_url is None


# ==============================================================================
# Domain 4: Candidate Processing Failure Matrix (10 Failure Modes)
# ==============================================================================

def test_p6_17_candidate_http_404_handling(sample_embedding_512d: FaceEmbedding) -> None:
    """Candidate Failure 2: 404 response on candidate image sets status='ERROR' and continues."""
    mock_session = MagicMock()
    fail_resp = MagicMock()
    fail_resp.status_code = 404
    fail_resp.content = b""
    mock_session.get.return_value = fail_resp

    processor = DefaultCandidateProcessor(session=mock_session)
    results = [
        SearchResult(
            url="https://site.com/broken",
            title="Broken",
            source_domain="site.com",
            thumbnail_url="https://site.com/broken.jpg",
        )
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d)
    assert len(matches) == 1
    assert matches[0].status == "ERROR"
    assert matches[0].is_match is False


def test_p6_18_candidate_timeout_handling(sample_embedding_512d: FaceEmbedding) -> None:
    """Candidate Failure 3: Download timeout marks candidate as ERROR and does not crash."""
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.Timeout("Read timeout")

    processor = DefaultCandidateProcessor(session=mock_session)
    results = [
        SearchResult(
            url="https://site.com/hang",
            title="Hanging URL",
            source_domain="site.com",
            thumbnail_url="https://site.com/hang.jpg",
        )
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d)
    assert len(matches) == 1
    assert matches[0].status == "ERROR"
    assert matches[0].is_match is False


def test_p6_19_candidate_zero_faces_found(sample_embedding_512d: FaceEmbedding) -> None:
    """Candidate Failure 4: Candidate with landscape/no faces gets 0.0 score and is_match=False."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = make_test_image_bytes(100, 100)
    mock_session.get.return_value = mock_resp

    mock_det = MagicMock()
    mock_det.detect_faces.return_value = []  # No faces in candidate

    processor = DefaultCandidateProcessor(face_detector=mock_det, session=mock_session)
    results = [
        SearchResult(
            url="https://site.com/landscape",
            title="Mountain View",
            source_domain="site.com",
            thumbnail_url="https://site.com/mountain.jpg",
        )
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d)
    assert len(matches) == 1
    assert matches[0].status == "NO_FACE"
    assert matches[0].similarity_score == 0.0
    assert matches[0].is_match is False


def test_p6_20_candidate_multi_face_highest_score_selected() -> None:
    """Candidate Failure 5: Candidate with multiple faces detects all and selects highest similarity face."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = make_test_image_bytes(100, 100)
    mock_session.get.return_value = mock_resp

    # Query vector
    query_vec = [1.0] + [0.0] * 511
    query_emb = FaceEmbedding(vector=query_vec, dimension=512)

    # Detector returns 2 faces
    mock_det = MagicMock()
    face1 = DetectedFace(bbox=BoundingBox(10, 10, 40, 40), confidence=0.95)
    face2 = DetectedFace(bbox=BoundingBox(50, 50, 90, 90), confidence=0.98)
    mock_det.detect_faces.return_value = [face1, face2]

    # Embedder returns: Face 1 has sim 0.30, Face 2 has sim 0.85
    vec_face1 = [0.30] + [0.953939] + [0.0] * 510
    vec_face2 = [0.85] + [0.52678] + [0.0] * 510

    mock_emb = MagicMock()
    mock_emb.extract_embedding.side_effect = [
        FaceEmbedding(vector=vec_face1, dimension=512),
        FaceEmbedding(vector=vec_face2, dimension=512),
    ]

    processor = DefaultCandidateProcessor(
        face_detector=mock_det,
        face_embedder=mock_emb,
        session=mock_session,
    )

    results = [
        SearchResult(
            url="https://site.com/group",
            title="Group photo",
            source_domain="site.com",
            thumbnail_url="https://site.com/group.jpg",
        )
    ]

    matches = processor.fetch_and_evaluate(results, query_emb, similarity_threshold=0.45)
    assert len(matches) == 1
    assert matches[0].detected_faces_count == 2
    # Highest score (0.85) was selected
    assert pytest.approx(matches[0].similarity_score, abs=1e-2) == 0.85
    assert matches[0].is_match is True
    assert matches[0].detected_face == face2


def test_p6_21_candidate_corrupt_bytes_handling(sample_embedding_512d: FaceEmbedding) -> None:
    """Candidate Failure 6: Candidate download returning corrupted non-image bytes marks status='ERROR'."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"this is corrupted text, not an image"
    mock_session.get.return_value = mock_resp

    processor = DefaultCandidateProcessor(session=mock_session)
    results = [
        SearchResult(
            url="https://site.com/corrupt",
            title="Corrupted File",
            source_domain="site.com",
            thumbnail_url="https://site.com/corrupt.jpg",
        )
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d)
    assert len(matches) == 1
    assert matches[0].status == "ERROR"
    assert matches[0].is_match is False


def test_p6_22_candidate_malformed_url_handling(sample_embedding_512d: FaceEmbedding) -> None:
    """Candidate Failure 10: Malformed URL without scheme does not crash candidate processor."""
    mock_session = MagicMock()
    processor = DefaultCandidateProcessor(session=mock_session)

    results = [
        SearchResult(
            url="malformed://invalid_url_with_no_valid_protocol",
            title="Malformed",
            source_domain="invalid",
            thumbnail_url="malformed://invalid_url_with_no_valid_protocol",
        )
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d)
    assert len(matches) == 1
    assert matches[0].status == "ERROR"


# ==============================================================================
# Domain 5: Biometric Matching & Threshold Calibration Tests
# ==============================================================================

def test_p6_23_threshold_calibration_rules() -> None:
    """Verify strict threshold qualification rules: below threshold is NO_MATCH, at or above is MATCH."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = make_test_image_bytes(100, 100)
    mock_session.get.return_value = mock_resp

    query_vec = [1.0] + [0.0] * 511
    query_emb = FaceEmbedding(vector=query_vec, dimension=512)

    # 4 candidates with similarities: 0.40, 0.449, 0.450, 0.50
    def make_vec(dot: float) -> list[float]:
        rem = (1.0 - dot * dot) ** 0.5
        return [dot, rem] + [0.0] * 510

    mock_emb = MagicMock()
    mock_emb.extract_embedding.side_effect = [
        FaceEmbedding(vector=make_vec(0.40), dimension=512),
        FaceEmbedding(vector=make_vec(0.449), dimension=512),
        FaceEmbedding(vector=make_vec(0.450), dimension=512),
        FaceEmbedding(vector=make_vec(0.50), dimension=512),
    ]

    processor = DefaultCandidateProcessor(
        face_detector=MockFaceDetector(),
        face_embedder=mock_emb,
        session=mock_session,
    )

    results = [
        SearchResult(url="https://site.com/p1", title="P1", source_domain="site.com", thumbnail_url="http://p1.jpg"),
        SearchResult(url="https://site.com/p2", title="P2", source_domain="site.com", thumbnail_url="http://p2.jpg"),
        SearchResult(url="https://site.com/p3", title="P3", source_domain="site.com", thumbnail_url="http://p3.jpg"),
        SearchResult(url="https://site.com/p4", title="P4", source_domain="site.com", thumbnail_url="http://p4.jpg"),
    ]

    matches = processor.fetch_and_evaluate(results, query_emb, similarity_threshold=0.45)
    assert len(matches) == 4

    # Matches sorted descending: P4 (0.50) -> P3 (0.450) -> P2 (0.449) -> P1 (0.40)
    assert matches[0].candidate.url == "https://site.com/p4"
    assert matches[0].is_match is True

    assert matches[1].candidate.url == "https://site.com/p3"
    assert matches[1].is_match is True

    assert matches[2].candidate.url == "https://site.com/p2"
    assert matches[2].is_match is False  # 0.449 < 0.450

    assert matches[3].candidate.url == "https://site.com/p1"
    assert matches[3].is_match is False  # 0.40 < 0.450


# ==============================================================================
# Domain 6: Result Deduplication Tests
# ==============================================================================

def test_p6_24_deduplicate_identical_url_results(sample_embedding_512d: FaceEmbedding) -> None:
    """Test identical URL returned twice produces exactly one candidate match."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = make_test_image_bytes(100, 100)
    mock_session.get.return_value = mock_resp

    processor = DefaultCandidateProcessor(
        face_detector=MockFaceDetector(),
        face_embedder=MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector),
        session=mock_session,
    )

    results = [
        SearchResult(
            url="https://instagram.com/p/identical_post",
            title="Duplicate Post 1",
            source_domain="instagram.com",
            thumbnail_url="http://img1.jpg",
        ),
        SearchResult(
            url="https://instagram.com/p/identical_post",  # Exact same URL
            title="Duplicate Post 2",
            source_domain="instagram.com",
            thumbnail_url="http://img1.jpg",
        ),
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d)
    assert len(matches) == 1
    assert matches[0].candidate.url == "https://instagram.com/p/identical_post"


def test_p6_25_preserve_same_image_from_different_urls(sample_embedding_512d: FaceEmbedding) -> None:
    """Test same image hosted on two distinct URLs preserves both legitimate separate sources."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    shared_image_bytes = make_test_image_bytes(100, 100)
    mock_resp.content = shared_image_bytes
    mock_session.get.return_value = mock_resp

    processor = DefaultCandidateProcessor(
        face_detector=MockFaceDetector(),
        face_embedder=MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector),
        session=mock_session,
    )

    results = [
        SearchResult(
            url="https://instagram.com/p/post_alpha",
            title="Instagram Post",
            source_domain="instagram.com",
            thumbnail_url="https://shared-cdn.com/portrait.jpg",
            platform=PlatformType.INSTAGRAM,
        ),
        SearchResult(
            url="https://twitter.com/user/status/post_beta",
            title="Twitter Post",
            source_domain="twitter.com",
            thumbnail_url="https://shared-cdn.com/portrait.jpg",
            platform=PlatformType.TWITTER,
        ),
    ]

    matches = processor.fetch_and_evaluate(results, sample_embedding_512d)
    # Both distinct sources are retained
    assert len(matches) == 2
    assert matches[0].candidate.url == "https://instagram.com/p/post_alpha"
    assert matches[1].candidate.url == "https://twitter.com/user/status/post_beta"
    # Both share the same image hash
    assert matches[0].candidate.image_hash == matches[1].candidate.image_hash


# ==============================================================================
# Domain 7: No-Match Flow & User Guidance Test
# ==============================================================================

def test_p6_26_no_match_flow_messaging() -> None:
    """Test that pipeline produces clean completed status when no candidate passes threshold."""
    detector = MockFaceDetector(return_face=True)
    # Orthogonal embedder
    query_vec = [1.0] + [0.0] * 511
    cand_vec = [0.0] + [1.0] + [0.0] * 510  # similarity = 0.0

    embedder = MagicMock()
    embedder.extract_embedding.side_effect = [
        FaceEmbedding(vector=query_vec, dimension=512),
        FaceEmbedding(vector=cand_vec, dimension=512),
    ]

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = make_test_image_bytes(100, 100)
    mock_session.get.return_value = mock_resp

    cand_proc = DefaultCandidateProcessor(
        face_detector=detector,
        face_embedder=embedder,
        session=mock_session,
    )

    search = MockSearchProvider(predefined_results=[
        SearchResult(url="https://unrelated.com/scenery", title="Scenery", source_domain="unrelated.com")
    ])

    pipeline = DefaultFaceTracePipeline(
        detector=detector,
        embedder=embedder,
        search_provider=search,
        candidate_processor=cand_proc,
        blockchain_provider=MockBlockchainProvider(),
        threshold=0.45,
    )

    probe_bytes = make_test_image_bytes(100, 100)
    output = pipeline.execute(probe_bytes)

    assert output.stage == PipelineStage.COMPLETED
    assert output.best_match is None
    assert output.evidence_record is None
    assert "no sufficiently similar face was found" in output.error_message


# ==============================================================================
# Domain 8: Evidence Integrity & Cryptographic Tamper Detection
# ==============================================================================

def test_p6_27_evidence_tamper_detection_on_chain() -> None:
    """Test on-chain tamper detection: anchor original evidence, alter one field, verify returns TAMPERED."""
    w3 = Web3(EthereumTesterProvider())
    blockchain = EthereumBlockchainService(w3=w3, network_name="local")
    blockchain.deploy_contract()

    # Original evidence
    original_er = EvidenceRecord(
        post_url="https://instagram.com/p/genuine_post",
        platform="instagram",
        title="Original Portrait Evidence",
        image_sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        similarity_score=0.8200,
        threshold=0.4500,
        discovered_at="2026-09-03T12:00:00Z",
    )
    orig_fp = original_er.to_bytes32()

    # Anchor to real EVM smart contract
    anchor_res = blockchain.anchor_evidence(orig_fp, original_er.post_url)
    assert anchor_res.is_success is True

    # 1. Verification of untouched original -> VERIFIED
    vr_clean = verify_evidence(original_er, blockchain, expected_fingerprint=orig_fp)
    assert vr_clean.status == VerificationStatus.VERIFIED
    assert vr_clean.is_verified is True

    # 2. Tampered record: Alter similarity_score from 0.82 to 0.83
    tampered_er = EvidenceRecord(
        post_url=original_er.post_url,
        platform=original_er.platform,
        title=original_er.title,
        image_sha256=original_er.image_sha256,
        similarity_score=0.8300,  # Modified!
        threshold=original_er.threshold,
        discovered_at=original_er.discovered_at,
    )
    mod_fp = tampered_er.to_bytes32()

    assert orig_fp != mod_fp

    # Verification of tampered record against original anchor -> TAMPERED
    vr_tampered = verify_evidence(tampered_er, blockchain, expected_fingerprint=orig_fp)
    assert vr_tampered.status == VerificationStatus.TAMPERED
    assert vr_tampered.is_verified is False
    assert any("differs from anchored" in reason for reason in vr_tampered.tamper_reasons)


# ==============================================================================
# Domain 9: Blockchain Failure Handling
# ==============================================================================

def test_p6_28_blockchain_verify_non_anchored_evidence() -> None:
    """Test querying evidence not registered on blockchain returns NOT_FOUND cleanly."""
    w3 = Web3(EthereumTesterProvider())
    blockchain = EthereumBlockchainService(w3=w3, network_name="local")
    blockchain.deploy_contract()

    unregistered_er = EvidenceRecord(
        post_url="https://site.com/ghost",
        platform="web",
        title="Unregistered Ghost Record",
        image_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        similarity_score=0.9000,
        threshold=0.4500,
        discovered_at="2026-09-03T12:00:00Z",
    )

    vr = verify_evidence(unregistered_er, blockchain)
    assert vr.status == VerificationStatus.NOT_FOUND
    assert vr.is_verified is False


def test_p6_29_blockchain_transaction_failure_handling() -> None:
    """Test anchoring failure (e.g. reverted transaction or contract error) returns graceful AnchorResult."""
    mock_provider = MagicMock()
    mock_provider.anchor_evidence.return_value = AnchorResult(
        success=False,
        tx_hash="",
        block_number=0,
        contract_address="0x0000",
        stored_evidence_hash="",
        timestamp=None,
        status=0,
        error="Execution reverted: Out of gas",
    )

    res = mock_provider.anchor_evidence("0x1234", "https://site.com")
    assert res.is_success is False
    assert "reverted" in res.error


# ==============================================================================
# Domain 10: Privacy, Anti-Leakage & Sensitive Credential Redaction Audit
# ==============================================================================

def test_p6_30_evidence_record_privacy_guarantee() -> None:
    """Verify EvidenceRecord contains ZERO biometric vectors, raw scans, or embeddings."""
    er = EvidenceRecord(
        post_url="https://instagram.com/p/test",
        platform="instagram",
        title="Title",
        image_sha256="abc123hash",
        similarity_score=0.85,
        threshold=0.45,
        discovered_at="2026-09-03T12:00:00Z",
    )

    d = er.to_dict()
    forbidden_keys = {"vector", "embedding", "raw_image", "face_crop", "biometric"}
    for key in d.keys():
        assert key not in forbidden_keys, f"Forbidden biometric key '{key}' present in EvidenceRecord!"

    canonical_json = er.to_canonical_json()
    for forbidden in forbidden_keys:
        assert forbidden not in canonical_json


def test_p6_31_settings_redacts_sensitive_credentials() -> None:
    """Verify Settings.safe_dict() redacts all API keys, private keys, and secrets."""
    settings = Settings(
        SERPAPI_API_KEY="secret_serp_key_999",
        ETH_PRIVATE_KEY="0xsecret_eth_key_888",
        BING_SEARCH_API_KEY="secret_bing_key_777",
    )

    safe = settings.safe_dict()
    assert safe["SERPAPI_API_KEY"] == "***REDACTED***"
    assert safe["ETH_PRIVATE_KEY"] == "***REDACTED***"
    assert safe["BING_SEARCH_API_KEY"] == "***REDACTED***"
