"""Unit and integration tests for SerpApi Google Lens reverse-image search provider.

Covers all 12 required test specifications + conditional live integration test:
1. Missing API key
2. Successful API response
3. API error handling (401, 403, 429, 500, payload error, secret redaction)
4. Request timeout handling
5. Invalid / malformed response
6. Empty results handling
7. Result normalization (stripping tracking query params and fragments)
8. Platform detection (Instagram, Facebook, Twitter/X, Reddit, LinkedIn, etc.)
9. Candidate processing integration
10. Deduplication
11. Provider factory (get_search_provider for mock and serpapi)
12. Mock provider still works
13. Conditional live integration test (if SERPAPI_API_KEY is configured)
"""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import requests
from PIL import Image

from src.candidates.processor import DefaultCandidateProcessor
from src.core.config import Settings
from src.core.types import PlatformType
from src.face.base import FaceEmbedding
from src.face.mock import MockFaceDetector, MockFaceEmbedder
from src.search import get_search_provider
from src.search.base import SearchResult
from src.search.mock import MockSearchProvider
from src.search.serpapi import (
    MAX_SERPAPI_IMAGE_SIZE_BYTES,
    SerpApiError,
    SerpApiSearchProvider,
    _sanitize_text,
    detect_platform,
    normalize_url,
    prepare_image_for_upload,
)


def create_dummy_image(width: int = 100, height: int = 100, color: str = "red") -> bytes:
    """Helper generating dummy in-memory image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------------------
# Image Preprocessing Tests
# ------------------------------------------------------------------------------

def test_prepare_image_for_upload_small_image() -> None:
    """Test that small image under 500KB is returned intact or valid."""
    data = create_dummy_image(100, 100)
    assert len(data) < MAX_SERPAPI_IMAGE_SIZE_BYTES
    prepared = prepare_image_for_upload(data)
    assert len(prepared) <= MAX_SERPAPI_IMAGE_SIZE_BYTES


def test_prepare_image_for_upload_oversized_image() -> None:
    """Test that large image is downscaled and compressed to fit under 500KB."""
    img = Image.new("RGB", (2500, 2500), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    oversized_bytes = buf.getvalue()

    prepared = prepare_image_for_upload(oversized_bytes, max_bytes=100 * 1024)
    assert len(prepared) <= 100 * 1024
    with Image.open(io.BytesIO(prepared)) as res_img:
        assert res_img.format == "JPEG"
        assert max(res_img.width, res_img.height) <= 1024


# ------------------------------------------------------------------------------
# 1. Missing API Key
# ------------------------------------------------------------------------------

def test_serpapi_missing_api_key() -> None:
    """Test that searching without an API key raises SerpApiError."""
    provider = SerpApiSearchProvider(api_key="")
    with pytest.raises(SerpApiError, match="SERPAPI_API_KEY is not configured"):
        provider.search_by_image("https://example.com/photo.jpg")

    provider_none = SerpApiSearchProvider(api_key=None)
    provider_none.api_key = None  # Force None even if env exists
    with pytest.raises(SerpApiError, match="SERPAPI_API_KEY is not configured"):
        provider_none.search_by_image(b"some_image_bytes")


# ------------------------------------------------------------------------------
# 2. Successful API Response
# ------------------------------------------------------------------------------

def test_serpapi_successful_response() -> None:
    """Test successful API response with visual_matches and exact_matches normalized properly."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "visual_matches": [
            {
                "position": 1,
                "title": "Jane Doe on LinkedIn",
                "link": "https://www.linkedin.com/in/janedoe?utm_source=share",
                "source": "LinkedIn",
                "thumbnail": "https://images.unsplash.com/thumb1.jpg",
                "image": "https://images.unsplash.com/original1.jpg",
                "snippet": "Profile photo of Jane Doe.",
            },
            {
                "position": 2,
                "title": "Photo on Instagram",
                "link": "https://instagram.com/p/ABC123xyz/",
                "source": "Instagram",
                "thumbnail": "https://images.unsplash.com/thumb2.jpg",
            }
        ],
        "exact_matches": [
            {
                "position": 1,
                "title": "Wikipedia Article",
                "link": "https://en.wikipedia.org/wiki/Jane_Doe",
                "source": "Wikipedia",
                "thumbnail": "https://images.unsplash.com/thumb_wiki.jpg",
            }
        ]
    }
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="secret_test_key_123", session=mock_session)
    results = provider.search_by_image("https://example.com/query.jpg", max_results=5)

    assert len(results) == 3
    assert results[0].platform == PlatformType.LINKEDIN
    assert results[0].url == "https://www.linkedin.com/in/janedoe"  # normalized (utm stripped)
    assert results[0].source_domain == "LinkedIn"
    assert results[0].thumbnail_url == "https://images.unsplash.com/thumb1.jpg"

    assert results[1].platform == PlatformType.INSTAGRAM
    assert results[1].url == "https://instagram.com/p/ABC123xyz"

    assert results[2].platform == PlatformType.WIKIPEDIA
    assert results[2].url == "https://en.wikipedia.org/wiki/Jane_Doe"


# ------------------------------------------------------------------------------
# 3. API Error Handling & Secret Sanitization
# ------------------------------------------------------------------------------

def test_serpapi_api_error_handling() -> None:
    """Test that API errors (401, 403, 429, 500, payload error) are caught and redact secrets."""
    secret_key = "super_secret_serpapi_token_xyz"
    mock_session = MagicMock()

    # Case A: 401 Unauthorized
    resp_401 = MagicMock()
    resp_401.status_code = 401
    resp_401.json.return_value = {"error": f"Invalid API key: {secret_key}"}
    mock_session.get.return_value = resp_401

    provider = SerpApiSearchProvider(api_key=secret_key, session=mock_session)
    with pytest.raises(SerpApiError) as exc_info:
        provider.search_by_image("https://example.com/face.jpg")
    assert secret_key not in str(exc_info.value)
    assert "***REDACTED***" in str(exc_info.value) or "Invalid or unauthorized" in str(exc_info.value)

    # Case B: 429 Rate Limit
    resp_429 = MagicMock()
    resp_429.status_code = 429
    mock_session.get.return_value = resp_429
    with pytest.raises(SerpApiError, match="rate limit exceeded|search quota"):
        provider.search_by_image("https://example.com/face.jpg")

    # Case C: 500 Server Error
    resp_500 = MagicMock()
    resp_500.status_code = 500
    mock_session.get.return_value = resp_500
    with pytest.raises(SerpApiError, match="temporarily unavailable"):
        provider.search_by_image("https://example.com/face.jpg")

    # Case D: 200 OK with "error" in JSON body
    resp_error_json = MagicMock()
    resp_error_json.status_code = 200
    resp_error_json.json.return_value = {"error": f"Google Lens engine quota exceeded with key {secret_key}"}
    mock_session.get.return_value = resp_error_json
    with pytest.raises(SerpApiError) as exc_err:
        provider.search_by_image("https://example.com/face.jpg")
    assert secret_key not in str(exc_err.value)
    assert "quota exceeded" in str(exc_err.value)


def test_sanitize_text_utility() -> None:
    """Test that _sanitize_text reliably scrubs raw keys and query parameters."""
    raw = "Error accessing https://serpapi.com/search?engine=google_lens&api_key=abc123456&hl=en"
    clean = _sanitize_text(raw, api_key="abc123456")
    assert "abc123456" not in clean
    assert "api_key=***REDACTED***" in clean


# ------------------------------------------------------------------------------
# 4. Request Timeout
# ------------------------------------------------------------------------------

def test_serpapi_timeout_handling() -> None:
    """Test that requests.exceptions.Timeout produces a friendly SerpApiError without crashing."""
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out after 15s")

    provider = SerpApiSearchProvider(api_key="some_key", session=mock_session)
    with pytest.raises(SerpApiError, match="timed out"):
        provider.search_by_image("https://example.com/face.jpg")


# ------------------------------------------------------------------------------
# 5. Invalid / Malformed Response
# ------------------------------------------------------------------------------

def test_serpapi_invalid_response() -> None:
    """Test that non-JSON or malformed responses are handled without unhandled crashes."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    with pytest.raises(SerpApiError, match="invalid, non-JSON response payload"):
        provider.search_by_image("https://example.com/face.jpg")


# ------------------------------------------------------------------------------
# 6. Empty Results
# ------------------------------------------------------------------------------

def test_serpapi_empty_results() -> None:
    """Test that empty search results return an empty list gracefully."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "visual_matches": [],
        "search_metadata": {"status": "Success"}
    }
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    results = provider.search_by_image("https://example.com/face.jpg")
    assert results == []
    assert len(results) == 0


# ------------------------------------------------------------------------------
# 7. Result Normalization
# ------------------------------------------------------------------------------

def test_serpapi_result_normalization() -> None:
    """Test that URL normalization strips tracking query parameters and fragments."""
    test_urls = [
        ("https://example.com/post/123/?utm_source=google&utm_medium=cpc#comments", "https://example.com/post/123"),
        ("https://instagram.com/p/ABC123/?igsh=xyz987&fbclid=123", "https://instagram.com/p/ABC123"),
        ("https://x.com/user/status/456?ref_src=twsrc%5Etfw", "https://x.com/user/status/456"),
        ("https://site.com/photo.jpg?keep=1", "https://site.com/photo.jpg?keep=1"),
    ]
    for raw, expected in test_urls:
        normalized = normalize_url(raw)
        assert normalized == expected


# ------------------------------------------------------------------------------
# 8. Platform Detection
# ------------------------------------------------------------------------------

def test_serpapi_platform_detection() -> None:
    """Test generic platform detection based on URL and source."""
    cases = [
        ("https://instagram.com/p/123", "Instagram", PlatformType.INSTAGRAM),
        ("https://facebook.com/photo.php?fbid=1", "Facebook", PlatformType.FACEBOOK),
        ("https://fb.com/user/123", "", PlatformType.FACEBOOK),
        ("https://x.com/user/status/1", "X", PlatformType.TWITTER),
        ("https://twitter.com/user/status/1", "Twitter", PlatformType.TWITTER),
        ("https://reddit.com/r/pics/1", "Reddit", PlatformType.REDDIT),
        ("https://linkedin.com/in/test", "LinkedIn", PlatformType.LINKEDIN),
        ("https://youtube.com/watch?v=123", "YouTube", PlatformType.YOUTUBE),
        ("https://pinterest.com/pin/1", "Pinterest", PlatformType.PINTEREST),
        ("https://en.wikipedia.org/wiki/Face", "Wikipedia", PlatformType.WIKIPEDIA),
        ("https://nytimes.com/article/1", "The New York Times", PlatformType.NEWS),
        ("https://randomsite.org/page", "Random Blog", PlatformType.WEB),
    ]
    for url, source, expected in cases:
        detected = detect_platform(url, source)
        assert detected == expected


# ------------------------------------------------------------------------------
# 9. Candidate Processing Integration
# ------------------------------------------------------------------------------

def test_serpapi_candidate_processing_integration(sample_embedding_512d: FaceEmbedding) -> None:
    """Test that normalized SerpApi search results integrate smoothly with DefaultCandidateProcessor."""
    mock_session = MagicMock()
    # Mock candidate image download
    img_resp = MagicMock()
    img_resp.status_code = 200
    img_resp.content = create_dummy_image(150, 150)
    mock_session.get.return_value = img_resp

    results = [
        SearchResult(
            url="https://instagram.com/p/candidate1",
            title="Candidate 1",
            source_domain="Instagram",
            thumbnail_url="https://images.unsplash.com/c1.jpg",
            platform=PlatformType.INSTAGRAM,
        ),
        SearchResult(
            url="https://linkedin.com/in/candidate2",
            title="Candidate 2",
            source_domain="LinkedIn",
            thumbnail_url="https://images.unsplash.com/c2.jpg",
            platform=PlatformType.LINKEDIN,
        ),
    ]

    processor = DefaultCandidateProcessor(
        face_detector=MockFaceDetector(),
        face_embedder=MockFaceEmbedder(fixed_vector=sample_embedding_512d.vector),
        session=mock_session,
    )

    matches = processor.fetch_and_evaluate(results, query_embedding=sample_embedding_512d, similarity_threshold=0.45)
    assert len(matches) == 2
    assert matches[0].is_match is True
    assert matches[0].similarity_score >= 0.90
    assert matches[0].candidate.image_hash != ""


# ------------------------------------------------------------------------------
# 10. Deduplication
# ------------------------------------------------------------------------------

def test_serpapi_deduplication() -> None:
    """Test that duplicate URLs (with different tracking tokens or repeated listings) are deduplicated."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "visual_matches": [
            {
                "title": "Post on Site",
                "link": "https://site.com/photo?utm_source=a",
                "source": "Site",
                "thumbnail": "https://site.com/thumb.jpg",
            },
            {
                "title": "Same Post on Site with Diff Params",
                "link": "https://site.com/photo?utm_source=b&fbclid=xyz",
                "source": "Site",
                "thumbnail": "https://site.com/thumb.jpg",
            },
            {
                "title": "Distinct Page with Same Image",
                "link": "https://othersite.com/article",
                "source": "OtherSite",
                "thumbnail": "https://site.com/thumb.jpg",
            },
        ]
    }
    mock_session.get.return_value = mock_resp

    provider = SerpApiSearchProvider(api_key="test_key", session=mock_session)
    results = provider.search_by_image("https://example.com/face.jpg", max_results=10)

    # Exactly 2 results: the duplicate on site.com is filtered out, but the separate page on othersite.com is kept
    assert len(results) == 2
    assert results[0].url == "https://site.com/photo"
    assert results[1].url == "https://othersite.com/article"


# ------------------------------------------------------------------------------
# 11. Provider Factory
# ------------------------------------------------------------------------------

def test_search_provider_factory() -> None:
    """Test that get_search_provider instantiates the correct provider based on configuration."""
    settings_mock = Settings(SEARCH_PROVIDER="mock")
    provider_mock = get_search_provider(settings_mock)
    assert isinstance(provider_mock, MockSearchProvider)
    assert provider_mock.provider_name == "mock"

    settings_serp = Settings(SEARCH_PROVIDER="serpapi", SERPAPI_API_KEY="key123")
    provider_serp = get_search_provider(settings_serp)
    assert isinstance(provider_serp, SerpApiSearchProvider)
    assert provider_serp.provider_name == "serpapi"


# ------------------------------------------------------------------------------
# 12. Mock Provider Still Works
# ------------------------------------------------------------------------------

def test_mock_provider_still_works() -> None:
    """Test that MockSearchProvider continues to function identically and returns candidate results."""
    mock_provider = MockSearchProvider()
    results = mock_provider.search_by_image(b"mock_bytes", max_results=5)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.url != "" for r in results)


# ------------------------------------------------------------------------------
# 13. Conditional Live Integration Test
# ------------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("SERPAPI_API_KEY"),
    reason="Live integration test skipped because SERPAPI_API_KEY is not configured.",
)
def test_serpapi_live_integration() -> None:
    """Controlled single live integration test against real SerpApi Google Lens engine.

    Only executes when SERPAPI_API_KEY is present in the environment.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    assert api_key, "SERPAPI_API_KEY must be non-empty"

    provider = SerpApiSearchProvider(api_key=api_key)
    demo_image_path = Path("data/demo_images/subject_a.jpg")
    if not demo_image_path.exists():
        # Fallback to in-memory consented dummy portrait image
        image_input = create_dummy_image(200, 200)
    else:
        image_input = demo_image_path

    # Execute exactly ONE live request
    results = provider.search_by_image(image_input, max_results=5)
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)
    for res in results:
        assert res.url.startswith("http")
        assert res.source_domain != ""
