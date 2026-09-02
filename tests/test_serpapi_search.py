"""Unit and integration tests for SerpApi Google Lens reverse-image search provider."""

import io
import pytest
from unittest.mock import MagicMock
from PIL import Image

from src.core.config import Settings
from src.core.types import PlatformType
from src.search.base import SearchResult
from src.search.mock import MockSearchProvider
from src.search.serpapi import (
    MAX_SERPAPI_IMAGE_SIZE_BYTES,
    SerpApiError,
    SerpApiSearchProvider,
    prepare_image_for_upload,
)
from src.search import get_search_provider


def create_dummy_image(width: int = 100, height: int = 100, color: str = "red") -> bytes:
    """Helper generating dummy in-memory image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_prepare_image_for_upload_small_image() -> None:
    """Test that small image under 500KB is returned intact or valid."""
    data = create_dummy_image(100, 100)
    assert len(data) < MAX_SERPAPI_IMAGE_SIZE_BYTES
    prepared = prepare_image_for_upload(data)
    assert len(prepared) <= MAX_SERPAPI_IMAGE_SIZE_BYTES


def test_prepare_image_for_upload_oversized_image() -> None:
    """Test that large image is downscaled and compressed to fit under 500KB."""
    # Create large image with noise to create high byte size
    img = Image.new("RGB", (2500, 2500), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # Uncompressed PNG
    oversized_bytes = buf.getvalue()

    prepared = prepare_image_for_upload(oversized_bytes, max_bytes=100 * 1024)
    assert len(prepared) <= 100 * 1024
    with Image.open(io.BytesIO(prepared)) as res_img:
        assert res_img.format == "JPEG"
        assert max(res_img.width, res_img.height) <= 1024


def test_serpapi_missing_api_key() -> None:
    """Test that searching without an API key raises SerpApiError."""
    provider = SerpApiSearchProvider(api_key="")
    with pytest.raises(SerpApiError, match="SERPAPI_API_KEY is not configured"):
        provider.search_by_image("https://example.com/photo.jpg")


def test_serpapi_search_by_url() -> None:
    """Test searching with an existing image URL calls Google Lens directly."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "visual_matches": [
            {
                "position": 1,
                "title": "Jane Doe on LinkedIn",
                "link": "https://www.linkedin.com/in/janedoe",
                "source": "LinkedIn",
                "thumbnail": "https://images.unsplash.com/thumb1.jpg",
                "image": "https://images.unsplash.com/original1.jpg",
            }
        ]
    }
    mock_session.get.return_value = mock_response

    provider = SerpApiSearchProvider(api_key="test_key_123", session=mock_session)
    results = provider.search_by_image("https://example.com/test_face.jpg", max_results=5)

    assert len(results) == 1
    assert results[0].url == "https://www.linkedin.com/in/janedoe"
    assert results[0].platform == PlatformType.LINKEDIN
    assert results[0].thumbnail_url == "https://images.unsplash.com/thumb1.jpg"

    # Verify session.get was called with proper params
    mock_session.get.assert_called_once()
    call_args, call_kwargs = mock_session.get.call_args
    assert call_kwargs["params"]["url"] == "https://example.com/test_face.jpg"
    assert call_kwargs["params"]["engine"] == "google_lens"
    assert call_kwargs["params"]["api_key"] == "test_key_123"


def test_serpapi_search_by_image_bytes(tmp_path) -> None:
    """Test searching with raw image bytes triggers /image upload first, then /search."""
    image_bytes = create_dummy_image(200, 200)

    mock_session = MagicMock()

    # Mock POST /image
    mock_upload_resp = MagicMock()
    mock_upload_resp.status_code = 200
    mock_upload_resp.json.return_value = {"image_id": "mock_image_id_456"}
    mock_session.post.return_value = mock_upload_resp

    # Mock GET /search
    mock_search_resp = MagicMock()
    mock_search_resp.status_code = 200
    mock_search_resp.json.return_value = {
        "visual_matches": [
            {
                "position": 1,
                "title": "Photo on Instagram",
                "link": "https://instagram.com/p/ABC123xyz/",
                "source": "Instagram",
                "thumbnail": "https://example.com/ig_thumb.jpg",
            },
            {
                "position": 2,
                "title": "Tweet from user",
                "link": "https://x.com/user/status/987654",
                "source": "X",
                "thumbnail": "https://example.com/x_thumb.jpg",
            },
        ]
    }
    mock_session.get.return_value = mock_search_resp

    provider = SerpApiSearchProvider(api_key="valid_key", session=mock_session)
    results = provider.search_by_image(image_bytes, max_results=2)

    assert len(results) == 2
    assert results[0].platform == PlatformType.INSTAGRAM
    assert results[1].platform == PlatformType.TWITTER

    # Assert upload was called
    mock_session.post.assert_called_once()
    _, post_kwargs = mock_session.post.call_args
    assert post_kwargs["data"]["api_key"] == "valid_key"

    # Assert search was called with image_id
    mock_session.get.assert_called_once()
    _, get_kwargs = mock_session.get.call_args
    assert get_kwargs["params"]["image_id"] == "mock_image_id_456"
    assert get_kwargs["params"]["engine"] == "google_lens"


def test_serpapi_api_error_handling() -> None:
    """Test that API error payload raises SerpApiError."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"error": "Invalid API key provided."}
    mock_session.get.return_value = mock_response

    provider = SerpApiSearchProvider(api_key="bad_key", session=mock_session)
    with pytest.raises(SerpApiError, match="Invalid API key provided"):
        provider.search_by_image("https://example.com/face.jpg")


def test_search_provider_factory() -> None:
    """Test that get_search_provider instantiates the correct provider from settings."""
    settings_mock = Settings(SEARCH_PROVIDER="mock")
    provider_mock = get_search_provider(settings_mock)
    assert isinstance(provider_mock, MockSearchProvider)
    assert provider_mock.provider_name == "mock"

    settings_serp = Settings(SEARCH_PROVIDER="serpapi", SERPAPI_API_KEY="key123")
    provider_serp = get_search_provider(settings_serp)
    assert isinstance(provider_serp, SerpApiSearchProvider)
    assert provider_serp.provider_name == "serpapi"
