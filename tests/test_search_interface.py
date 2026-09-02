"""Unit tests for SearchProvider data structures and URL/platform parsing."""

import pytest

from src.core.types import PlatformType
from src.search.base import SearchResult


def test_search_result_domain_auto_inference() -> None:
    """Test that domain is automatically extracted from URL if not specified."""
    res = SearchResult(
        url="https://www.instagram.com/p/C12345XYZ/",
        title="Public Post",
        source_domain="",
    )
    assert res.source_domain == "www.instagram.com"
    assert res.platform == PlatformType.INSTAGRAM


def test_platform_type_detection() -> None:
    """Test platform detection for key platforms."""
    urls = [
        ("https://twitter.com/user/status/123", PlatformType.TWITTER),
        ("https://x.com/user/status/456", PlatformType.TWITTER),
        ("https://www.reddit.com/r/pics/comments/789", PlatformType.REDDIT),
        ("https://en.wikipedia.org/wiki/Portrait", PlatformType.WIKIPEDIA),
        ("https://example.com/some/blog/post", PlatformType.WEB),
    ]
    for url, expected_platform in urls:
        res = SearchResult(url=url, title="Test", source_domain="")
        assert res.platform == expected_platform


def test_search_result_empty_url() -> None:
    """Test that empty URL raises ValueError."""
    with pytest.raises(ValueError):
        SearchResult(url="", title="Invalid", source_domain="")
