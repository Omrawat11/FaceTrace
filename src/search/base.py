"""Abstract interfaces and data structures for genuine web/social search."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from src.core.types import PlatformType


@dataclass
class SearchResult:
    """A single genuine web/social search result returned by a search provider."""

    url: str
    title: str
    source_domain: str
    thumbnail_url: str | None = None
    snippet: str | None = None
    platform: PlatformType = PlatformType.UNKNOWN
    raw_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("SearchResult url must not be empty.")
        # Auto-infer domain if not explicitly provided
        if not self.source_domain:
            parsed = urlparse(self.url)
            self.source_domain = parsed.netloc or "unknown"
        # Auto-infer platform from url
        if self.platform == PlatformType.UNKNOWN:
            self.platform = PlatformType.from_url(self.url)


class SearchProvider(ABC):
    """Abstract base class for genuine reverse-image and web search providers.

    All implementations MUST execute real runtime searches (e.g. SerpApi Google Lens,
    Bing Visual Search, or local search index) without hardcoding or pre-selecting results.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the search provider."""
        pass

    @abstractmethod
    def search_by_image(
        self,
        image_data: bytes | Path | str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Execute genuine search using the provided query image data or path.

        Args:
            image_data: Local image file path, raw image bytes, or URL.
            max_results: Maximum candidate matches to return.

        Returns:
            List of SearchResult objects discovered on public indexed pages.
        """
        pass
