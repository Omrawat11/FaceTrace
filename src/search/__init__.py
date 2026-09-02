"""Search module providing provider abstractions and search models."""

from src.core.config import Settings, get_settings
from src.search.base import SearchProvider, SearchResult
from src.search.mock import MockSearchProvider
from src.search.serpapi import SerpApiError, SerpApiSearchProvider


def get_search_provider(settings: Settings | None = None) -> SearchProvider:
    """Factory function creating the configured search provider instance."""
    cfg = settings or get_settings()
    provider_type = cfg.SEARCH_PROVIDER.lower()

    if provider_type == "serpapi":
        return SerpApiSearchProvider(
            api_key=cfg.SERPAPI_API_KEY,
            timeout=cfg.SEARCH_REQUEST_TIMEOUT_SECONDS,
        )
    elif provider_type == "mock":
        return MockSearchProvider()
    else:
        raise ValueError(
            f"Unsupported search provider: '{cfg.SEARCH_PROVIDER}'. Must be 'serpapi' or 'mock'."
        )


__all__ = [
    "SearchProvider",
    "SearchResult",
    "SerpApiSearchProvider",
    "SerpApiError",
    "MockSearchProvider",
    "get_search_provider",
]
