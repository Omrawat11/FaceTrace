"""Deterministic mock search provider serving controlled demo data for local validation."""

import logging
from pathlib import Path
from typing import Sequence

from src.core.config import get_settings
from src.core.types import PlatformType
from src.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class MockSearchProvider(SearchProvider):
    """Mock search provider returning candidates from the local controlled demo dataset."""

    def __init__(
        self,
        predefined_results: Sequence[SearchResult] | None = None,
        demo_dir: Path | str | None = None,
    ) -> None:
        self._predefined_results = list(predefined_results) if predefined_results else None
        settings = get_settings()
        self.demo_dir = Path(demo_dir or settings.DEMO_IMAGES_DIR)

    @property
    def provider_name(self) -> str:
        return "mock"

    def _discover_demo_results(self) -> list[SearchResult]:
        """Scan demo_images directory and package candidates into SearchResult objects."""
        if not self.demo_dir.exists():
            logger.warning("Demo directory does not exist: %s", self.demo_dir)
            return []

        results: list[SearchResult] = []
        # Exclude query baseline face from search results
        excluded_files = {"user_primary.jpg", ".gitkeep", "README.md"}

        # Canonical title & platform mappings for known demo images
        manifest = {
            "user_match_01.jpg": (
                "https://instagram.com/p/C_abc123xyz/",
                "Jane Doe (@janedoe) • Instagram portrait photo (Match 01)",
                PlatformType.INSTAGRAM,
            ),
            "user_match_02.jpg": (
                "https://x.com/janedoe/status/1830123456789",
                "Jane Doe on X: 'Outdoor afternoon coffee' (Match 02)",
                PlatformType.TWITTER,
            ),
            "user_match_03.jpg": (
                "https://linkedin.com/in/janedoe-cto",
                "Jane Doe - VP of Engineering - Acme Corp | LinkedIn (Match 03)",
                PlatformType.LINKEDIN,
            ),
            "non_match_01.jpg": (
                "https://wikipedia.org/wiki/Professor_Smith",
                "Professor Arthur Smith - Academic Biography | Wikipedia",
                PlatformType.WIKIPEDIA,
            ),
            "non_match_02.jpg": (
                "https://reddit.com/r/portraits/comments/autumn_park",
                "Autumn foliage outdoor portrait session : r/portraits",
                PlatformType.REDDIT,
            ),
            "multi_face_01.jpg": (
                "https://twitter.com/devcon_official/status/booth_team",
                "DevCon Engineering Booth Team and Panel Speakers | X",
                PlatformType.TWITTER,
            ),
            "no_face_01.jpg": (
                "https://pinterest.com/pin/alpine_lake_reflection",
                "Scenic Alpine Lake and Mountain Morning Landscape | Pinterest",
                PlatformType.PINTEREST,
            ),
        }

        # Look for supported image files
        for img_path in sorted(self.demo_dir.glob("*.jpg")) + sorted(self.demo_dir.glob("*.png")):
            if img_path.name in excluded_files:
                continue

            resolved_path = str(img_path.resolve())

            if img_path.name in manifest:
                url, title, platform = manifest[img_path.name]
            else:
                # Dynamic fallback for user-placed custom images
                stem = img_path.stem
                url = f"https://example.com/demo/{stem}"
                title = f"Demo Candidate: {stem.replace('_', ' ').title()}"
                platform = PlatformType.WEB

            result = SearchResult(
                url=url,
                title=title,
                source_domain="",  # Auto-inferred in SearchResult
                thumbnail_url=resolved_path,
                snippet=f"Local demo candidate extracted from {img_path.name}",
                platform=platform,
                raw_metadata={
                    "file_path": resolved_path,
                    "filename": img_path.name,
                    "demo": True,
                },
            )
            results.append(result)

        return results

    def search_by_image(
        self,
        image_data: bytes | Path | str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Return simulated candidate web and social media search results."""
        if not image_data:
            raise ValueError("image_data cannot be empty")

        if self._predefined_results is not None:
            return self._predefined_results[:max_results]

        results = self._discover_demo_results()
        return results[:max_results]
