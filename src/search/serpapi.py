"""SerpApi Google Lens reverse-image search provider implementation."""

import io
import logging
from pathlib import Path
from typing import Any
import requests
from PIL import Image

from src.core.config import get_settings
from src.core.types import PlatformType
from src.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

SERPAPI_IMAGE_UPLOAD_URL = "https://serpapi.com/image"
SERPAPI_SEARCH_URL = "https://serpapi.com/search"
MAX_SERPAPI_IMAGE_SIZE_BYTES = 500 * 1024  # 500 KB limit from SerpApi specs


class SerpApiError(Exception):
    """Exception raised for errors during SerpApi visual search."""

    pass


def prepare_image_for_upload(image_bytes: bytes, max_bytes: int = MAX_SERPAPI_IMAGE_SIZE_BYTES) -> bytes:
    """Ensure image is valid format and strictly within SerpApi's 500 KB upload limit.

    If size exceeds max_bytes, resizes dimensions and re-encodes as JPEG.
    """
    # Check if image is already a supported format, within byte limit, and within dimensions
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if (
                img.format in ("JPEG", "PNG", "WEBP")
                and len(image_bytes) <= max_bytes
                and max(img.width, img.height) <= 1024
            ):
                return image_bytes
    except Exception:
        pass

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb_img = img.convert("RGB")
            # Downscale if dimensions exceed 1024px
            max_dim = 1024
            if max(rgb_img.width, rgb_img.height) > max_dim:
                rgb_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            # Compress iteratively to fit max_bytes
            quality = 85
            while quality >= 25:
                buffer = io.BytesIO()
                rgb_img.save(buffer, format="JPEG", quality=quality, optimize=True)
                data = buffer.getvalue()
                if len(data) <= max_bytes:
                    return data
                quality -= 15

            return data
    except Exception as exc:
        raise ValueError(f"Failed to process or compress image for upload: {exc}") from exc


class SerpApiSearchProvider(SearchProvider):
    """Genuine reverse-image search provider utilizing the SerpApi Google Lens engine."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.SERPAPI_API_KEY
        self.timeout = timeout or settings.SEARCH_REQUEST_TIMEOUT_SECONDS
        self._session = session or requests.Session()

    @property
    def provider_name(self) -> str:
        return "serpapi"

    def _upload_local_image(self, image_bytes: bytes) -> str:
        """Upload image to SerpApi Image API and return the temporary image_id."""
        if not self.api_key:
            raise SerpApiError(
                "SERPAPI_API_KEY is not configured. Set SERPAPI_API_KEY in .env or pass to constructor."
            )

        processed_bytes = prepare_image_for_upload(image_bytes)

        files = {"image": ("query.jpg", processed_bytes, "image/jpeg")}
        data = {"api_key": self.api_key}

        logger.info("Uploading query face image to SerpApi (%d bytes)...", len(processed_bytes))
        try:
            response = self._session.post(
                SERPAPI_IMAGE_UPLOAD_URL,
                files=files,
                data=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            res_json = response.json()
        except requests.RequestException as exc:
            raise SerpApiError(f"Failed to upload image to SerpApi: {exc}") from exc

        image_id = res_json.get("image_id")
        if not image_id:
            error_msg = res_json.get("error", "No image_id returned from SerpApi Image API")
            raise SerpApiError(f"SerpApi upload error: {error_msg}")

        logger.info("Image successfully uploaded to SerpApi. image_id: %s", image_id)
        return str(image_id)

    def search_by_image(
        self,
        image_data: bytes | Path | str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Execute genuine reverse-image search via SerpApi Google Lens engine.

        Args:
            image_data: Image file path (Path or str), public image URL (str starting with http),
                        or raw image bytes.
            max_results: Maximum candidate search results to return.

        Returns:
            List of SearchResult objects discovered on public indexed pages.
        """
        if not self.api_key:
            raise SerpApiError(
                "SERPAPI_API_KEY is not configured. Set SERPAPI_API_KEY in .env or pass to constructor."
            )

        params: dict[str, Any] = {
            "engine": "google_lens",
            "api_key": self.api_key,
            "hl": "en",
        }

        # Case 1: image_data is a public URL
        if isinstance(image_data, str) and (
            image_data.startswith("http://") or image_data.startswith("https://")
        ):
            params["url"] = image_data
            logger.info("Querying Google Lens via URL: %s", image_data)

        # Case 2: image_data is a file path or Path object
        elif isinstance(image_data, (Path, str)) and Path(str(image_data)).exists():
            file_path = Path(str(image_data))
            image_bytes = file_path.read_bytes()
            image_id = self._upload_local_image(image_bytes)
            params["image_id"] = image_id

        # Case 3: image_data is raw bytes
        elif isinstance(image_data, bytes):
            if not image_data:
                raise ValueError("image_data bytes cannot be empty.")
            image_id = self._upload_local_image(image_data)
            params["image_id"] = image_id

        else:
            raise ValueError(
                f"Invalid image_data: must be existing file path, public URL, or non-empty bytes. Received: {type(image_data)}"
            )

        logger.info("Executing Google Lens search via SerpApi...")
        try:
            resp = self._session.get(
                SERPAPI_SEARCH_URL,
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise SerpApiError(f"SerpApi Google Lens search request failed: {exc}") from exc

        if "error" in payload:
            raise SerpApiError(f"SerpApi error: {payload['error']}")

        return self._parse_google_lens_results(payload, max_results=max_results)

    def _parse_google_lens_results(
        self, payload: dict[str, Any], max_results: int
    ) -> list[SearchResult]:
        """Extract SearchResult items from SerpApi visual_matches and exact_matches."""
        candidates_raw: list[dict[str, Any]] = []

        # Google Lens returns results under 'visual_matches' and optionally 'exact_matches'
        if "visual_matches" in payload and isinstance(payload["visual_matches"], list):
            candidates_raw.extend(payload["visual_matches"])

        if "exact_matches" in payload and isinstance(payload["exact_matches"], list):
            candidates_raw.extend(payload["exact_matches"])

        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for item in candidates_raw:
            if len(results) >= max_results:
                break

            page_url = item.get("link")
            if not page_url or page_url in seen_urls:
                continue

            seen_urls.add(page_url)

            title = item.get("title") or "Visual Match Candidate"
            source = item.get("source") or ""
            thumbnail = item.get("thumbnail") or item.get("image")
            platform = PlatformType.from_url(page_url)

            try:
                search_res = SearchResult(
                    url=page_url,
                    title=title,
                    source_domain=source,
                    thumbnail_url=thumbnail,
                    snippet=item.get("snippet"),
                    platform=platform,
                    raw_metadata=item,
                )
                results.append(search_res)
            except ValueError as exc:
                logger.warning("Skipping invalid candidate result: %s", exc)
                continue

        logger.info("Discovered %d candidate search results from Google Lens.", len(results))
        return results
