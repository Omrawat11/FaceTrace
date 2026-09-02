"""SerpApi Google Lens reverse-image search provider implementation."""

import io
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
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


def _sanitize_text(text: str, api_key: str | None = None) -> str:
    """Ensure API key and sensitive credentials never leak into error messages or logs."""
    if not text:
        return ""
    sanitized = text
    if api_key and api_key in sanitized:
        sanitized = sanitized.replace(api_key, "***REDACTED***")
    # Also scrub any URL query parameters with api_key=...
    sanitized = re.sub(r"([?&]api_key=)[^&]+", r"\1***REDACTED***", sanitized)
    return sanitized


def normalize_url(url: str) -> str:
    """Normalize URL by stripping tracking parameters, fragments, and trailing slashes."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            return url.strip().rstrip("/")

        # Strip common tracking query parameters
        query_params = parse_qsl(parsed.query, keep_blank_values=False)
        tracking_prefixes = ("utm_", "fbclid", "gclid", "ref", "igsh", "ref_src", "_ga")
        filtered_params = [
            (k, v)
            for k, v in query_params
            if not any(k.lower().startswith(p) for p in tracking_prefixes)
        ]
        new_query = urlencode(filtered_params)
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.params,
            new_query,
            "",  # Remove fragment
        ))
        return normalized or url.strip().rstrip("/")
    except Exception:
        return url.strip().rstrip("/")


def detect_platform(url: str, source: str = "") -> PlatformType:
    """Detect platform type generically from URL domain and provider source name."""
    platform = PlatformType.from_url(url)
    if platform not in (PlatformType.WEB, PlatformType.UNKNOWN):
        return platform

    # If domain inference yielded generic WEB, inspect source name
    src_lower = source.lower() if source else ""
    if "instagram" in src_lower:
        return PlatformType.INSTAGRAM
    if "facebook" in src_lower or "fb" in src_lower:
        return PlatformType.FACEBOOK
    if "twitter" in src_lower or " x" in src_lower or src_lower == "x":
        return PlatformType.TWITTER
    if "reddit" in src_lower:
        return PlatformType.REDDIT
    if "linkedin" in src_lower:
        return PlatformType.LINKEDIN
    if "tiktok" in src_lower:
        return PlatformType.TIKTOK
    if "youtube" in src_lower:
        return PlatformType.YOUTUBE
    if "pinterest" in src_lower:
        return PlatformType.PINTEREST
    if "wikipedia" in src_lower or "wikimedia" in src_lower:
        return PlatformType.WIKIPEDIA
    if any(k in src_lower for k in ("news", "times", "post", "bbc", "cnn", "reuters", "guardian", "daily")):
        return PlatformType.NEWS

    return PlatformType.WEB


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
        except requests.exceptions.Timeout as exc:
            raise SerpApiError("SerpApi image upload timed out. Check your network connection.") from exc
        except requests.RequestException as exc:
            safe_msg = _sanitize_text(str(exc), self.api_key)
            raise SerpApiError(f"Failed to upload image to SerpApi: {safe_msg}") from exc

        # Handle HTTP response status
        if response.status_code == 401 or response.status_code == 403:
            raise SerpApiError("Invalid or unauthorized SerpApi API key. Check your SERPAPI_API_KEY in .env.")
        elif response.status_code == 429:
            raise SerpApiError("SerpApi rate limit exceeded or search quota exhausted. Check your SerpApi account plan.")
        elif response.status_code >= 500:
            raise SerpApiError("SerpApi service is temporarily unavailable. Check your SerpApi configuration or try again.")

        try:
            res_json = response.json()
        except Exception as exc:
            raise SerpApiError("SerpApi Image API returned an invalid, non-JSON response payload.") from exc

        image_id = res_json.get("image_id")
        if not image_id:
            error_msg = res_json.get("error", "No image_id returned from SerpApi Image API")
            safe_err = _sanitize_text(str(error_msg), self.api_key)
            raise SerpApiError(f"SerpApi upload error: {safe_err}")

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

        # Single request execution with at most one retry for transient errors
        resp = None
        for attempt in range(2):
            try:
                resp = self._session.get(
                    SERPAPI_SEARCH_URL,
                    params=params,
                    timeout=self.timeout,
                )
                # If server error (5xx), retry once after brief pause
                if resp.status_code >= 500 and attempt == 0:
                    logger.warning("SerpApi 5xx transient response. Retrying once...")
                    time.sleep(1.0)
                    continue
                break
            except requests.exceptions.Timeout as exc:
                raise SerpApiError(
                    f"SerpApi search request timed out after {self.timeout}s. Check your network connection."
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                if attempt == 0:
                    logger.warning("SerpApi connection error. Retrying once...")
                    time.sleep(1.0)
                    continue
                safe_msg = _sanitize_text(str(exc), self.api_key)
                raise SerpApiError(f"SerpApi network connection failure: {safe_msg}") from exc
            except requests.RequestException as exc:
                safe_msg = _sanitize_text(str(exc), self.api_key)
                raise SerpApiError(f"SerpApi Google Lens search request failed: {safe_msg}") from exc

        if resp is None:
            raise SerpApiError("Live search is temporarily unavailable. Check your SerpApi configuration or try again.")

        # HTTP error categorization
        if resp.status_code == 401 or resp.status_code == 403:
            try:
                err_payload = resp.json()
                err_msg = err_payload.get("error", "Invalid or unauthorized SerpApi API key.")
            except Exception:
                err_msg = "Invalid or unauthorized SerpApi API key."
            safe_err = _sanitize_text(str(err_msg), self.api_key)
            raise SerpApiError(f"SerpApi authentication failure: {safe_err}")

        if resp.status_code == 429:
            raise SerpApiError(
                "SerpApi rate limit exceeded or search quota exhausted. Check your SerpApi account plan."
            )

        if resp.status_code >= 500:
            raise SerpApiError(
                "Live search is temporarily unavailable. Check your SerpApi configuration or try again."
            )

        try:
            payload = resp.json()
        except Exception as exc:
            raise SerpApiError("SerpApi returned an invalid, non-JSON response payload.") from exc

        if "error" in payload:
            safe_err = _sanitize_text(str(payload["error"]), self.api_key)
            raise SerpApiError(f"SerpApi error: {safe_err}")

        return self._parse_google_lens_results(payload, max_results=max_results)

    def _parse_google_lens_results(
        self, payload: dict[str, Any], max_results: int
    ) -> list[SearchResult]:
        """Extract and normalize SearchResult items from SerpApi visual_matches and exact_matches.

        Applies canonical URL normalization and candidate deduplication.
        """
        candidates_raw: list[dict[str, Any]] = []

        # Google Lens returns results under 'visual_matches' and optionally 'exact_matches'
        if "visual_matches" in payload and isinstance(payload["visual_matches"], list):
            candidates_raw.extend(payload["visual_matches"])

        if "exact_matches" in payload and isinstance(payload["exact_matches"], list):
            candidates_raw.extend(payload["exact_matches"])

        results: list[SearchResult] = []
        # Deduplication tracking: (normalized_page_url, image_source)
        seen_keys: set[tuple[str, str]] = set()

        for item in candidates_raw:
            if len(results) >= max_results:
                break

            page_url = item.get("link")
            if not page_url or not isinstance(page_url, str):
                continue

            # Normalize URL to strip tracking tokens
            clean_url = normalize_url(page_url)
            thumbnail = item.get("thumbnail") or item.get("image")
            img_key = str(thumbnail or "")

            dedup_key = (clean_url, img_key)
            if dedup_key in seen_keys:
                logger.debug("Skipping duplicate search result: %s", clean_url)
                continue

            seen_keys.add(dedup_key)

            title = item.get("title") or "Visual Match Candidate"
            source = item.get("source") or ""
            platform = detect_platform(clean_url, source)

            try:
                search_res = SearchResult(
                    url=clean_url,
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
