"""Candidate retrieval, facial feature evaluation, similarity scoring, and ranking."""

import io
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import requests
from PIL import Image

from src.candidates.base import Candidate, CandidateMatch, CandidateProcessor
from src.core.config import get_settings
from src.face.base import (
    FaceDetector,
    FaceEmbedder,
    FaceEmbedding,
    compute_cosine_similarity,
)
from src.face.insightface_engine import get_face_detector, get_face_embedder
from src.search.base import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 FaceTrace/1.0"
)


class DefaultCandidateProcessor(CandidateProcessor):
    """Downloads candidate imagery, extracts facial embeddings, and ranks biometric matches."""

    def __init__(
        self,
        face_detector: FaceDetector | None = None,
        face_embedder: FaceEmbedder | None = None,
        session: requests.Session | None = None,
        timeout: int | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.face_detector = face_detector or get_face_detector()
        self.face_embedder = face_embedder or get_face_embedder()
        self.timeout = timeout or settings.SEARCH_REQUEST_TIMEOUT_SECONDS
        self.cache_dir = cache_dir or settings.CANDIDATES_CACHE_DIR

        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_image_bytes(self, image_url: str) -> bytes | None:
        """Retrieve candidate image bytes from local disk, file URI, or HTTP URL.

        Validates that the retrieved bytes represent a readable image.
        """
        if not image_url or not isinstance(image_url, str):
            return None

        # Case 1: Local file path on disk
        candidate_path = Path(image_url)
        if candidate_path.exists() and candidate_path.is_file():
            try:
                data = candidate_path.read_bytes()
                if self._verify_image_data(data):
                    return data
            except Exception as exc:
                logger.warning("Failed to read local candidate file %s: %s", image_url, exc)
                return None

        # Case 2: file:// URI scheme
        if image_url.startswith("file://"):
            try:
                parsed = urlparse(image_url)
                # On Windows, path may be /C:/path
                local_path_str = unquote(parsed.path)
                if local_path_str.startswith("/") and len(local_path_str) > 2 and local_path_str[2] == ":":
                    local_path_str = local_path_str[1:]
                local_path = Path(local_path_str)
                if local_path.exists() and local_path.is_file():
                    data = local_path.read_bytes()
                    if self._verify_image_data(data):
                        return data
            except Exception as exc:
                logger.warning("Failed to parse or read file URI %s: %s", image_url, exc)
                return None

        # Case 3: HTTP / HTTPS URL
        if image_url.startswith("http://") or image_url.startswith("https://"):
            try:
                logger.debug("Fetching candidate image from web: %s", image_url)
                resp = self._session.get(image_url, timeout=self.timeout)
                if resp.status_code == 200 and resp.content:
                    if self._verify_image_data(resp.content):
                        return resp.content
                logger.warning(
                    "HTTP fetch failed for %s (Status: %s)", image_url, resp.status_code
                )
                return None
            except requests.RequestException as exc:
                logger.warning("Error fetching candidate image %s: %s", image_url, exc)
                return None

        return None

    def _verify_image_data(self, data: bytes) -> bool:
        """Ensure byte payload is non-empty and decodes as a valid image."""
        if not data or len(data) == 0:
            return False
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
            return True
        except Exception:
            return False

    def fetch_and_evaluate(
        self,
        search_results: list[SearchResult],
        query_embedding: FaceEmbedding,
        similarity_threshold: float = 0.45,
    ) -> list[CandidateMatch]:
        """Download candidate images, detect faces, compute similarities, and return ranked matches.

        Args:
            search_results: Candidate search results from SearchProvider.
            query_embedding: 512D biometric vector of the validated query face.
            similarity_threshold: Minimum cosine similarity required to qualify as a match.

        Returns:
            List of CandidateMatch instances sorted descending by similarity_score.
        """
        matches: list[CandidateMatch] = []

        for result in search_results:
            # Determine candidate image source
            image_source = (
                result.thumbnail_url
                or (result.raw_metadata.get("file_path") if isinstance(result.raw_metadata, dict) else None)
                or (result.raw_metadata.get("image") if isinstance(result.raw_metadata, dict) else None)
                or result.url
            )

            # Retrieve image bytes safely
            image_bytes = self._fetch_image_bytes(str(image_source)) if image_source else None

            # Handle missing or invalid candidate image
            if not image_bytes:
                logger.warning("Candidate image missing or invalid for result: %s", result.url)
                cand = Candidate.create(
                    url=result.url,
                    image_url=str(image_source or ""),
                    image_bytes=b"",
                    title=result.title,
                    platform=result.platform,
                )
                match = CandidateMatch(
                    candidate=cand,
                    similarity_score=0.0,
                    detected_face=None,
                    is_match=False,
                    detected_faces_count=0,
                    status="ERROR",
                    threshold_used=similarity_threshold,
                    error_message="Candidate image missing or invalid",
                )
                matches.append(match)
                continue

            # Valid candidate entity with SHA-256 hash
            candidate = Candidate.create(
                url=result.url,
                image_url=str(image_source),
                image_bytes=image_bytes,
                title=result.title,
                platform=result.platform,
            )

            local_path = (
                result.raw_metadata.get("file_path")
                if isinstance(result.raw_metadata, dict)
                else None
            )

            # Optional local caching
            if self.cache_dir:
                cache_file = self.cache_dir / f"{candidate.image_hash[:16]}.jpg"
                if not cache_file.exists():
                    try:
                        cache_file.write_bytes(image_bytes)
                    except Exception as exc:
                        logger.debug("Failed to cache candidate image: %s", exc)

            # Detect faces in candidate image
            try:
                detected_faces = self.face_detector.detect_faces(image_bytes)
            except Exception as exc:
                logger.warning("Face detection failure on '%s': %s", candidate.title, exc)
                detected_faces = []

            face_count = len(detected_faces)

            # Case: Zero faces detected
            if face_count == 0:
                logger.info("No face detected in candidate '%s'. Status: NO_FACE_DETECTED", candidate.title)
                match = CandidateMatch(
                    candidate=candidate,
                    similarity_score=0.0,
                    detected_face=None,
                    is_match=False,
                    detected_faces_count=0,
                    status="NO_FACE_DETECTED",
                    threshold_used=similarity_threshold,
                    local_image_path=str(local_path) if local_path else None,
                )
                matches.append(match)
                continue

            # Case: One or more faces detected
            # Compare query embedding against EVERY detected face and find highest similarity
            face_evaluations: list[tuple[Any, float]] = []

            for face in detected_faces:
                try:
                    candidate_embedding = self.face_embedder.extract_embedding(
                        image_bytes, bbox=face.bbox
                    )
                    score = compute_cosine_similarity(
                        query_embedding.vector, candidate_embedding.vector
                    )
                    face_evaluations.append((face, score))
                except Exception as exc:
                    logger.warning(
                        "Failed to extract face embedding in candidate '%s': %s",
                        candidate.title,
                        exc,
                    )

            if not face_evaluations:
                # All embedding attempts failed
                match = CandidateMatch(
                    candidate=candidate,
                    similarity_score=0.0,
                    detected_face=None,
                    is_match=False,
                    detected_faces_count=face_count,
                    status="ERROR",
                    threshold_used=similarity_threshold,
                    error_message="Failed to extract facial embeddings from detected faces",
                    local_image_path=str(local_path) if local_path else None,
                )
                matches.append(match)
                continue

            # Select the face producing the highest similarity score
            best_face, highest_score = max(face_evaluations, key=lambda item: item[1])
            is_match = highest_score >= similarity_threshold
            status = "MATCH" if is_match else "NO_MATCH"

            match = CandidateMatch(
                candidate=candidate,
                similarity_score=highest_score,
                detected_face=best_face,
                is_match=is_match,
                detected_faces_count=face_count,
                status=status,
                threshold_used=similarity_threshold,
                local_image_path=str(local_path) if local_path else None,
            )
            matches.append(match)
            logger.info(
                "Candidate '%s': %d face(s), best_score=%.4f (threshold=%.2f, status=%s)",
                candidate.title,
                face_count,
                highest_score,
                similarity_threshold,
                status,
            )

        # Sort all candidates by similarity score in descending order
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches

    def select_best_match(
        self,
        matches: list[CandidateMatch],
    ) -> CandidateMatch | None:
        """Select the highest-confidence match meeting or exceeding the threshold.

        Returns None if no candidate qualifies as a match.
        """
        valid_matches = [m for m in matches if m.is_match]
        if not valid_matches:
            return None
        return max(valid_matches, key=lambda m: m.similarity_score)
