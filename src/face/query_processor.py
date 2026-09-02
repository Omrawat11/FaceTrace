"""Validation, face localization, and biometric embedding for probe query faces."""

import io
import logging
from pathlib import Path
from PIL import Image

from src.face.base import DetectedFace, FaceDetector, FaceEmbedder, FaceEmbedding
from src.face.insightface_engine import get_face_detector, get_face_embedder

logger = logging.getLogger(__name__)


class QueryFaceProcessingError(Exception):
    """Base exception for query face processing failures."""

    pass


class InvalidImageError(QueryFaceProcessingError):
    """Raised when the query image file or bytes cannot be decoded."""

    pass


class NoFaceDetectedError(QueryFaceProcessingError):
    """Raised when zero faces are detected in the query image."""

    pass


class MultipleFacesDetectedError(QueryFaceProcessingError):
    """Raised when more than one face is detected in the query image."""

    pass


class QueryFaceProcessor:
    """Processes user query face images, enforcing single-face validation and biometric embedding."""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        embedder: FaceEmbedder | None = None,
    ) -> None:
        self.detector = detector or get_face_detector()
        self.embedder = embedder or get_face_embedder()

    def validate_image_format(self, image_data: bytes | Path | str) -> bytes:
        """Validate that input data represents a decodable image and return its binary bytes."""
        if isinstance(image_data, bytes):
            raw_bytes = image_data
        elif isinstance(image_data, (Path, str)):
            path = Path(str(image_data))
            if not path.exists():
                raise InvalidImageError(f"Query image path does not exist: {path}")
            raw_bytes = path.read_bytes()
        else:
            raise InvalidImageError(f"Unsupported image data type: {type(image_data)}")

        if len(raw_bytes) == 0:
            raise InvalidImageError("Query image payload is empty (0 bytes).")

        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()
        except Exception as exc:
            raise InvalidImageError(f"Failed to decode query image: {exc}") from exc

        return raw_bytes

    def process_query_image(
        self,
        image_data: bytes | Path | str,
    ) -> tuple[DetectedFace, FaceEmbedding]:
        """Validate, detect, and extract ArcFace embedding from the query face.

        Args:
            image_data: Image file bytes, Path, or string path.

        Returns:
            Tuple of (DetectedFace, FaceEmbedding)

        Raises:
            InvalidImageError: If the image is invalid or cannot be decoded.
            NoFaceDetectedError: If zero faces are found.
            MultipleFacesDetectedError: If more than one face is found.
        """
        raw_bytes = self.validate_image_format(image_data)

        # Detect faces
        faces = self.detector.detect_faces(raw_bytes)
        face_count = len(faces)

        if face_count == 0:
            logger.warning("Zero faces detected in query image.")
            raise NoFaceDetectedError(
                "No face detected in query image. Please provide an image containing a clearly visible face."
            )

        if face_count > 1:
            logger.warning("Multiple faces detected in query image (%d faces).", face_count)
            raise MultipleFacesDetectedError(
                f"Multiple faces detected ({face_count} faces found). "
                "Please provide an image containing only your face."
            )

        primary_face = faces[0]
        logger.info(
            "Single query face validated successfully (bbox=%s, conf=%.3f).",
            primary_face.bbox.as_list(),
            primary_face.confidence,
        )

        embedding = self.embedder.extract_embedding(raw_bytes, bbox=primary_face.bbox)
        return primary_face, embedding
