"""Mock face detection and embedding extraction for testing and offline execution."""

from pathlib import Path
from typing import Sequence

from src.face.base import (
    BoundingBox,
    DetectedFace,
    FaceDetector,
    FaceEmbedder,
    FaceEmbedding,
)


class MockFaceDetector(FaceDetector):
    """Mock face detector returning deterministic detected faces."""

    def __init__(
        self,
        default_bbox: BoundingBox | None = None,
        default_confidence: float = 0.985,
    ) -> None:
        self.default_bbox = default_bbox or BoundingBox(x1=50, y1=60, x2=200, y2=250)
        self.default_confidence = default_confidence

    def detect_faces(self, image_data: bytes | Path | str) -> list[DetectedFace]:
        """Return a single detected face if image_data is valid, empty list if empty bytes."""
        if isinstance(image_data, bytes) and len(image_data) == 0:
            return []

        return [
            DetectedFace(
                bbox=self.default_bbox,
                confidence=self.default_confidence,
                landmarks=[(100.0, 110.0), (150.0, 110.0), (125.0, 140.0)],
            )
        ]


class MockFaceEmbedder(FaceEmbedder):
    """Mock face embedder returning deterministic 512D unit-normalized embeddings."""

    def __init__(
        self,
        fixed_vector: Sequence[float] | None = None,
        dimension: int = 512,
        model_name: str = "arcface_mock",
    ) -> None:
        self.dimension = dimension
        self.model_name = model_name
        if fixed_vector is not None:
            self._vector = list(fixed_vector)
        else:
            unit_val = 1.0 / (dimension**0.5)
            self._vector = [unit_val] * dimension

    def extract_embedding(
        self,
        image_data: bytes | Path | str,
        bbox: BoundingBox | None = None,
    ) -> FaceEmbedding:
        """Return the deterministic 512D embedding vector."""
        return FaceEmbedding(
            vector=list(self._vector),
            dimension=self.dimension,
            model_name=self.model_name,
        )
