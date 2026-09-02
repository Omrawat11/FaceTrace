"""Base interfaces and data structures for face detection and biometric embedding."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates for a detected face [x1, y1, x2, y2]."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class DetectedFace:
    """Represents a localized face detection with confidence score and facial landmarks."""

    bbox: BoundingBox
    confidence: float
    landmarks: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class FaceEmbedding:
    """Biometric feature representation (typically 512D normalized ArcFace vector)."""

    vector: list[float]
    dimension: int
    model_name: str = "arcface"

    def __post_init__(self) -> None:
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Embedding length {len(self.vector)} does not match specified dimension {self.dimension}"
            )


def compute_cosine_similarity(
    vec1: Sequence[float],
    vec2: Sequence[float],
    eps: float = 1e-9,
) -> float:
    """Compute cosine similarity between two feature vectors: dot(u, v) / (|u| * |v|).

    Returns a normalized similarity score in range [-1.0, 1.0].
    """
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vector dimensions must match for cosine similarity: {len(vec1)} vs {len(vec2)}"
        )

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 < eps or norm2 < eps:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    # Clip numerical float inaccuracies to [-1.0, 1.0]
    return max(-1.0, min(1.0, similarity))


class FaceDetector(ABC):
    """Abstract base class for face detection algorithms (e.g. InsightFace SCRFD)."""

    @abstractmethod
    def detect_faces(self, image_data: bytes | Path | str) -> list[DetectedFace]:
        """Detect all faces in the provided image."""
        pass


class FaceEmbedder(ABC):
    """Abstract base class for extracting deep biometric embeddings (e.g. InsightFace ArcFace)."""

    @abstractmethod
    def extract_embedding(
        self,
        image_data: bytes | Path | str,
        bbox: BoundingBox | None = None,
    ) -> FaceEmbedding:
        """Extract a high-dimensional feature vector for a face."""
        pass
