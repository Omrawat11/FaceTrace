"""Face analysis and biometric embedding module."""

from src.face.base import (
    BoundingBox,
    DetectedFace,
    FaceDetector,
    FaceEmbedder,
    FaceEmbedding,
    compute_cosine_similarity,
)
from src.face.insightface_engine import (
    InsightFaceDetector,
    InsightFaceEmbedder,
    get_face_detector,
    get_face_embedder,
)
from src.face.mock import MockFaceDetector, MockFaceEmbedder
from src.face.query_processor import (
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    QueryFaceProcessingError,
    QueryFaceProcessor,
)

__all__ = [
    "BoundingBox",
    "DetectedFace",
    "FaceEmbedding",
    "FaceDetector",
    "FaceEmbedder",
    "compute_cosine_similarity",
    "InsightFaceDetector",
    "InsightFaceEmbedder",
    "MockFaceDetector",
    "MockFaceEmbedder",
    "get_face_detector",
    "get_face_embedder",
    "QueryFaceProcessor",
    "QueryFaceProcessingError",
    "InvalidImageError",
    "NoFaceDetectedError",
    "MultipleFacesDetectedError",
]
