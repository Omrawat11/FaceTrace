"""Real biometric face detection and ArcFace embedding extraction using InsightFace."""

import logging
import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from src.core.config import Settings, get_settings
from src.face.base import (
    BoundingBox,
    DetectedFace,
    FaceDetector,
    FaceEmbedder,
    FaceEmbedding,
)
from src.face.mock import MockFaceDetector, MockFaceEmbedder

logger = logging.getLogger(__name__)

# Global cached FaceAnalysis instance
_APP_INSTANCE = None


def get_insightface_app(model_name: str = "buffalo_s") -> Any:
    """Retrieve or lazily initialize the singleton InsightFace FaceAnalysis application."""
    global _APP_INSTANCE
    if _APP_INSTANCE is not None:
        return _APP_INSTANCE

    try:
        from insightface.app import FaceAnalysis

        root_dir = os.path.expanduser("~/.insightface")
        app = FaceAnalysis(name=model_name, root=root_dir, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _APP_INSTANCE = app
        logger.info("InsightFace FaceAnalysis (%s) initialized successfully on CPU.", model_name)
        return _APP_INSTANCE
    except Exception as exc:
        logger.error("Failed to initialize InsightFace FaceAnalysis: %s", exc)
        raise RuntimeError(f"InsightFace initialization failed: {exc}") from exc


def load_bgr_image(image_data: bytes | Path | str) -> np.ndarray:
    """Safely decode image bytes, Path, or filepath string into an OpenCV BGR numpy array."""
    if isinstance(image_data, bytes):
        if len(image_data) == 0:
            raise ValueError("Cannot decode empty image bytes.")
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes with OpenCV.")
        return img

    path_obj = Path(str(image_data))
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file does not exist: {path_obj}")

    # Read binary bytes first to avoid non-ASCII Windows path encoding bugs in cv2.imread
    raw_bytes = path_obj.read_bytes()
    if len(raw_bytes) == 0:
        raise ValueError(f"Image file is empty: {path_obj}")
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV failed to decode image from path: {path_obj}")
    return img


class InsightFaceDetector(FaceDetector):
    """Production face detector using InsightFace SCRFD model."""

    def __init__(
        self,
        model_name: str = "buffalo_s",
        confidence_threshold: float = 0.50,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._app = get_insightface_app(model_name=self.model_name)

    def detect_faces(self, image_data: bytes | Path | str) -> list[DetectedFace]:
        """Detect all faces in the image meeting the confidence threshold."""
        try:
            bgr_img = load_bgr_image(image_data)
        except Exception as exc:
            logger.warning("Could not load image for face detection: %s", exc)
            return []

        raw_faces = self._app.get(bgr_img)
        detected: list[DetectedFace] = []

        for f in raw_faces:
            conf = float(f.det_score)
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            landmarks: list[tuple[float, float]] = []
            if hasattr(f, "kps") and f.kps is not None:
                landmarks = [(float(pt[0]), float(pt[1])) for pt in f.kps]

            detected.append(
                DetectedFace(
                    bbox=bbox,
                    confidence=conf,
                    landmarks=landmarks,
                )
            )

        logger.debug("Detected %d face(s) in image (threshold=%.2f)", len(detected), self.confidence_threshold)
        return detected


class InsightFaceEmbedder(FaceEmbedder):
    """Production biometric embedder using InsightFace ArcFace 512D model."""

    def __init__(self, model_name: str = "buffalo_s") -> None:
        self.model_name = model_name
        self._app = get_insightface_app(model_name=self.model_name)

    def extract_embedding(
        self,
        image_data: bytes | Path | str,
        bbox: BoundingBox | None = None,
    ) -> FaceEmbedding:
        """Extract a 512D normalized ArcFace biometric embedding."""
        bgr_img = load_bgr_image(image_data)
        raw_faces = self._app.get(bgr_img)

        if not raw_faces:
            raise ValueError("No face detected in image to extract biometric embedding.")

        selected_face = raw_faces[0]
        if bbox is not None and len(raw_faces) > 1:
            # Match the face closest to the requested bounding box
            best_iou = -1.0
            for f in raw_faces:
                fx1, fy1, fx2, fy2 = f.bbox
                # Simple intersection over union
                ix1 = max(bbox.x1, fx1)
                iy1 = max(bbox.y1, fy1)
                ix2 = min(bbox.x2, fx2)
                iy2 = min(bbox.y2, fy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = bbox.area + ((fx2 - fx1) * (fy2 - fy1)) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    selected_face = f

        raw_vec = selected_face.embedding
        norm = np.linalg.norm(raw_vec)
        if norm > 1e-9:
            norm_vec = (raw_vec / norm).tolist()
        else:
            norm_vec = raw_vec.tolist()

        return FaceEmbedding(
            vector=[float(v) for v in norm_vec],
            dimension=len(norm_vec),
            model_name=f"arcface_{self.model_name}",
        )


def get_face_detector(settings: Settings | None = None) -> FaceDetector:
    """Factory returning configured FaceDetector implementation."""
    cfg = settings or get_settings()
    provider = cfg.FACE_DETECTOR_PROVIDER.lower()
    if provider == "insightface":
        return InsightFaceDetector(confidence_threshold=cfg.FACE_DETECTION_CONFIDENCE)
    elif provider == "mock":
        return MockFaceDetector()
    else:
        raise ValueError(f"Unsupported face detector provider: {provider}")


def get_face_embedder(settings: Settings | None = None) -> FaceEmbedder:
    """Factory returning configured FaceEmbedder implementation."""
    cfg = settings or get_settings()
    provider = cfg.FACE_DETECTOR_PROVIDER.lower()
    if provider == "insightface":
        return InsightFaceEmbedder()
    elif provider == "mock":
        return MockFaceEmbedder()
    else:
        raise ValueError(f"Unsupported face embedder provider: {provider}")
