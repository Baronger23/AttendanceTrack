"""
InsightFace ArcFace backend.

This backend is optional. It is used when InsightFace can load its model;
otherwise FaceService falls back to the existing FaceNet pipeline.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


class ArcFaceBackend:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.app = None
        self.available = False
        self.model_name = getattr(settings, "INSIGHTFACE_MODEL_NAME", "buffalo_l")

        if getattr(settings, "AI_FACE_BACKEND", "insightface").lower() != "insightface":
            self._initialized = True
            logger.info("InsightFace ArcFace backend disabled by AI_FACE_BACKEND")
            return

        try:
            from insightface.app import FaceAnalysis

            providers = getattr(settings, "INSIGHTFACE_PROVIDERS", ["CPUExecutionProvider"])
            self.app = FaceAnalysis(name=self.model_name, providers=providers)
            self.app.prepare(
                ctx_id=getattr(settings, "INSIGHTFACE_CTX_ID", -1),
                det_size=getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640)),
            )
            self.available = True
            logger.info("InsightFace ArcFace backend loaded: %s", self.model_name)
        except Exception as exc:
            logger.warning("InsightFace ArcFace backend unavailable: %s", exc)
            self.app = None
            self.available = False

        self._initialized = True

    def extract_embedding(self, image: np.ndarray) -> dict:
        if not self.available or self.app is None:
            return {"success": False, "error": "InsightFace backend is not available"}

        if image is None or not hasattr(image, "shape") or image.size == 0:
            return {"success": False, "error": "Invalid image for InsightFace backend"}

        try:
            faces = self.app.get(image)
        except Exception as exc:
            logger.warning("InsightFace extraction failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": f"InsightFace extraction failed: {exc}",
                "fallback_allowed": True,
            }
        if not faces:
            return {"success": False, "error": "Không phát hiện khuôn mặt!"}

        face = self._select_largest_face(faces, image.shape)
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return {
            "success": True,
            "embedding": embedding,
            "confidence": round(float(getattr(face, "det_score", 0.0)) * 100, 1),
            "method": "insightface_arcface",
            "bbox": [float(v) for v in getattr(face, "bbox", [])],
        }

    def extract_batch(self, images: list[np.ndarray]) -> list[dict]:
        return [self.extract_embedding(img) for img in images]

    def _select_largest_face(self, faces, image_shape):
        img_h, img_w = image_shape[:2]
        center_x, center_y = img_w / 2.0, img_h / 2.0

        def score(face):
            x1, y1, x2, y2 = face.bbox
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist = float(np.hypot(cx - center_x, cy - center_y))
            return area - (dist * 30.0)

        return max(faces, key=score)


def resize_for_arcface(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    if image is None or not hasattr(image, "shape") or image.size == 0:
        return image

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image

    largest = max(h, w)
    if largest <= max_side:
        return image
    scale = max_side / float(largest)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
