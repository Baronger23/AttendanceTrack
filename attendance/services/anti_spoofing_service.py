"""
Optional ONNX anti-spoofing service.

Set ANTISPOOF_ONNX_MODEL_PATH to enable. Without a model, this service returns
neutral scores so the heuristic liveness pipeline continues to work.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


class AntiSpoofingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.session = None
        self.input_name = None
        self.enabled = False
        self.model_path = getattr(settings, "ANTISPOOF_ONNX_MODEL_PATH", "")
        self.input_size = getattr(settings, "ANTISPOOF_INPUT_SIZE", (80, 80))

        if self.model_path:
            path = Path(self.model_path)
            if path.exists():
                try:
                    import onnxruntime as ort

                    self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    self.input_name = self.session.get_inputs()[0].name
                    self.enabled = True
                    logger.info("Anti-spoofing ONNX loaded: %s", path)
                except Exception as exc:
                    logger.warning("Anti-spoofing ONNX unavailable: %s", exc)
            else:
                logger.warning("Anti-spoofing ONNX path does not exist: %s", path)

        self._initialized = True

    def predict(self, image: np.ndarray, landmarks: list[tuple[float, float]] | None = None) -> dict:
        if not self.enabled or self.session is None:
            return {
                "enabled": False,
                "live_score": 1.0,
                "spoof_score": 0.0,
                "model": "",
            }

        crop = self._crop_face(image, landmarks)
        if crop is None:
            return {
                "enabled": True,
                "live_score": 0.5,
                "spoof_score": 0.5,
                "model": self.model_path,
            }

        tensor = self._preprocess(crop)
        output = self.session.run(None, {self.input_name: tensor})[0]
        scores = self._softmax(np.asarray(output).reshape(-1))

        if len(scores) == 1:
            live_score = float(scores[0])
        else:
            live_index = getattr(settings, "ANTISPOOF_LIVE_CLASS_INDEX", 1)
            live_index = min(max(int(live_index), 0), len(scores) - 1)
            live_score = float(scores[live_index])

        spoof_score = 1.0 - live_score
        return {
            "enabled": True,
            "live_score": round(live_score, 3),
            "spoof_score": round(spoof_score, 3),
            "model": self.model_path,
        }

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor

    def _crop_face(self, image: np.ndarray, landmarks: list[tuple[float, float]] | None):
        if image is None or image.size == 0:
            return None

        if not landmarks:
            return image

        h, w = image.shape[:2]
        points = np.asarray(landmarks, dtype=np.float32)
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1 or bh <= 1:
            return image

        margin = 0.25
        x1 = max(0, int(x1 - bw * margin))
        y1 = max(0, int(y1 - bh * margin))
        x2 = min(w - 1, int(x2 + bw * margin))
        y2 = min(h - 1, int(y2 + bh * margin))
        crop = image[y1:y2, x1:x2]
        return crop if crop.size else None

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        values = values - np.max(values)
        exp = np.exp(values)
        total = np.sum(exp)
        if total <= 0:
            return np.ones_like(exp) / len(exp)
        return exp / total
