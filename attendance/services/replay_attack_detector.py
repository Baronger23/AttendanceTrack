"""
Replay attack heuristics for webcam-only liveness.

These checks target the common "phone close to camera" attack:
  - a rectangular screen frame around the face,
  - face landmarks moving like one flat image,
  - display artifacts such as moire, banding, or flat brightness.

The output is a risk score, not a biometric decision by itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


TRACKED_LANDMARKS = [1, 10, 33, 61, 152, 199, 234, 263, 291, 454]


@dataclass
class ReplayRisk:
    phone_frame: float = 0.0
    planar_motion: float = 0.0
    screen_artifact: float = 0.0
    score: float = 0.0

    @property
    def suspicious(self) -> bool:
        return self.score >= 0.72 or self.phone_frame >= 0.86


class ReplayAttackDetector:
    """Heuristic replay detector for phone/screen presentation attacks."""

    def __init__(self, history_size: int = 12):
        self.history_size = history_size
        self.landmark_buffer: list[np.ndarray] = []
        self.bbox_buffer: list[tuple[int, int, int, int]] = []

    def reset(self) -> None:
        self.landmark_buffer = []
        self.bbox_buffer = []

    def update(self, image: np.ndarray, landmarks: list[tuple[float, float]]) -> ReplayRisk:
        bbox = self._face_bbox(image.shape, landmarks)
        sampled = self._sample_landmarks(landmarks)

        if sampled is not None:
            self.landmark_buffer.append(sampled)
            if len(self.landmark_buffer) > self.history_size:
                self.landmark_buffer.pop(0)

        if bbox is not None:
            self.bbox_buffer.append(bbox)
            if len(self.bbox_buffer) > self.history_size:
                self.bbox_buffer.pop(0)

        phone_frame = self.phone_frame_score(image, bbox)
        planar_motion = self.planar_motion_score()
        screen_artifact = self.screen_artifact_score(image, bbox)

        score = (0.42 * phone_frame) + (0.38 * planar_motion) + (0.20 * screen_artifact)
        return ReplayRisk(
            phone_frame=round(float(phone_frame), 3),
            planar_motion=round(float(planar_motion), 3),
            screen_artifact=round(float(screen_artifact), 3),
            score=round(float(score), 3),
        )

    def phone_frame_score(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> float:
        """Detect a large rectangular phone/screen frame enclosing the face."""
        if bbox is None or image is None or image.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 60, 160)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as exc:
            logger.debug("phone_frame_score skipped: %s", exc)
            return 0.0

        x1, y1, x2, y2 = bbox
        face_area = max(1, (x2 - x1) * (y2 - y1))
        img_h, img_w = image.shape[:2]
        best = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < face_area * 1.05 or area > img_h * img_w * 0.92:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            if len(approx) != 4:
                continue

            rx, ry, rw, rh = cv2.boundingRect(approx)
            if rw <= 0 or rh <= 0:
                continue

            contains_face = rx <= x1 and ry <= y1 and rx + rw >= x2 and ry + rh >= y2
            if not contains_face:
                continue

            aspect = rw / float(rh)
            aspect_score = 1.0 if 0.42 <= aspect <= 2.45 else 0.35
            area_ratio = area / float(face_area)
            ratio_score = float(np.clip((area_ratio - 1.1) / 5.0, 0.0, 1.0))
            rectangularity = float(np.clip(area / float(rw * rh), 0.0, 1.0))

            candidate = (0.40 * aspect_score) + (0.35 * ratio_score) + (0.25 * rectangularity)
            best = max(best, candidate)

        return float(np.clip(best, 0.0, 1.0))

    def planar_motion_score(self) -> float:
        """Score whether landmarks move like one flat 2D plane."""
        if len(self.landmark_buffer) < 4 or len(self.bbox_buffer) < 4:
            return 0.0

        pair_scores = []
        for prev, curr, bbox in zip(
            self.landmark_buffer[:-1],
            self.landmark_buffer[1:],
            self.bbox_buffer[1:],
        ):
            try:
                matrix, inliers = cv2.estimateAffinePartial2D(
                    prev.astype(np.float32),
                    curr.astype(np.float32),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                )
            except Exception:
                continue

            if matrix is None:
                continue

            try:
                predicted = cv2.transform(prev.reshape(1, -1, 2).astype(np.float32), matrix)[0]
                residual = np.linalg.norm(predicted - curr, axis=1)
                displacement = np.linalg.norm(curr - prev, axis=1)

                x1, y1, x2, y2 = bbox
                face_diag = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
                residual_ratio = float(np.median(residual) / face_diag)
                displacement_ratio = float(np.median(displacement) / face_diag)
            except Exception:
                continue

            if displacement_ratio < 0.012:
                continue

            flatness = float(np.clip((0.025 - residual_ratio) / 0.025, 0.0, 1.0))
            motion = float(np.clip((displacement_ratio - 0.012) / 0.060, 0.0, 1.0))
            pair_scores.append(flatness * motion)

        if not pair_scores:
            return 0.0
        return float(np.clip(np.mean(pair_scores), 0.0, 1.0))

    def screen_artifact_score(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> float:
        """Detect display-like texture: flat brightness plus regular high-frequency energy."""
        crop = self._crop_face(image, bbox)
        if crop is None:
            return 0.0

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        except Exception as exc:
            logger.debug("screen_artifact_score skipped: %s", exc)
            return 0.0

        gray_float = gray.astype(np.float32)
        mean = float(np.mean(gray_float))
        std = float(np.std(gray_float))
        if mean <= 1:
            return 0.0

        uniformity_score = float(np.clip((42.0 - std) / 42.0, 0.0, 1.0))

        centered = gray_float - mean
        spectrum = np.fft.fftshift(np.fft.fft2(centered))
        magnitude = np.log1p(np.abs(spectrum))
        h, w = magnitude.shape
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        high = magnitude[radius > 34]
        mid = magnitude[(radius > 10) & (radius <= 34)]
        high_ratio = float(np.mean(high) / (np.mean(mid) + 1e-6))
        moire_score = float(np.clip((high_ratio - 0.82) / 0.55, 0.0, 1.0))

        row_means = np.mean(gray_float, axis=1)
        banding_score = float(np.clip(np.std(np.diff(row_means)) / 4.0, 0.0, 1.0))

        return float(np.clip((0.35 * uniformity_score) + (0.45 * moire_score) + (0.20 * banding_score), 0.0, 1.0))

    def _sample_landmarks(self, landmarks: list[tuple[float, float]]) -> np.ndarray | None:
        if len(landmarks) <= max(TRACKED_LANDMARKS):
            return None
        return np.array([landmarks[idx] for idx in TRACKED_LANDMARKS], dtype=np.float32)

    def _face_bbox(
        self,
        image_shape: tuple[int, ...],
        landmarks: list[tuple[float, float]],
        margin: float = 0.18,
    ) -> tuple[int, int, int, int] | None:
        if not landmarks:
            return None

        h, w = image_shape[:2]
        points = np.array(landmarks, dtype=np.float32)
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1 or bh <= 1:
            return None

        x1 -= bw * margin
        x2 += bw * margin
        y1 -= bh * margin
        y2 += bh * margin
        return (
            max(0, int(x1)),
            max(0, int(y1)),
            min(w - 1, int(x2)),
            min(h - 1, int(y2)),
        )

    def _crop_face(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray | None:
        if bbox is None or image is None or image.size == 0:
            return None
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop
