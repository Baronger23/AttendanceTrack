"""
Hybrid Liveness Detection Service

3-layer hybrid liveness detection:
  - Layer 1: Blink Detection (EAR Amplitude)
  - Layer 2: Head Movement (PnP / Perspective-n-Point)
  - Layer 3: Temporal Consistency (EAR std-dev + EAR-yaw correlation)
"""

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark index constants
# ---------------------------------------------------------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 6 MediaPipe landmark indices used for PnP head-pose estimation
PNP_LANDMARKS = {
    "nose_tip":         1,
    "chin":             152,
    "left_eye_corner":  33,
    "right_eye_corner": 263,
    "left_mouth":       61,
    "right_mouth":      291,
}

# Corresponding 3-D model points (mm, generic face model)
PNP_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye corner
        (225.0, 170.0, -135.0),    # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),   # Right mouth corner
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# LivenessService
# ---------------------------------------------------------------------------


class LivenessService:
    """
    Hybrid liveness detection service.

    Each KioskConsumer creates its own instance — no singleton.
    Public interface: update(image) -> dict, reset() -> None.
    """

    def __init__(self):
        # Lazy import of mediapipe so Django startup is not slowed down when
        # the library is absent.
        self.disabled = False
        try:
            import mediapipe as mp  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "mediapipe is required. Run: pip install mediapipe"
            )

        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except AttributeError:
            logger.warning(
                "MediaPipe is installed without mp.solutions; liveness checks are disabled."
            )
            self.face_mesh = None
            self.disabled = True

        # Read thresholds from Django settings (with safe defaults)
        from django.conf import settings  # noqa: PLC0415

        self.EAR_THRESHOLD = getattr(settings, "LIVENESS_EAR_THRESHOLD", 0.22)
        self.EAR_AMPLITUDE_MIN = getattr(settings, "LIVENESS_EAR_AMPLITUDE_MIN", 0.08)
        self.YAW_RANGE_MIN = getattr(settings, "LIVENESS_YAW_RANGE_MIN", 8.0)
        self.PITCH_RANGE_MIN = getattr(settings, "LIVENESS_PITCH_RANGE_MIN", 6.0)
        self.TEMPORAL_STD_MIN = getattr(settings, "LIVENESS_TEMPORAL_STD_MIN", 0.005)
        self.TEMPORAL_STD_MAX = getattr(settings, "LIVENESS_TEMPORAL_STD_MAX", 0.08)
        self.timeout_seconds = getattr(settings, "LIVENESS_TIMEOUT_SECONDS", 8.0)
        self.MIN_PASS_FRAMES = getattr(settings, "LIVENESS_MIN_PASS_FRAMES", 8)
        self.PASS_SCORE_THRESHOLD = getattr(settings, "LIVENESS_PASS_SCORE_THRESHOLD", 0.52)
        self.REPLAY_RISK_THRESHOLD = getattr(settings, "LIVENESS_REPLAY_RISK_THRESHOLD", 0.72)
        self.ANTISPOOF_SPOOF_THRESHOLD = getattr(settings, "ANTISPOOF_SPOOF_THRESHOLD", 0.65)

        # Session state
        self.frame_buffer: list = []
        self.history_size: int = 30
        self.blink_detected: bool = False
        self.movement_detected: bool = False
        self.consistency_passed: bool = False
        self.session_start: float = time.time()

        # Anti-spoofing and Replay detectors (In-Memory Only)
        from .anti_spoofing_service import AntiSpoofingService
        from .replay_attack_detector import ReplayAttackDetector
        self.anti_spoofing = AntiSpoofingService()
        self.replay_detector = ReplayAttackDetector()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all session state and prepare for a new liveness session."""
        self.frame_buffer = []
        self.blink_detected = False
        self.movement_detected = False
        self.consistency_passed = False
        self.session_start = time.time()
        if hasattr(self, "replay_detector"):
            self.replay_detector.reset()

    # ------------------------------------------------------------------
    # Stub methods — implemented in Tasks 3-6
    # ------------------------------------------------------------------

    def update(self, image: np.ndarray) -> dict:
        """
        Process a new frame and return the current liveness status.

        Returns:
            {
                "status": "pending" | "passed" | "failed",
                "feedback_ui": str
            }
        """
        try:
            if self.disabled:
                return {
                    "status": "passed",
                    "feedback_ui": "Kiểm tra liveness tạm thời được bỏ qua trên môi trường này.",
                }

            # 1. Timeout check
            if time.time() - self.session_start > self.timeout_seconds:
                self.reset()
                return {"status": "failed", "feedback_ui": "Hết thời gian, vui lòng thử lại."}

            # 2. Run MediaPipe FaceMesh on RGB frame
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            if not results or not results.multi_face_landmarks:
                return {"status": "pending", "feedback_ui": "Không tìm thấy khuôn mặt rõ ràng."}

            # 3. Get first face landmarks, convert to pixel coordinates
            face_landmarks_raw = results.multi_face_landmarks[0].landmark
            h, w = image.shape[:2]
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks_raw]

            # Run Anti-Spoofing and Replay Attack checks
            anti_spoof = self.anti_spoofing.predict(image, landmarks)
            if anti_spoof.get("spoof_score", 0.0) >= self.ANTISPOOF_SPOOF_THRESHOLD:
                logger.warning(
                    "Anti-spoofing model rejected frame: live=%.3f spoof=%.3f",
                    anti_spoof.get("live_score", 0.0),
                    anti_spoof.get("spoof_score", 0.0),
                )
                self.reset()
                return {
                    "status": "failed",
                    "feedback_ui": "Phát hiện khả năng giả mạo khuôn mặt, vui lòng thử lại trực tiếp trước camera.",
                }

            replay_risk = self.replay_detector.update(image, landmarks)
            if replay_risk.score >= self.REPLAY_RISK_THRESHOLD or replay_risk.suspicious:
                logger.warning(
                    "Replay attack suspected: score=%.3f phone_frame=%.3f planar=%.3f screen=%.3f",
                    replay_risk.score,
                    replay_risk.phone_frame,
                    replay_risk.planar_motion,
                    replay_risk.screen_artifact,
                )
                self.reset()
                return {
                    "status": "failed",
                    "feedback_ui": "Phát hiện khả năng dùng màn hình điện thoại, vui lòng dùng khuôn mặt thật trước camera.",
                }

            # 4. Compute EAR for both eyes
            left_ear = self.compute_ear(landmarks, LEFT_EYE)
            right_ear = self.compute_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # 5. Estimate head pose
            pitch, yaw, roll = self.estimate_head_pose(image, landmarks)

            # 6. Append to frame buffer, cap at history_size
            self.frame_buffer.append({
                "timestamp": time.time(),
                "ear": avg_ear,
                "pitch": pitch,
                "yaw": yaw,
            })
            if len(self.frame_buffer) > self.history_size:
                self.frame_buffer.pop(0)

            # 7. Layer 1: Blink detection (when buffer >= 5 and not yet detected)
            if len(self.frame_buffer) >= 5 and not self.blink_detected:
                ears = [f["ear"] for f in self.frame_buffer]
                ear_range = max(ears) - min(ears)
                if ear_range > self.EAR_AMPLITUDE_MIN and min(ears) < self.EAR_THRESHOLD:
                    self.blink_detected = True

            # 8. Layer 2: Head movement detection (when buffer >= 5 and not yet detected)
            if len(self.frame_buffer) >= 5 and not self.movement_detected:
                yaws = [f["yaw"] for f in self.frame_buffer]
                pitches = [f["pitch"] for f in self.frame_buffer]
                yaw_range = max(yaws) - min(yaws)
                pitch_range = max(pitches) - min(pitches)
                if yaw_range > self.YAW_RANGE_MIN or pitch_range > self.PITCH_RANGE_MIN:
                    self.movement_detected = True

            # Spoofing check (buffer >= 15)
            if len(self.frame_buffer) >= 15:
                if self._check_spoofing_correlation():
                    ears = [f["ear"] for f in self.frame_buffer]
                    yaws = [f["yaw"] for f in self.frame_buffer]
                    corr = np.corrcoef(ears, yaws)[0, 1]
                    session_duration = time.time() - self.session_start
                    logger.warning(
                        f"Spoofing detected: corr={corr:.3f}, session_duration={session_duration:.1f}s"
                    )
                    self.reset()
                    return {"status": "failed", "feedback_ui": "Phát hiện hành vi bất thường, vui lòng thử lại tự nhiên."}

            # Temporal consistency check (buffer >= 10)
            if len(self.frame_buffer) >= 10:
                self.consistency_passed = self._check_temporal_consistency()
                if not self.consistency_passed and self.blink_detected and self.movement_detected:
                    self.reset()
                    return {"status": "failed", "feedback_ui": "Phát hiện hành vi bất thường, vui lòng thử lại tự nhiên."}
            else:
                self.consistency_passed = False

            # Hybrid decision: all 3 layers must pass
            liveness_score = self._active_liveness_score()

            enough_frames = len(self.frame_buffer) >= self.MIN_PASS_FRAMES
            strong_active_signal = self.blink_detected or self.movement_detected
            score_passed = liveness_score >= self.PASS_SCORE_THRESHOLD
            classic_passed = self.blink_detected and self.movement_detected and self.consistency_passed

            if enough_frames and strong_active_signal and (score_passed or classic_passed):
                return {"status": "passed", "feedback_ui": "Xác thực thành công!"}

            # Dynamic feedback for pending state
            feedback = []
            if not self.blink_detected:
                feedback.append("chớp mắt")
            if not self.movement_detected:
                feedback.append("quay đầu nhẹ")
            if len(self.frame_buffer) >= 10 and not self.consistency_passed:
                feedback.append("giữ nguyên tư thế")
            if not feedback:
                feedback.append("giữ nguyên tư thế")
            return {"status": "pending", "feedback_ui": "Vui lòng " + " và ".join(feedback)}

        except Exception as e:
            logger.error(f"Unexpected error in LivenessService.update(): {e}")
            return {"status": "pending", "feedback_ui": "Lỗi xử lý frame, vui lòng thử lại."}

    def _active_liveness_score(self) -> float:
        """Combine active challenge signals into a forgiving live-user score."""
        if not self.frame_buffer:
            return 0.0

        ears = [f["ear"] for f in self.frame_buffer]
        yaws = [f["yaw"] for f in self.frame_buffer]
        pitches = [f["pitch"] for f in self.frame_buffer]

        ear_range = max(ears) - min(ears)
        yaw_range = max(yaws) - min(yaws)
        pitch_range = max(pitches) - min(pitches)
        movement_range = max(
            yaw_range / max(self.YAW_RANGE_MIN, 1e-6),
            pitch_range / max(self.PITCH_RANGE_MIN, 1e-6),
        )

        blink_score = 1.0 if self.blink_detected else float(np.clip(ear_range / max(self.EAR_AMPLITUDE_MIN, 1e-6), 0.0, 1.0))
        movement_score = 1.0 if self.movement_detected else float(np.clip(movement_range, 0.0, 1.0))
        consistency_score = 1.0 if self.consistency_passed else 0.0
        frame_score = float(np.clip(len(self.frame_buffer) / max(self.MIN_PASS_FRAMES, 1), 0.0, 1.0))

        score = (0.42 * blink_score) + (0.42 * movement_score) + (0.10 * consistency_score) + (0.06 * frame_score)
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    def compute_ear(self, landmarks: list, eye_indices: list) -> float:
        """
        Compute Eye Aspect Ratio from 6 landmark points.

        Args:
            landmarks: list of (x, y) pixel coordinate tuples
            eye_indices: list of 6 indices into landmarks

        Formula: (A + B) / (2 * C)
        where A, B are vertical distances and C is the horizontal distance.
        Returns 0.0 if C == 0.
        """
        p0 = landmarks[eye_indices[0]]
        p1 = landmarks[eye_indices[1]]
        p2 = landmarks[eye_indices[2]]
        p3 = landmarks[eye_indices[3]]
        p4 = landmarks[eye_indices[4]]
        p5 = landmarks[eye_indices[5]]

        # A: vertical distance between indices[1] and indices[5]
        A = np.linalg.norm(np.array([p1[0] - p5[0], p1[1] - p5[1]]))
        # B: vertical distance between indices[2] and indices[4]
        B = np.linalg.norm(np.array([p2[0] - p4[0], p2[1] - p4[1]]))
        # C: horizontal distance between indices[0] and indices[3]
        C = np.linalg.norm(np.array([p0[0] - p3[0], p0[1] - p3[1]]))

        if C == 0:
            return 0.0

        return float((A + B) / (2.0 * C))

    def estimate_head_pose(
        self, image: np.ndarray, face_landmarks: list
    ) -> tuple:
        """
        Estimate head rotation angles (pitch, yaw, roll) via cv2.solvePnP.

        Args:
            image: full image as np.ndarray
            face_landmarks: list of (x, y) pixel coordinate tuples for all 468 landmarks

        Returns:
            (pitch, yaw, roll) as floats in degrees, or (0.0, 0.0, 0.0) on failure.
        """
        try:
            import cv2  # noqa: PLC0415

            h, w = image.shape[:2]

            # Extract 2D image points for the 6 PnP landmarks (in order)
            pnp_indices = [
                PNP_LANDMARKS["nose_tip"],
                PNP_LANDMARKS["chin"],
                PNP_LANDMARKS["left_eye_corner"],
                PNP_LANDMARKS["right_eye_corner"],
                PNP_LANDMARKS["left_mouth"],
                PNP_LANDMARKS["right_mouth"],
            ]
            image_points = np.array(
                [
                    (face_landmarks[idx][0], face_landmarks[idx][1])
                    for idx in pnp_indices
                ],
                dtype=np.float64,
            )

            # Build camera matrix
            focal_length = float(w)
            camera_matrix = np.array(
                [
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vec, translation_vec = cv2.solvePnP(
                PNP_3D_POINTS, image_points, camera_matrix, dist_coeffs
            )

            if not success:
                return (0.0, 0.0, 0.0)

            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # Extract Euler angles via RQDecomp3x3
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
            pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])

            return (pitch, yaw, roll)

        except Exception:
            return (0.0, 0.0, 0.0)

    def _check_temporal_consistency(self) -> bool:
        """
        Layer 3: Check whether EAR std-dev is within [STD_MIN, STD_MAX].
        Should only be called when frame_buffer has >= 10 entries.
        """
        std = np.std([f["ear"] for f in self.frame_buffer])
        return bool(self.TEMPORAL_STD_MIN <= std <= self.TEMPORAL_STD_MAX)

    def _check_spoofing_correlation(self) -> bool:
        """
        Anti-spoofing: Check EAR-yaw correlation.
        Should only be called when frame_buffer has >= 15 entries.

        Returns True if spoofing is detected (|corr| > 0.95).
        """
        ears = [f["ear"] for f in self.frame_buffer]
        yaws = [f["yaw"] for f in self.frame_buffer]

        # Correlation is undefined when std of either series is 0
        if np.std(ears) == 0 or np.std(yaws) == 0:
            return False

        corr = np.corrcoef(ears, yaws)[0, 1]
        return bool(abs(corr) > 0.95)
