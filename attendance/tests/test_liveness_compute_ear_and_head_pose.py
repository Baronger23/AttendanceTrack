"""
Unit tests for LivenessService.compute_ear() and LivenessService.estimate_head_pose()
Task 3: hybrid-liveness-detection spec
"""

import math
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def make_service():
    """Create a LivenessService instance with MediaPipe mocked out."""
    mp_mock = MagicMock()
    mp_mock.solutions.face_mesh.FaceMesh.return_value = MagicMock()
    sys.modules["mediapipe"] = mp_mock
    sys.modules["mediapipe.solutions"] = mp_mock.solutions
    sys.modules["mediapipe.solutions.face_mesh"] = mp_mock.solutions.face_mesh

    from attendance.services.liveness_service import LivenessService

    service = LivenessService.__new__(LivenessService)
    service.face_mesh = MagicMock()
    service.EAR_THRESHOLD = 0.22
    service.EAR_AMPLITUDE_MIN = 0.08
    service.YAW_RANGE_MIN = 8.0
    service.PITCH_RANGE_MIN = 6.0
    service.TEMPORAL_STD_MIN = 0.005
    service.TEMPORAL_STD_MAX = 0.08
    service.timeout_seconds = 8.0
    service.frame_buffer = []
    service.history_size = 30
    service.blink_detected = False
    service.movement_detected = False
    service.consistency_passed = False
    service.session_start = time.time()
    return service


class TestComputeEAR(unittest.TestCase):
    """Tests for LivenessService.compute_ear()

    compute_ear takes:
      - landmarks: list of (x, y) pixel coordinate tuples
      - eye_indices: list of 6 indices into landmarks
    """

    def setUp(self):
        self.service = make_service()

    def _make_eye_landmarks(self, p0, p1, p2, p3, p4, p5):
        """
        Build a landmark list of (x, y) tuples covering indices 0-5.
        Returns (landmarks, eye_indices) where eye_indices = [0,1,2,3,4,5].
        """
        landmarks = [p0, p1, p2, p3, p4, p5]
        return landmarks, [0, 1, 2, 3, 4, 5]

    def test_open_eye_returns_positive_ear(self):
        """Open eye geometry should produce a positive EAR value."""
        # Horizontal span = 1.0, vertical openings = 0.3 each
        # C = 1.0, A = 0.3, B = 0.3 → EAR = (0.3+0.3)/(2*1.0) = 0.3
        landmarks, indices = self._make_eye_landmarks(
            (0.0, 0.0),   # p0 (left corner)
            (0.2, 0.3),   # p1 (top-left)
            (0.8, 0.3),   # p2 (top-right)
            (1.0, 0.0),   # p3 (right corner)
            (0.8, -0.3),  # p4 (bottom-right)
            (0.2, -0.3),  # p5 (bottom-left)
        )
        ear = self.service.compute_ear(landmarks, indices)
        self.assertGreater(ear, 0.0)

    def test_known_geometry_exact_value(self):
        """
        Verify exact EAR calculation with known geometry.
        p0=(0,0), p3=(1,0) → C=1.0
        p1=(0.2,0.3), p5=(0.2,-0.3) → A=0.6
        p2=(0.8,0.3), p4=(0.8,-0.3) → B=0.6
        EAR = (0.6+0.6)/(2*1.0) = 0.6
        """
        landmarks, indices = self._make_eye_landmarks(
            (0.0, 0.0),
            (0.2, 0.3),
            (0.8, 0.3),
            (1.0, 0.0),
            (0.8, -0.3),
            (0.2, -0.3),
        )
        ear = self.service.compute_ear(landmarks, indices)
        self.assertAlmostEqual(ear, 0.6, places=6)

    def test_closed_eye_low_ear(self):
        """Closed eye (vertical distances near 0) should produce near-zero EAR."""
        # Horizontal span = 1.0, vertical openings ≈ 0
        landmarks, indices = self._make_eye_landmarks(
            (0.0, 0.0),
            (0.2, 0.001),
            (0.8, 0.001),
            (1.0, 0.0),
            (0.8, -0.001),
            (0.2, -0.001),
        )
        ear = self.service.compute_ear(landmarks, indices)
        self.assertLess(ear, 0.01)

    def test_zero_horizontal_distance_returns_zero(self):
        """When C == 0 (p0 == p3), should return 0.0 to avoid division by zero."""
        landmarks, indices = self._make_eye_landmarks(
            (0.5, 0.5),   # p0
            (0.5, 0.8),   # p1
            (0.5, 0.8),   # p2
            (0.5, 0.5),   # p3 — same as p0 → C = 0
            (0.5, 0.2),   # p4
            (0.5, 0.2),   # p5
        )
        ear = self.service.compute_ear(landmarks, indices)
        self.assertEqual(ear, 0.0)

    def test_returns_float(self):
        """compute_ear() must return a Python float."""
        landmarks, indices = self._make_eye_landmarks(
            (0.0, 0.0), (0.2, 0.3), (0.8, 0.3),
            (1.0, 0.0), (0.8, -0.3), (0.2, -0.3),
        )
        ear = self.service.compute_ear(landmarks, indices)
        self.assertIsInstance(ear, float)

    def test_uses_correct_index_mapping(self):
        """
        Verify that the method uses eye_indices correctly (not just 0-5 directly).
        Place meaningful landmarks at non-zero indices.
        """
        # Build a larger landmark list; eye_indices will point into it
        # indices = [10, 11, 12, 13, 14, 15]
        # p0=lm[10]=(0,0), p3=lm[13]=(1,0) → C=1.0
        # p1=lm[11]=(0.2,0.4), p5=lm[15]=(0.2,-0.4) → A=0.8
        # p2=lm[12]=(0.8,0.4), p4=lm[14]=(0.8,-0.4) → B=0.8
        # EAR = (0.8+0.8)/(2*1.0) = 0.8
        landmarks = [(0.0, 0.0)] * 16
        landmarks[10] = (0.0, 0.0)
        landmarks[11] = (0.2, 0.4)
        landmarks[12] = (0.8, 0.4)
        landmarks[13] = (1.0, 0.0)
        landmarks[14] = (0.8, -0.4)
        landmarks[15] = (0.2, -0.4)
        eye_indices = [10, 11, 12, 13, 14, 15]
        ear = self.service.compute_ear(landmarks, eye_indices)
        self.assertAlmostEqual(ear, 0.8, places=6)

    def test_symmetric_eye_ear_formula(self):
        """
        EAR formula: (A + B) / (2 * C).
        p0=(0,0), p3=(0.5,0) → C=0.5
        p1=(0.1,0.2), p5=(0.1,-0.2) → A=0.4
        p2=(0.4,0.2), p4=(0.4,-0.2) → B=0.4
        EAR = (0.4+0.4)/(2*0.5) = 0.8
        """
        landmarks, indices = self._make_eye_landmarks(
            (0.0, 0.0),
            (0.1, 0.2),
            (0.4, 0.2),
            (0.5, 0.0),
            (0.4, -0.2),
            (0.1, -0.2),
        )
        ear = self.service.compute_ear(landmarks, indices)
        self.assertAlmostEqual(ear, 0.8, places=6)


class TestEstimateHeadPose(unittest.TestCase):
    """Tests for LivenessService.estimate_head_pose()

    estimate_head_pose takes:
      - image: np.ndarray
      - face_landmarks: list of (x, y) pixel coordinate tuples for all 468 landmarks
    """

    def setUp(self):
        self.service = make_service()

    def _make_face_landmarks_pixels(self, image_shape, overrides=None):
        """
        Build a list of 468 (x, y) pixel coordinate tuples.
        Default: all landmarks at image center.
        overrides: dict of {index: (x_pixel, y_pixel)} to set specific landmarks.
        """
        h, w = image_shape[:2]
        cx, cy = w / 2.0, h / 2.0
        landmarks = [(cx, cy)] * 468
        if overrides:
            for idx, (x, y) in overrides.items():
                landmarks[idx] = (x, y)
        return landmarks

    def test_returns_tuple_of_three_floats(self):
        """estimate_head_pose() must return a tuple of 3 floats."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        # Use a realistic frontal face landmark configuration (pixel coords)
        overrides = {
            1:   (0.50 * w, 0.55 * h),   # nose tip
            152: (0.50 * w, 0.80 * h),   # chin
            33:  (0.35 * w, 0.45 * h),   # left eye corner
            263: (0.65 * w, 0.45 * h),   # right eye corner
            61:  (0.42 * w, 0.70 * h),   # left mouth
            291: (0.58 * w, 0.70 * h),   # right mouth
        }
        landmarks = self._make_face_landmarks_pixels(image.shape, overrides)
        result = self.service.estimate_head_pose(image, landmarks)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        for val in result:
            self.assertIsInstance(val, float)

    def test_returns_fallback_on_degenerate_landmarks(self):
        """
        When all landmarks are at the same point (degenerate), solvePnP may fail
        or produce garbage — the method should return gracefully.
        """
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # All landmarks at the exact same pixel → degenerate 2D points
        landmarks = self._make_face_landmarks_pixels(image.shape)
        result = self.service.estimate_head_pose(image, landmarks)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        # Should not raise; may return (0,0,0) or valid angles
        for val in result:
            self.assertIsInstance(val, float)

    def test_returns_fallback_on_exception(self):
        """
        If cv2 raises an unexpected exception, the method must return (0.0, 0.0, 0.0).
        """
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = self._make_face_landmarks_pixels(image.shape)

        with patch("cv2.solvePnP", side_effect=RuntimeError("cv2 error")):
            result = self.service.estimate_head_pose(image, landmarks)

        self.assertEqual(result, (0.0, 0.0, 0.0))

    def test_returns_fallback_when_solvepnp_fails(self):
        """When cv2.solvePnP returns success=False, return (0.0, 0.0, 0.0)."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = self._make_face_landmarks_pixels(image.shape)

        with patch("cv2.solvePnP", return_value=(False, None, None)):
            result = self.service.estimate_head_pose(image, landmarks)

        self.assertEqual(result, (0.0, 0.0, 0.0))

    def test_uses_image_dimensions_for_pixel_coords(self):
        """
        Verify that pixel coordinates are passed directly to solvePnP.
        Different image sizes with proportionally scaled pixel landmarks
        should call solvePnP with proportionally scaled pixel coordinates.
        """
        # Normalized positions
        norm_overrides = {
            1:   (0.50, 0.55),
            152: (0.50, 0.80),
            33:  (0.35, 0.45),
            263: (0.65, 0.45),
            61:  (0.42, 0.70),
            291: (0.58, 0.70),
        }

        image_small = np.zeros((240, 320, 3), dtype=np.uint8)
        image_large = np.zeros((480, 640, 3), dtype=np.uint8)

        h_s, w_s = image_small.shape[:2]
        h_l, w_l = image_large.shape[:2]

        # Build pixel landmarks for each image size
        overrides_small = {idx: (nx * w_s, ny * h_s) for idx, (nx, ny) in norm_overrides.items()}
        overrides_large = {idx: (nx * w_l, ny * h_l) for idx, (nx, ny) in norm_overrides.items()}

        landmarks_small = self._make_face_landmarks_pixels(image_small.shape, overrides_small)
        landmarks_large = self._make_face_landmarks_pixels(image_large.shape, overrides_large)

        captured_calls = []

        def mock_solvepnp(obj_pts, img_pts, cam_mat, dist):
            captured_calls.append(img_pts.copy())
            return (False, None, None)

        with patch("cv2.solvePnP", side_effect=mock_solvepnp):
            self.service.estimate_head_pose(image_small, landmarks_small)
            self.service.estimate_head_pose(image_large, landmarks_large)

        self.assertEqual(len(captured_calls), 2)
        # Pixel coords for large image should be 2x those of small image
        np.testing.assert_allclose(
            captured_calls[1], captured_calls[0] * 2.0, rtol=1e-5
        )

    def test_frontal_face_returns_near_zero_angles(self):
        """
        A perfectly frontal face should produce pitch/yaw/roll close to 0.
        This is a sanity check using real cv2 (not mocked).
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not available")

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        from attendance.services.liveness_service import PNP_3D_POINTS, PNP_LANDMARKS

        h, w = image.shape[:2]
        focal = float(w)
        cx, cy = w / 2.0, h / 2.0

        # Project 3D points to 2D using identity rotation and z=600 translation
        z_offset = 600.0
        projected_pixels = []
        for pt in PNP_3D_POINTS:
            px = (pt[0] * focal / (pt[2] + z_offset)) + cx
            py = (pt[1] * focal / (pt[2] + z_offset)) + cy
            projected_pixels.append((px, py))

        pnp_order = [
            PNP_LANDMARKS["nose_tip"],
            PNP_LANDMARKS["chin"],
            PNP_LANDMARKS["left_eye_corner"],
            PNP_LANDMARKS["right_eye_corner"],
            PNP_LANDMARKS["left_mouth"],
            PNP_LANDMARKS["right_mouth"],
        ]

        landmarks = self._make_face_landmarks_pixels(image.shape)
        for i, idx in enumerate(pnp_order):
            landmarks[idx] = projected_pixels[i]

        pitch, yaw, roll = self.service.estimate_head_pose(image, landmarks)
        # Angles should be small (within ±15 degrees) for a near-frontal pose
        self.assertLess(abs(pitch), 15.0)
        self.assertLess(abs(yaw), 15.0)


if __name__ == "__main__":
    unittest.main()
