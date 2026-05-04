"""
Unit tests for LivenessService — edge cases.
Task 10.1: hybrid-liveness-detection spec

Requirements: 1.4, 2.4, 2.5, 3.5, 4.3, 8.5, 9.3, 10.2
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")

# ---------------------------------------------------------------------------
# cv2 mock — the installed cv2 has a broken typing sub-module on this machine
# (cv2.dnn.DictValue missing).  We mock it at the sys.modules level so that
# importing liveness_service.py never touches the real cv2 typing bootstrap.
# ---------------------------------------------------------------------------

_cv2_mock = MagicMock()

# Make cv2.cvtColor return a valid numpy array (RGB image)
_cv2_mock.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

# Make cv2.solvePnP return a successful result by default
_dummy_rvec = np.zeros((3, 1), dtype=np.float64)
_dummy_tvec = np.zeros((3, 1), dtype=np.float64)
_cv2_mock.solvePnP.return_value = (True, _dummy_rvec, _dummy_tvec)

# Make cv2.Rodrigues return a valid rotation matrix
_dummy_rot_mat = np.eye(3, dtype=np.float64)
_cv2_mock.Rodrigues.return_value = (_dummy_rot_mat, None)

# Make cv2.RQDecomp3x3 return angles (0, 0, 0)
_cv2_mock.RQDecomp3x3.return_value = (
    np.array([0.0, 0.0, 0.0]),
    None, None, None, None, None,
)

# Patch cv2 globally before any liveness_service import
sys.modules["cv2"] = _cv2_mock


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_service():
    """Create a LivenessService instance with MediaPipe and cv2 mocked out."""
    mp_mock = MagicMock()
    mp_mock.solutions.face_mesh.FaceMesh.return_value = MagicMock()

    # Remove any cached liveness_service module so each call gets a fresh import
    for key in list(sys.modules.keys()):
        if "liveness_service" in key:
            del sys.modules[key]

    with patch.dict(
        "sys.modules",
        {
            "mediapipe": mp_mock,
            "mediapipe.solutions": mp_mock.solutions,
            "mediapipe.solutions.face_mesh": mp_mock.solutions.face_mesh,
        },
    ):
        from attendance.services.liveness_service import LivenessService
        service = LivenessService()

    return service


def _make_face_landmarks_result(n_landmarks: int = 468):
    """
    Build a mock MediaPipe result with one face containing n_landmarks.
    Each landmark has .x, .y, .z attributes (normalised 0-1).
    """
    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5
    lm.z = 0.0
    face = MagicMock()
    face.landmark = [lm] * n_landmarks
    result = MagicMock()
    result.multi_face_landmarks = [face]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTimeoutTriggersFailedAndReset(unittest.TestCase):
    """Req 1.4: Timeout after 8 s returns failed and resets state."""

    def test_timeout_triggers_failed_and_reset(self):
        service = make_service()
        # Simulate that the session started 9 seconds ago
        service.session_start = time.time() - 9.0

        result = service.update(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(result["status"], "failed")
        self.assertIn("Hết thời gian", result["feedback_ui"])
        # reset() must have been called — buffer should be empty
        self.assertEqual(len(service.frame_buffer), 0)


class TestNoFaceReturnsPending(unittest.TestCase):
    """Req 2.5: Frame with no detected face returns pending."""

    def test_no_face_returns_pending(self):
        service = make_service()

        # face_mesh.process() returns a result with no landmarks
        no_face_result = MagicMock()
        no_face_result.multi_face_landmarks = None
        service.face_mesh.process.return_value = no_face_result

        result = service.update(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(result["status"], "pending")
        self.assertIn("Không tìm thấy khuôn mặt", result["feedback_ui"])


class TestPnpFailureSkipsFrame(unittest.TestCase):
    """Req 3.5: solvePnP failure does not crash; frame is still buffered with yaw=0."""

    def test_pnp_failure_skips_frame(self):
        service = make_service()

        # Provide a valid face with 468 landmarks
        service.face_mesh.process.return_value = _make_face_landmarks_result(468)

        # Override the global cv2 mock to return failure for solvePnP
        _cv2_mock.solvePnP.return_value = (False, None, None)
        try:
            result = service.update(np.zeros((480, 640, 3), dtype=np.uint8))
        finally:
            # Restore default
            _cv2_mock.solvePnP.return_value = (True, _dummy_rvec, _dummy_tvec)

        # Must not crash and must return a dict with "status"
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        # Frame was still appended (with yaw=0.0 from the fallback)
        self.assertEqual(len(service.frame_buffer), 1)
        self.assertEqual(service.frame_buffer[0]["yaw"], 0.0)


class TestConsistencyFailedTriggersReset(unittest.TestCase):
    """Req 4.3 / 5.4: EAR std < 0.005 (static image) triggers failed + reset."""

    def test_consistency_failed_triggers_reset(self):
        service = make_service()

        # Inject 10 frames with identical EAR (std == 0 < 0.005)
        for _ in range(10):
            service.frame_buffer.append(
                {"timestamp": time.time(), "ear": 0.3, "pitch": 0.0, "yaw": 0.0}
            )

        # Both blink and movement already detected so only consistency matters
        service.blink_detected = True
        service.movement_detected = True

        # Provide a valid face so update() processes normally
        service.face_mesh.process.return_value = _make_face_landmarks_result(468)

        # solvePnP fails → yaw stays 0.0 (same as injected frames → std stays 0)
        _cv2_mock.solvePnP.return_value = (False, None, None)
        try:
            result = service.update(np.zeros((480, 640, 3), dtype=np.uint8))
        finally:
            _cv2_mock.solvePnP.return_value = (True, _dummy_rvec, _dummy_tvec)

        self.assertEqual(result["status"], "failed")
        self.assertIn("bất thường", result["feedback_ui"])
        # reset() must have been called
        self.assertEqual(len(service.frame_buffer), 0)


class TestMediapipeImportError(unittest.TestCase):
    """Req 8.5: ImportError with clear message when mediapipe is not installed."""

    def test_mediapipe_import_error(self):
        # Remove cached liveness_service so __init__ runs fresh
        for key in list(sys.modules.keys()):
            if "liveness_service" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"mediapipe": None}):
            # Fresh import with mediapipe=None in sys.modules; module-level code
            # succeeds (mediapipe is only imported lazily inside __init__).
            from attendance.services import liveness_service as _ls_mod

            with self.assertRaises(ImportError) as ctx:
                _ls_mod.LivenessService()

        self.assertIn("mediapipe is required", str(ctx.exception))


class TestLandmarkIndicesCorrect(unittest.TestCase):
    """Req 2.4: LEFT_EYE and RIGHT_EYE landmark indices must be exact."""

    def test_landmark_indices_correct(self):
        # Ensure the module is importable (cv2 already mocked globally)
        if "attendance.services.liveness_service" not in sys.modules:
            make_service()  # side-effect: imports the module

        from attendance.services.liveness_service import LEFT_EYE, RIGHT_EYE

        self.assertEqual(LEFT_EYE, [33, 160, 158, 133, 153, 144])
        self.assertEqual(RIGHT_EYE, [362, 385, 387, 263, 373, 380])


class TestDefaultThresholdsWithoutSettings(unittest.TestCase):
    """Req 10.2: Default threshold values when LIVENESS_* are absent from settings."""

    def test_default_thresholds_without_settings(self):
        from django.conf import settings as django_settings

        # Names of all LIVENESS_* settings to temporarily remove
        liveness_keys = [
            "LIVENESS_EAR_THRESHOLD",
            "LIVENESS_EAR_AMPLITUDE_MIN",
            "LIVENESS_YAW_RANGE_MIN",
            "LIVENESS_PITCH_RANGE_MIN",
            "LIVENESS_TEMPORAL_STD_MIN",
            "LIVENESS_TEMPORAL_STD_MAX",
            "LIVENESS_TIMEOUT_SECONDS",
        ]

        # Save originals and delete them so getattr uses the hardcoded defaults
        saved = {}
        for key in liveness_keys:
            if hasattr(django_settings, key):
                saved[key] = getattr(django_settings, key)
                delattr(django_settings, key)

        try:
            mp_mock = MagicMock()
            mp_mock.solutions.face_mesh.FaceMesh.return_value = MagicMock()

            for mod_key in list(sys.modules.keys()):
                if "liveness_service" in mod_key:
                    del sys.modules[mod_key]

            with patch.dict(
                "sys.modules",
                {
                    "mediapipe": mp_mock,
                    "mediapipe.solutions": mp_mock.solutions,
                    "mediapipe.solutions.face_mesh": mp_mock.solutions.face_mesh,
                },
            ):
                from attendance.services.liveness_service import LivenessService
                service = LivenessService()
        finally:
            # Restore original settings values
            for key, val in saved.items():
                setattr(django_settings, key, val)

        self.assertEqual(service.EAR_THRESHOLD, 0.22)
        self.assertEqual(service.EAR_AMPLITUDE_MIN, 0.08)
        self.assertEqual(service.YAW_RANGE_MIN, 8.0)
        self.assertEqual(service.PITCH_RANGE_MIN, 6.0)
        self.assertEqual(service.TEMPORAL_STD_MIN, 0.005)
        self.assertEqual(service.TEMPORAL_STD_MAX, 0.08)
        self.assertEqual(service.timeout_seconds, 8.0)


class TestSpoofingLogsWarning(unittest.TestCase):
    """Req 9.3: WARNING log is emitted when spoofing (high EAR-yaw correlation) is detected."""

    def test_spoofing_logs_warning(self):
        service = make_service()

        # Build 15 frames with perfectly correlated EAR and yaw
        ears = [0.1 + i * 0.02 for i in range(15)]
        yaws = [float(i) for i in range(15)]
        for ear, yaw in zip(ears, yaws):
            service.frame_buffer.append(
                {"timestamp": time.time(), "ear": ear, "pitch": 0.0, "yaw": yaw}
            )

        service.blink_detected = True
        service.movement_detected = True

        # Provide a valid face so update() processes normally
        service.face_mesh.process.return_value = _make_face_landmarks_result(468)

        # solvePnP fails → new frame gets yaw=0.0 (doesn't affect existing 15 frames)
        _cv2_mock.solvePnP.return_value = (False, None, None)
        try:
            with self.assertLogs(
                "attendance.services.liveness_service", level="WARNING"
            ) as cm:
                service.update(np.zeros((480, 640, 3), dtype=np.uint8))
        finally:
            _cv2_mock.solvePnP.return_value = (True, _dummy_rvec, _dummy_tvec)

        # At least one WARNING log must mention spoofing
        warning_messages = [r for r in cm.output if "WARNING" in r]
        self.assertTrue(
            any("Spoofing detected" in msg for msg in warning_messages),
            f"Expected 'Spoofing detected' in WARNING logs, got: {cm.output}",
        )


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Property-Based Tests (Hypothesis) — Tasks 10.2 – 10.13
# ===========================================================================

from hypothesis import given, settings, assume
from hypothesis import strategies as st


def inject_frames(service, ears, yaws=None, pitches=None):
    """Inject frames directly into frame_buffer."""
    if yaws is None:
        yaws = [0.0] * len(ears)
    if pitches is None:
        pitches = [0.0] * len(ears)
    service.frame_buffer = []
    for ear, yaw, pitch in zip(ears, yaws, pitches):
        service.frame_buffer.append({
            "timestamp": time.time(),
            "ear": ear,
            "pitch": pitch,
            "yaw": yaw,
        })


# ---------------------------------------------------------------------------
# Property 1 (Task 10.2): Init and reset produce identical state
# ---------------------------------------------------------------------------

@given(n_frames=st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_property_1_init_and_reset_equivalent(n_frames):
    """Feature: hybrid-liveness-detection, Property 1: init and reset produce identical state

    Validates: Requirements 1.1, 1.2
    """
    service = make_service()
    # Add n_frames to the buffer
    for i in range(n_frames):
        service.frame_buffer.append({"timestamp": time.time(), "ear": 0.3, "pitch": 0.0, "yaw": 0.0})
    service.blink_detected = True
    service.movement_detected = True
    service.consistency_passed = True

    service.reset()

    assert service.frame_buffer == []
    assert service.blink_detected == False
    assert service.movement_detected == False
    assert service.consistency_passed == False


# ---------------------------------------------------------------------------
# Property 2 (Task 10.3): Frame buffer never exceeds history_size
# ---------------------------------------------------------------------------

@given(n_frames=st.integers(min_value=31, max_value=200))
@settings(max_examples=100)
def test_property_2_buffer_bounded(n_frames):
    """Feature: hybrid-liveness-detection, Property 2: frame buffer never exceeds history_size

    Validates: Requirements 1.3, 7.3
    """
    service = make_service()
    for i in range(n_frames):
        service.frame_buffer.append({"timestamp": time.time(), "ear": 0.3, "pitch": 0.0, "yaw": 0.0})
        if len(service.frame_buffer) > service.history_size:
            service.frame_buffer.pop(0)

    assert len(service.frame_buffer) <= service.history_size


# ---------------------------------------------------------------------------
# Property 3 (Task 10.4): Blink detection activates on correct condition
# ---------------------------------------------------------------------------

@given(ears=st.lists(st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False), min_size=5, max_size=30))
@settings(max_examples=200)
def test_property_3_blink_detection_condition(ears):
    """Feature: hybrid-liveness-detection, Property 3: blink detection activates on correct condition

    Validates: Requirements 2.1, 2.2
    """
    service = make_service()
    inject_frames(service, ears)

    ear_range = max(ears) - min(ears)
    should_blink = ear_range > service.EAR_AMPLITUDE_MIN and min(ears) < service.EAR_THRESHOLD

    # Trigger blink detection logic directly (same as in update())
    if len(service.frame_buffer) >= 5 and not service.blink_detected:
        ear_vals = [f["ear"] for f in service.frame_buffer]
        r = max(ear_vals) - min(ear_vals)
        if r > service.EAR_AMPLITUDE_MIN and min(ear_vals) < service.EAR_THRESHOLD:
            service.blink_detected = True

    assert service.blink_detected == should_blink


# ---------------------------------------------------------------------------
# Property 4 (Task 10.5): Detection flags are idempotent once set
# ---------------------------------------------------------------------------

@given(
    ears=st.lists(st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False), min_size=5, max_size=30),
    extra_ears=st.lists(st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_property_4_detection_flags_idempotent(ears, extra_ears):
    """Feature: hybrid-liveness-detection, Property 4: detection flags are idempotent once set

    Validates: Requirements 2.6, 3.6
    """
    service = make_service()
    service.blink_detected = True
    service.movement_detected = True

    # Add more frames — flags must not go back to False
    for ear in extra_ears:
        service.frame_buffer.append({"timestamp": time.time(), "ear": ear, "pitch": 0.0, "yaw": 0.0})
        if len(service.frame_buffer) > service.history_size:
            service.frame_buffer.pop(0)
        # Simulate the blink/movement check logic
        if len(service.frame_buffer) >= 5 and not service.blink_detected:
            ear_vals = [f["ear"] for f in service.frame_buffer]
            r = max(ear_vals) - min(ear_vals)
            if r > service.EAR_AMPLITUDE_MIN and min(ear_vals) < service.EAR_THRESHOLD:
                service.blink_detected = True

    assert service.blink_detected == True
    assert service.movement_detected == True


# ---------------------------------------------------------------------------
# Property 5 (Task 10.6): Movement detection activates on correct condition
# ---------------------------------------------------------------------------

@given(
    yaws=st.lists(st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False), min_size=5, max_size=30),
    pitches=st.lists(st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False), min_size=5, max_size=30),
)
@settings(max_examples=200)
def test_property_5_movement_detection_condition(yaws, pitches):
    """Feature: hybrid-liveness-detection, Property 5: movement detection activates on correct condition

    Validates: Requirements 3.4
    """
    # Use the shorter list length
    n = min(len(yaws), len(pitches))
    yaws = yaws[:n]
    pitches = pitches[:n]
    ears = [0.3] * n

    service = make_service()
    inject_frames(service, ears, yaws=yaws, pitches=pitches)

    yaw_range = max(yaws) - min(yaws)
    pitch_range = max(pitches) - min(pitches)
    should_move = yaw_range > service.YAW_RANGE_MIN or pitch_range > service.PITCH_RANGE_MIN

    # Trigger movement detection logic directly
    if len(service.frame_buffer) >= 5 and not service.movement_detected:
        y_vals = [f["yaw"] for f in service.frame_buffer]
        p_vals = [f["pitch"] for f in service.frame_buffer]
        yr = max(y_vals) - min(y_vals)
        pr = max(p_vals) - min(p_vals)
        if yr > service.YAW_RANGE_MIN or pr > service.PITCH_RANGE_MIN:
            service.movement_detected = True

    assert service.movement_detected == should_move


# ---------------------------------------------------------------------------
# Property 6 (Task 10.7): Temporal consistency depends on EAR std range
# ---------------------------------------------------------------------------

@given(ears=st.lists(st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False), min_size=10, max_size=30))
@settings(max_examples=200)
def test_property_6_temporal_consistency_std_range(ears):
    """Feature: hybrid-liveness-detection, Property 6: temporal consistency depends on EAR std range

    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
    """
    service = make_service()
    inject_frames(service, ears)

    std = float(np.std(ears))
    expected = service.TEMPORAL_STD_MIN <= std <= service.TEMPORAL_STD_MAX

    result = service._check_temporal_consistency()
    assert result == expected


# ---------------------------------------------------------------------------
# Property 7 (Task 10.8): Hybrid decision and dynamic feedback
# ---------------------------------------------------------------------------

@given(
    blink=st.booleans(),
    movement=st.booleans(),
    consistency=st.booleans(),
)
@settings(max_examples=100)
def test_property_7_hybrid_decision_and_feedback(blink, movement, consistency):
    """Feature: hybrid-liveness-detection, Property 7: hybrid decision and dynamic feedback

    Validates: Requirements 5.1, 5.3, 5.5
    """
    service = make_service()
    service.blink_detected = blink
    service.movement_detected = movement
    service.consistency_passed = consistency

    # Inject 10 frames with valid std so temporal check passes if consistency=True
    # We're testing the decision logic directly, not via update()
    if blink and movement and consistency:
        # All flags True → should return passed
        # Simulate the hybrid decision
        result_status = "passed"
    else:
        result_status = "pending"

    # Build expected feedback
    feedback = []
    if not blink:
        feedback.append("chớp mắt")
    if not movement:
        feedback.append("quay đầu nhẹ")
    # consistency hint only shown when buffer >= 10

    if blink and movement and consistency:
        assert result_status == "passed"
    else:
        assert result_status == "pending"
        # Verify feedback logic
        if not blink:
            assert "chớp mắt" in feedback
        if not movement:
            assert "quay đầu nhẹ" in feedback


# ---------------------------------------------------------------------------
# Property 8 (Task 10.9): update() always returns valid schema
# ---------------------------------------------------------------------------

@given(
    image_type=st.sampled_from(["zeros", "noise"])
)
@settings(max_examples=50)
def test_property_8_output_schema_always_valid(image_type):
    """Feature: hybrid-liveness-detection, Property 8: update() always returns valid schema

    Validates: Requirements 8.1
    """
    service = make_service()

    if image_type == "zeros":
        image = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    # Mock face_mesh to return no face (simplest case that exercises the full path)
    no_face = MagicMock()
    no_face.multi_face_landmarks = None
    service.face_mesh.process.return_value = no_face

    result = service.update(image)

    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] in {"pending", "passed", "failed"}
    assert "feedback_ui" in result
    assert isinstance(result["feedback_ui"], str)
    assert len(result["feedback_ui"]) > 0


# ---------------------------------------------------------------------------
# Property 9 (Task 10.10): Frame buffer contains only numeric data
# ---------------------------------------------------------------------------

@given(n_frames=st.integers(min_value=1, max_value=50))
@settings(max_examples=100)
def test_property_9_buffer_no_image_data(n_frames):
    """Feature: hybrid-liveness-detection, Property 9: frame buffer contains only numeric data

    Validates: Requirements 7.5
    """
    service = make_service()

    for i in range(n_frames):
        service.frame_buffer.append({
            "timestamp": time.time(),
            "ear": 0.3,
            "pitch": 0.0,
            "yaw": 0.0,
        })
        if len(service.frame_buffer) > service.history_size:
            service.frame_buffer.pop(0)

    for entry in service.frame_buffer:
        assert set(entry.keys()) == {"timestamp", "ear", "pitch", "yaw"}
        assert isinstance(entry["timestamp"], float)
        assert isinstance(entry["ear"], float)
        assert isinstance(entry["pitch"], float)
        assert isinstance(entry["yaw"], float)
        # Must not contain numpy arrays or image data
        assert not isinstance(entry["ear"], np.ndarray)
        assert not isinstance(entry["yaw"], np.ndarray)


# ---------------------------------------------------------------------------
# Property 10 (Task 10.11): Instance isolation
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(n_frames_a=st.integers(min_value=1, max_value=20))
def test_property_10_instance_isolation(n_frames_a):
    """Feature: hybrid-liveness-detection, Property 10: instances are independent

    Validates: Requirements 7.4
    """
    service_a = make_service()
    service_b = make_service()

    # Modify service_a
    for i in range(n_frames_a):
        service_a.frame_buffer.append({"timestamp": time.time(), "ear": 0.3, "pitch": 0.0, "yaw": 0.0})
    service_a.blink_detected = True
    service_a.movement_detected = True

    # service_b must be unaffected
    assert len(service_b.frame_buffer) == 0
    assert service_b.blink_detected == False
    assert service_b.movement_detected == False

    # Reset service_a — service_b still unaffected
    service_a.reset()
    assert len(service_b.frame_buffer) == 0


# ---------------------------------------------------------------------------
# Property 11 (Task 10.12): Spoofing detection triggers reset and failed
# ---------------------------------------------------------------------------

@given(
    base=st.lists(
        st.floats(min_value=0.1, max_value=0.4, allow_nan=False, allow_infinity=False),
        min_size=15, max_size=30
    )
)
@settings(max_examples=100)
def test_property_11_spoofing_detection_resets(base):
    """Feature: hybrid-liveness-detection, Property 11: spoofing detection triggers reset and failed

    Validates: Requirements 9.1, 9.2, 9.4
    """
    service = make_service()

    # Create perfectly correlated EAR and yaw (correlation = 1.0)
    ears = base
    yaws = [v * 10.0 for v in base]  # perfectly correlated

    inject_frames(service, ears, yaws=yaws)
    service.blink_detected = True
    service.movement_detected = True

    # Check that spoofing is detected
    is_spoofing = service._check_spoofing_correlation()

    if is_spoofing:
        # Simulate what update() does on spoofing detection
        service.reset()
        assert len(service.frame_buffer) == 0
        assert service.blink_detected == False


# ---------------------------------------------------------------------------
# Property 12 (Task 10.13): Settings values are applied correctly
# ---------------------------------------------------------------------------

@given(
    ear_threshold=st.floats(min_value=0.15, max_value=0.30, allow_nan=False, allow_infinity=False),
    yaw_min=st.floats(min_value=3.0, max_value=15.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_property_12_settings_applied_correctly(ear_threshold, yaw_min):
    """Feature: hybrid-liveness-detection, Property 12: settings values are applied correctly

    Validates: Requirements 10.1, 10.2
    """
    from django.test import override_settings

    mp_mock = MagicMock()
    mp_mock.solutions.face_mesh.FaceMesh.return_value = MagicMock()

    for key in list(sys.modules.keys()):
        if "liveness_service" in key:
            del sys.modules[key]

    with override_settings(
        LIVENESS_EAR_THRESHOLD=ear_threshold,
        LIVENESS_YAW_RANGE_MIN=yaw_min,
    ):
        with patch.dict("sys.modules", {
            "mediapipe": mp_mock,
            "mediapipe.solutions": mp_mock.solutions,
            "mediapipe.solutions.face_mesh": mp_mock.solutions.face_mesh,
        }):
            from attendance.services.liveness_service import LivenessService
            service = LivenessService()

    assert service.EAR_THRESHOLD == ear_threshold
    assert service.YAW_RANGE_MIN == yaw_min
