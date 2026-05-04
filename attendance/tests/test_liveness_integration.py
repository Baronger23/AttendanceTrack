"""
Integration tests for KioskConsumer + LivenessService.
Tasks 12.1–12.4: hybrid-liveness-detection spec

Requirements: 5.2, 6.1, 6.4, 6.5, 8.3
"""

import base64
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")

# ---------------------------------------------------------------------------
# Mock cv2 and mediapipe before any imports that might pull them in
# ---------------------------------------------------------------------------
_cv2_mock = MagicMock()
_cv2_mock.imdecode.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
_cv2_mock.IMREAD_COLOR = 1
_cv2_mock.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
_dummy_rvec = np.zeros((3, 1), dtype=np.float64)
_dummy_tvec = np.zeros((3, 1), dtype=np.float64)
_cv2_mock.solvePnP.return_value = (True, _dummy_rvec, _dummy_tvec)
_cv2_mock.Rodrigues.return_value = (np.eye(3, dtype=np.float64), None)
_cv2_mock.RQDecomp3x3.return_value = (np.array([0.0, 0.0, 0.0]), None, None, None, None, None)
sys.modules["cv2"] = _cv2_mock

_mp_mock = MagicMock()
_mp_mock.solutions.face_mesh.FaceMesh.return_value = MagicMock()
sys.modules["mediapipe"] = _mp_mock
sys.modules["mediapipe.solutions"] = _mp_mock.solutions
sys.modules["mediapipe.solutions.face_mesh"] = _mp_mock.solutions.face_mesh

import django
django.setup()


def make_dummy_image_b64() -> str:
    """Create a minimal valid base64 image string for testing."""
    import io
    try:
        from PIL import Image
        img = Image.new("RGB", (4, 4), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Fallback: raw bytes that cv2 mock will accept
        return base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 100).decode()


IMAGE_B64 = make_dummy_image_b64()


def _make_consumer():
    """
    Create a KioskConsumer instance with state pre-initialised,
    bypassing the real connect() which requires a live channel layer.
    """
    # Remove cached consumers module to get a fresh import
    for key in list(sys.modules.keys()):
        if "consumers" in key and "attendance" in key:
            del sys.modules[key]
    for key in list(sys.modules.keys()):
        if "liveness_service" in key:
            del sys.modules[key]

    from attendance.consumers import KioskConsumer
    from attendance.services.liveness_service import LivenessService

    consumer = KioskConsumer()
    consumer.channel_name = "test_channel"
    consumer.group_name = "kiosk_test"
    consumer.liveness_passed = False
    consumer.liveness_service = LivenessService()
    return consumer


# ---------------------------------------------------------------------------
# Task 12.1 — Consumer dispatches Celery after liveness pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumer_dispatches_celery_after_liveness_pass():
    """
    Req 5.2, 8.3: Consumer dispatches identify_face_task.delay() exactly once
    when LivenessService returns status='passed'.
    """
    consumer = _make_consumer()

    sent_messages = []

    async def mock_send(text_data):
        sent_messages.append(json.loads(text_data))

    consumer.send = mock_send

    mock_task = MagicMock()
    mock_task.id = "test-task-id-123"

    with patch(
        "attendance.services.liveness_service.LivenessService.update",
        return_value={"status": "passed", "feedback_ui": "Xác thực thành công!"},
    ), patch(
        "attendance.tasks.identify_face_task"
    ) as mock_celery_task:
        mock_celery_task.delay.return_value = mock_task

        await consumer.handle_checkin({"image": IMAGE_B64})

    # Celery task must have been dispatched exactly once
    mock_celery_task.delay.assert_called_once()

    # Should have received liveness_feedback "Xác thực thành công!"
    types = [m["type"] for m in sent_messages]
    assert "liveness_feedback" in types
    assert any(
        m["type"] == "liveness_feedback" and m["message"] == "Xác thực thành công!"
        for m in sent_messages
    )

    # Should have received processing messages
    assert any(m["type"] == "processing" for m in sent_messages)

    # The processing step 2 message should carry the task_id
    step2_msgs = [m for m in sent_messages if m.get("type") == "processing" and m.get("step") == 2]
    assert len(step2_msgs) == 1
    assert step2_msgs[0]["task_id"] == "test-task-id-123"


# ---------------------------------------------------------------------------
# Task 12.2 — Consumer sends liveness_feedback when pending
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumer_sends_liveness_feedback_on_pending():
    """
    Req 6.1: Consumer sends {"type": "liveness_feedback", "message": ...}
    when LivenessService returns status='pending'.
    """
    consumer = _make_consumer()

    sent_messages = []

    async def mock_send(text_data):
        sent_messages.append(json.loads(text_data))

    consumer.send = mock_send

    with patch(
        "attendance.services.liveness_service.LivenessService.update",
        return_value={"status": "pending", "feedback_ui": "Vui lòng chớp mắt"},
    ):
        await consumer.handle_checkin({"image": IMAGE_B64})

    # Must receive exactly one liveness_feedback message with the pending text
    feedback_msgs = [m for m in sent_messages if m.get("type") == "liveness_feedback"]
    assert len(feedback_msgs) == 1
    assert feedback_msgs[0]["message"] == "Vui lòng chớp mắt"

    # Must NOT have dispatched any Celery task (no processing messages)
    assert not any(m["type"] == "processing" for m in sent_messages)


# ---------------------------------------------------------------------------
# Task 12.3 — Consumer resets after successful dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumer_resets_after_successful_dispatch():
    """
    Req 6.5: After liveness passes and Celery task is dispatched,
    liveness_service.reset() is called and liveness_passed = False.
    """
    consumer = _make_consumer()

    sent_messages = []

    async def mock_send(text_data):
        sent_messages.append(json.loads(text_data))

    consumer.send = mock_send

    mock_task = MagicMock()
    mock_task.id = "reset-test-task-id"

    with patch(
        "attendance.services.liveness_service.LivenessService.update",
        return_value={"status": "passed", "feedback_ui": "Xác thực thành công!"},
    ), patch.object(
        consumer.liveness_service, "reset"
    ) as mock_reset, patch(
        "attendance.tasks.identify_face_task"
    ) as mock_celery_task:
        mock_celery_task.delay.return_value = mock_task

        await consumer.handle_checkin({"image": IMAGE_B64})

    # reset() must have been called at least once
    mock_reset.assert_called()

    # liveness_passed must be False after the dispatch
    assert consumer.liveness_passed is False


# ---------------------------------------------------------------------------
# Task 12.4 — Consumer sends result when liveness failed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumer_sends_result_on_liveness_failed():
    """
    Req 6.4: Consumer sends {"type": "result", "result": {"success": False, "feedback_ui": ...}}
    when LivenessService returns status='failed'.
    """
    consumer = _make_consumer()

    sent_messages = []

    async def mock_send(text_data):
        sent_messages.append(json.loads(text_data))

    consumer.send = mock_send

    with patch(
        "attendance.services.liveness_service.LivenessService.update",
        return_value={"status": "failed", "feedback_ui": "Hết thời gian..."},
    ):
        await consumer.handle_checkin({"image": IMAGE_B64})

    # Must receive a result message with success=False
    result_msgs = [m for m in sent_messages if m.get("type") == "result"]
    assert len(result_msgs) == 1
    assert result_msgs[0]["result"]["success"] is False
    assert result_msgs[0]["result"]["feedback_ui"] == "Hết thời gian..."

    # Must NOT have dispatched any Celery task
    assert not any(m["type"] == "processing" for m in sent_messages)

    # liveness_passed must remain False
    assert consumer.liveness_passed is False
