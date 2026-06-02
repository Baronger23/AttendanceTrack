"""
Celery async tasks for face recognition.

Moves heavy AI computation (MTCNN → Align → Embed → SVM) to a background
worker so the web server responds instantly with "Đang xử lý...".
"""

import logging
import base64
import numpy as np
import cv2
from celery import shared_task

logger = logging.getLogger(__name__)


def _json_safe(value):
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _json_safe(val)
            for key, val in value.items()
            if key not in {"embedding", "all_embeddings"}
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@shared_task(bind=True, max_retries=0, time_limit=30, soft_time_limit=25)
def identify_face_task(self, image_base64: str, previous_image_base64: str = None, channel_group_name: str = None, liveness_result: dict = None) -> dict:
    """
    Async face identification task.
    
    Args:
        image_base64: Base64 encoded JPEG image
        channel_group_name: Name of the websocket channel
    """
    try:
        # Decode base64 images
        img_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                'success': False,
                'error': 'Không thể đọc ảnh!',
            }
            
        from attendance.services.face_service import FaceService
        face_service = FaceService()
        
        # NOTE: Liveness is now handled directly by the WebSocket consumer
        # before this Celery task is even dispatched. This saves Celery queue
        # from being spammed with liveness frames.
        
        # Run CNN pipeline
        result = face_service.identify_face(img)
        
        if not result['success']:
            task_result = {
                'success': False,
                'error': result.get('error', 'Lỗi không xác định'),
                'feedback_ui': result.get('feedback_ui', 'Lỗi hệ thống, vui lòng thử lại.'),
            }
            task_result = _json_safe(task_result)
            _push_to_websocket(channel_group_name, task_result)
            return task_result
        
        # Get user info
        from attendance.models import User
        from attendance.services.attendance_policy import AttendancePolicyService
        
        try:
            user = User.objects.get(id=result['user_id'])
        except User.DoesNotExist:
            return {
                'success': False,
                'error': 'Không tìm thấy nhân viên trong hệ thống!',
            }
        
        decision = AttendancePolicyService.record_checkin(
            user=user,
            recognition_result=result,
            liveness_result=liveness_result or {},
        )
        
        task_result = {
            'success': True,
            'user_id': user.id,
            'user_name': user.username,
            'user_full_name': str(user),
            'first_initial': (user.first_name[:1].upper() if user.first_name else '?'),
            'confidence': result['confidence'],
            'avatar_url': user.get_avatar_url() or '',
            'shift_name': user.work_shift.name if user.work_shift else '',
            'checkin_status': decision.checkin_status,
            'message': decision.message,
            'risk_score': decision.risk_score,
            'attendance_status': decision.status,
        }
        task_result = _json_safe(task_result)
        _push_to_websocket(channel_group_name, task_result)
        return task_result
        
    except Exception as e:
        logger.error(f"identify_face_task failed: {e}", exc_info=True)
        task_result = {
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}',
        }
        task_result = _json_safe(task_result)
        _push_to_websocket(channel_group_name, task_result)
        return task_result


def _push_to_websocket(group_name, result):
    """Push result to WebSocket channel if group_name is provided."""
    if not group_name:
        return
    try:
        result = _json_safe(result)
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        channel_layer = get_channel_layer()
        if channel_layer:
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': 'kiosk.result',
                    'result': result,
                }
            )
    except Exception as e:
        logger.warning(f"Failed to push to WebSocket: {e}")
