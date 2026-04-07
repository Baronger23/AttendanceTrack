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


@shared_task(bind=True, max_retries=0, time_limit=30, soft_time_limit=25)
def identify_face_task(self, image_base64: str, channel_group_name: str = None) -> dict:
    """
    Async face identification task.
    
    Args:
        image_base64: Base64 encoded JPEG image
    
    Returns:
        dict with: user_id, user_name, user_full_name, confidence, 
                   avatar_url, shift_name, status, message
    """
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                'success': False,
                'error': 'Không thể đọc ảnh!',
            }
        
        # Run CNN pipeline
        from attendance.services.face_service import FaceService
        face_service = FaceService()
        result = face_service.identify_face(img)
        
        if not result['success']:
            return {
                'success': False,
                'error': result['error'],
            }
        
        # Get user info
        from attendance.models import User, AttendanceLog
        from django.utils import timezone
        
        try:
            user = User.objects.get(id=result['user_id'])
        except User.DoesNotExist:
            return {
                'success': False,
                'error': 'Không tìm thấy nhân viên trong hệ thống!',
            }
        
        confidence = result['confidence']
        today = timezone.now().date()
        now_time = timezone.localtime(timezone.now()).time()
        
        # Check shift
        shift_error = None
        if user.work_shift:
            shift = user.work_shift
            from datetime import timedelta, datetime as dt
            
            EARLY_MINUTES = 60
            LATE_AFTER_END_MINUTES = 30
            
            today_dt = timezone.localtime(timezone.now()).date()
            shift_start_dt = dt.combine(today_dt, shift.start_time)
            shift_end_dt = dt.combine(today_dt, shift.end_time)
            
            allowed_start = (shift_start_dt - timedelta(minutes=EARLY_MINUTES)).time()
            allowed_end = (shift_end_dt + timedelta(minutes=LATE_AFTER_END_MINUTES)).time()
            
            if shift.start_time > shift.end_time:
                in_shift = now_time >= allowed_start or now_time <= allowed_end
            else:
                in_shift = allowed_start <= now_time <= allowed_end
            
            if not in_shift:
                shift_error = (
                    f'Chưa đến ca của bạn! Ca "{shift.name}" bắt đầu lúc '
                    f'{shift.start_time.strftime("%H:%M")} - {shift.end_time.strftime("%H:%M")}. '
                    f'Bạn có thể chấm công từ {allowed_start.strftime("%H:%M")}.'
                )
        
        # Check existing log
        existing_log = AttendanceLog.objects.filter(
            user=user,
            timestamp__date=today
        ).first()
        
        status_msg = ''
        checkin_status = ''
        
        if shift_error:
            status_msg = shift_error
            checkin_status = 'shift_error'
        elif existing_log:
            status_msg = f'{user.get_full_name()} đã chấm công hôm nay lúc {existing_log.timestamp.strftime("%H:%M")}!'
            checkin_status = 'already_checked'
        else:
            log = AttendanceLog.objects.create(user=user)
            status_msg = f'Chấm công thành công! Xin chào {user.get_full_name()} - {log.get_status_display()}'
            checkin_status = 'success'
        
        task_result = {
            'success': True,
            'user_id': user.id,
            'user_name': user.username,
            'user_full_name': str(user),
            'first_initial': (user.first_name[:1].upper() if user.first_name else '?'),
            'confidence': confidence,
            'avatar_url': user.get_avatar_url() or '',
            'shift_name': user.work_shift.name if user.work_shift else '',
            'checkin_status': checkin_status,
            'message': status_msg,
        }
        _push_to_websocket(channel_group_name, task_result)
        return task_result
        
    except Exception as e:
        logger.error(f"identify_face_task failed: {e}", exc_info=True)
        task_result = {
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}',
        }
        _push_to_websocket(channel_group_name, task_result)
        return task_result


def _push_to_websocket(group_name, result):
    """Push result to WebSocket channel if group_name is provided."""
    if not group_name:
        return
    try:
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
