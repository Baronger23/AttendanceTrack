"""
WebSocket consumer for real-time kiosk face recognition results.

Flow:
  1. Browser connects to ws://host/ws/kiosk/
  2. Browser sends {'type': 'checkin', 'image': base64_data, 'session_id': '...'}
  3. Consumer dispatches Celery task with session_id
  4. Celery task completes → channel_layer.group_send() → Consumer receives
  5. Consumer pushes result JSON to browser instantly
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)


def _json_safe(value):
    """Convert AI/debug objects to JSON-safe values for WebSocket responses."""
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


class KioskConsumer(AsyncWebsocketConsumer):
    
    async def connect(self):
        """Accept WebSocket connection and join a unique group."""
        # Generate a unique session group for this connection
        import re
        safe_channel_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', self.channel_name)
        self.session_id = self.scope.get('url_route', {}).get('kwargs', {}).get(
            'session_id', safe_channel_name
        )
        self.group_name = f'kiosk_{safe_channel_name}'
        
        # Join the group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Initialize stateful LivenessService for this connection
        from attendance.services.liveness_service import LivenessService
        self.liveness_service = LivenessService()
        self.liveness_passed = False
        self.last_liveness_result = {}
        
        # Send connection confirmation
        await self.send(text_data=json.dumps({
            'type': 'connected',
            'session_id': self.group_name,
        }))
    
    async def disconnect(self, close_code):
        """Leave the group on disconnect."""
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Handle incoming messages from the browser."""
        try:
            data = json.loads(text_data)
            msg_type = data.get('type')
            
            if msg_type == 'checkin':
                await self.handle_checkin(data)
            elif msg_type == 'ping':
                await self.send(text_data=json.dumps({'type': 'pong'}))
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'error': 'Invalid JSON',
            }))
    
    async def handle_checkin(self, data):
        """Dispatch face recognition to Celery worker."""
        image_data = data.get('image', '')
        recognition_image_data = data.get('recognition_image') or image_data
        import base64
        import cv2
        import numpy as np
        from asgiref.sync import sync_to_async
        
        # Strip data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        if ',' in recognition_image_data:
            recognition_image_data = recognition_image_data.split(',', 1)[1]
            
        # 1. Decode image for Liveness Check
        try:
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid image")
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'error': 'Lỗi giải mã ảnh',
            }))
            return

        # 2. Perform Liveness Check if not already passed
        if not self.liveness_passed:
            liveness_result = await sync_to_async(self.liveness_service.update)(img)
            
            if liveness_result["status"] == "pending":
                # Need more frames, send feedback to UI
                await self.send(text_data=json.dumps({
                    'type': 'liveness_feedback',
                    'message': liveness_result["feedback_ui"],
                }))
                return
            elif liveness_result["status"] == "failed":
                # Timeout or spoofing detected — reset and allow retry
                self.liveness_passed = False
                self.liveness_service.reset()
                await self.send(text_data=json.dumps({
                    'type': 'result',
                    'result': {'success': False, 'feedback_ui': liveness_result["feedback_ui"]},
                }))
                return
            else:
                # Passed!
                self.liveness_passed = True
                self.last_liveness_result = liveness_result
                await self.send(text_data=json.dumps({
                    'type': 'liveness_feedback',
                    'message': liveness_result["feedback_ui"],  # "Xác thực thành công!"
                }))
        
        # 3. Liveness passed -> Dispatch Face Recognition
        # Send "processing" acknowledgement
        await self.send(text_data=json.dumps({
            'type': 'processing',
            'step': 1,
            'message': 'Đang nhận diện danh tính...',
        }))
        
        from django.conf import settings
        if not getattr(settings, "KIOSK_USE_CELERY", False):
            await self.send(text_data=json.dumps({
                'type': 'processing',
                'step': 2,
                'message': 'Äang phÃ¢n tÃ­ch Ä‘áº·c trÆ°ng khuÃ´n máº·t...',
            }))
            try:
                await self.run_sync_fallback(recognition_image_data, None)
            finally:
                self.liveness_passed = False
                self.last_liveness_result = {}
                self.liveness_service.reset()
            return

        try:
            from attendance.tasks import identify_face_task
            # Remove previous_image_data since we don't use MSE anymore
            task = identify_face_task.delay(recognition_image_data, None, self.group_name, self.last_liveness_result)
            
            await self.send(text_data=json.dumps({
                'type': 'processing',
                'step': 2,
                'task_id': task.id,
                'message': 'Đang phân tích đặc trưng khuôn mặt...',
            }))
            
            # Reset liveness for the next person after sending to Celery
            self.liveness_passed = False
            self.last_liveness_result = {}
            self.liveness_service.reset()
            
        except Exception as e:
            logger.warning(f"Celery unavailable: {e}")
            await self.run_sync_fallback(recognition_image_data, None)
    
    async def run_sync_fallback(self, image_data, previous_image_data=None):
        """Synchronous fallback when Celery is unavailable."""
        import base64
        import cv2
        import numpy as np
        from asgiref.sync import sync_to_async
        
        try:
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                await self.send(text_data=json.dumps({
                    'type': 'result',
                    'result': {'success': False, 'error': 'Không thể đọc ảnh!'},
                }))
                return
            
            from attendance.services.face_service import FaceService
            face_service = FaceService()
            
            # [ANTI-SPOOFING] Old MSE logic removed because LivenessService handles it now.
            result = await sync_to_async(face_service.identify_face)(img)
            
            if not result['success']:
                await self.send(text_data=json.dumps({
                    'type': 'result',
                    'result': _json_safe(result),
                }))
                return
            
            from attendance.models import User
            from attendance.services.attendance_policy import AttendancePolicyService
            
            user = await sync_to_async(User.objects.get)(id=result['user_id'])
            decision = await sync_to_async(AttendancePolicyService.record_checkin)(
                user=user,
                recognition_result=result,
                liveness_result=getattr(self, "last_liveness_result", {}),
            )
            
            avatar_url = await sync_to_async(user.get_avatar_url)() or ''
            shift_name = ''
            if user.work_shift_id:
                shift = await sync_to_async(lambda: user.work_shift)()
                shift_name = shift.name if shift else ''
            
            await self.send(text_data=json.dumps({
                'type': 'result',
                'result': {
                    'success': True,
                    'user_id': user.id,
                    'user_full_name': str(user),
                    'first_initial': (user.first_name[:1].upper() if user.first_name else '?'),
                    'confidence': result['confidence'],
                    'avatar_url': avatar_url,
                    'shift_name': shift_name,
                    'checkin_status': decision.checkin_status,
                    'message': decision.message,
                    'risk_score': decision.risk_score,
                    'attendance_status': decision.status,
                }
            }))
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'result',
                'result': {'success': False, 'error': str(e)},
            }))
    
    # --- Handler for messages from Celery (via channel_layer) ---
    async def kiosk_result(self, event):
        """Receive result from Celery task and push to browser."""
        await self.send(text_data=json.dumps({
            'type': 'result',
            'result': _json_safe(event['result']),
        }))
