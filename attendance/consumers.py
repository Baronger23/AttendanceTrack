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


class KioskConsumer(AsyncWebsocketConsumer):
    
    async def connect(self):
        """Accept WebSocket connection and join a unique group."""
        # Generate a unique session group for this connection
        self.session_id = self.scope.get('url_route', {}).get('kwargs', {}).get(
            'session_id', self.channel_name
        )
        self.group_name = f'kiosk_{self.channel_name}'
        
        # Join the group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        await self.accept()
        
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
        
        if not image_data:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'error': 'Không có dữ liệu ảnh!',
            }))
            return
        
        # Strip data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        # Send "processing" acknowledgement immediately
        await self.send(text_data=json.dumps({
            'type': 'processing',
            'step': 1,
            'message': 'Đang gửi ảnh lên server...',
        }))
        
        # Dispatch Celery task with this group_name so it can push results back
        try:
            from attendance.tasks import identify_face_task
            task = identify_face_task.delay(image_data, self.group_name)
            
            await self.send(text_data=json.dumps({
                'type': 'processing',
                'step': 2,
                'task_id': task.id,
                'message': 'Đang phân tích đặc trưng khuôn mặt...',
            }))
        except Exception as e:
            # Celery unavailable — run sync fallback
            logger.warning(f"Celery unavailable: {e}")
            await self.run_sync_fallback(image_data)
    
    async def run_sync_fallback(self, image_data):
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
            result = await sync_to_async(face_service.identify_face)(img)
            
            if not result['success']:
                await self.send(text_data=json.dumps({
                    'type': 'result',
                    'result': result,
                }))
                return
            
            from attendance.models import User, AttendanceLog
            from django.utils import timezone
            
            user = await sync_to_async(User.objects.get)(id=result['user_id'])
            today = timezone.now().date()
            
            existing_log = await sync_to_async(
                AttendanceLog.objects.filter(user=user, timestamp__date=today).first
            )()
            
            if existing_log:
                checkin_status = 'already_checked'
                ts = await sync_to_async(lambda: existing_log.timestamp.strftime("%H:%M"))()
                full_name = await sync_to_async(user.get_full_name)()
                msg = f'{full_name} đã chấm công hôm nay lúc {ts}!'
            else:
                log = await sync_to_async(AttendanceLog.objects.create)(user=user)
                full_name = await sync_to_async(user.get_full_name)()
                status_display = await sync_to_async(log.get_status_display)()
                checkin_status = 'success'
                msg = f'Chấm công thành công! Xin chào {full_name} - {status_display}'
            
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
                    'checkin_status': checkin_status,
                    'message': msg,
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
            'result': event['result'],
        }))
