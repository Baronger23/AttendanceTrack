from .models import Notification

def notifications_processor(request):
    """Context processor để inject notifications vào tất cả admin templates"""
    if request.user.is_authenticated and hasattr(request.user, 'role') and request.user.role == 'ADMIN':
        # Lấy 10 thông báo mới nhất
        notifications = Notification.objects.all()[:10]
        unread_count = Notification.objects.filter(is_read=False).count()
        
        return {
            'notifications': notifications,
            'unread_notifications_count': unread_count,
        }
    return {}
