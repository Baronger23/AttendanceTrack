from django.apps import AppConfig
import os


class AttendanceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'attendance'

    def ready(self):
        import attendance.services.cache_service
        from attendance.services.cache_service import FaceCacheService
        import sys
        
        # Prevent running cache warm-up during migrations or test runs to avoid errors
        if ('runserver' in sys.argv or 'daphne' in sys.argv) and os.getenv('REDIS_URL'):
            try:
                FaceCacheService.warmup_cache()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to warm-up Redis cache: {e}")
