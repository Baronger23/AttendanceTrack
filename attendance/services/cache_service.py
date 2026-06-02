"""
Cache Service for Face Embeddings.
Handles caching of user embeddings to Redis to improve Vector Search fallback performance
and reduce database load, while strictly maintaining cache integrity (preventing stale data).
"""

import logging
from django.core.cache import cache
from attendance.models import FaceEmbedding, User

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL = 3600  # 1 hour
EMBEDDINGS_KEY_PREFIX = "face_embeddings_v1:"
USER_CACHE_VERSION_KEY = "face_embeddings_version"

class FaceCacheService:
    METHOD_FILTERS = {
        'arcface': {'insightface_arcface'},
        'facenet': {'facenet_inceptionresnet', 'legacy_unknown', ''},
    }

    @staticmethod
    def allowed_methods(method_filter: str | None) -> set:
        if not method_filter:
            return set()
        return FaceCacheService.METHOD_FILTERS.get(method_filter, set())

    @staticmethod
    def get_cache_version() -> int:
        """Get the current global version for embeddings cache."""
        version = cache.get(USER_CACHE_VERSION_KEY)
        if version is None:
            version = 1
            cache.set(USER_CACHE_VERSION_KEY, version, None) # Never expires
        return version

    @staticmethod
    def invalidate_all():
        """Invalidate all embeddings cache by bumping the global version."""
        current_version = FaceCacheService.get_cache_version()
        cache.set(USER_CACHE_VERSION_KEY, current_version + 1, None)
        logger.info(f"Invalidated global embedding cache. New version: {current_version + 1}")

    @staticmethod
    def invalidate_user(user_id: int):
        """Invalidate specific user cache."""
        version = FaceCacheService.get_cache_version()
        key = f"{EMBEDDINGS_KEY_PREFIX}{version}:{user_id}"
        cache.delete(key)
        logger.info(f"Invalidated embedding cache for user {user_id}")

    @staticmethod
    def get_all_embeddings(method_filter: str | None = None) -> list:
        """
        Retrieve all embeddings. Uses cache if available.
        Returns a list of tuples: (embedding_numpy_array, user_id)
        """
        version = FaceCacheService.get_cache_version()
        filter_key = method_filter or 'all'
        key = f"{EMBEDDINGS_KEY_PREFIX}{version}:all:{filter_key}"
        
        cached_data = cache.get(key)
        if cached_data is not None:
            return cached_data
            
        # Cache miss, fetch from DB
        logger.debug("Embedding cache miss. Fetching from DB.")
        
        # Prioritize high quality embeddings
        embeddings_qs = FaceEmbedding.objects.all()
        allowed_methods = FaceCacheService.allowed_methods(method_filter)
        if allowed_methods:
            embeddings_qs = embeddings_qs.filter(recognition_method__in=allowed_methods)
        embeddings_qs = embeddings_qs.order_by('-quality_score')
        
        data = []
        for emb in embeddings_qs:
            numpy_arr = emb.get_embedding()
            if numpy_arr is not None:
                data.append((numpy_arr, emb.user_id, emb.recognition_method or 'legacy_unknown'))
                
        # Cache for 1 hour
        cache.set(key, data, CACHE_TTL)
        return data

    @staticmethod
    def warmup_cache():
        """Preload embeddings into Redis on app start to prevent slow first-requests."""
        logger.info("Starting Redis Cache Warm-up for embeddings...")
        data = FaceCacheService.get_all_embeddings()
        logger.info(f"Cache Warm-up completed. Loaded {len(data)} embeddings into Redis.")

# Signal receivers to automatically invalidate cache when embeddings change
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

@receiver(post_save, sender=FaceEmbedding)
@receiver(post_delete, sender=FaceEmbedding)
def invalidate_embedding_cache(sender, instance, **kwargs):
    """Automatically invalidates cache when FaceEmbedding is created/updated/deleted."""
    FaceCacheService.invalidate_all()
    # Or specifically: FaceCacheService.invalidate_user(instance.user_id)
    # But since Python fallback iterates all, bumping global version is safer.
