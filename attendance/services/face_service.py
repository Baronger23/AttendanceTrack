"""
Face Service — Main Facade.

Orchestrates the full face recognition pipeline:
  Detection (MTCNN) → Alignment → Preprocessing (CLAHE) → Embedding (512D) → Classification (SVM)

Provides two main operations:
  1. register_face() — For staff registration/enrollment
  2. identify_face() — For kiosk check-in
"""

import numpy as np
import cv2
import logging
import time
import json

from .face_detection import FaceDetector
from .face_alignment import align_face
from .preprocessing import preprocess_for_detection, preprocess_for_embedding
from .embedding import FaceEmbedder
from .augmentation import augment_face
from .classifier import FaceClassifier

logger = logging.getLogger(__name__)


class FaceService:
    """Main AI service for face recognition operations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.classifier = FaceClassifier()
        
        self._initialized = True
        logger.info("FaceService initialized")
    
    # ==================== REGISTRATION ====================
    
    def register_face(self, image: np.ndarray, num_augmented: int = 20) -> dict:
        """
        Full registration pipeline: detect → align → preprocess → augment → embed.
        
        Args:
            image: BGR numpy array (from uploaded image)
            num_augmented: Number of augmented variants to generate
        
        Returns:
            dict with:
                - 'success': bool
                - 'embedding': 512D numpy array (average of all variants)
                - 'all_embeddings': list of 512D arrays (for pgvector storage)
                - 'confidence': face detection confidence
                - 'error': error message (if failed)
        """
        try:
            # 1. Preprocess for detection
            processed = preprocess_for_detection(image)
            
            # 2. Detect face (largest face for single-person registration)
            face = self.detector.detect_largest(processed)
            
            if face is None:
                return {
                    'success': False,
                    'error': 'Không phát hiện khuôn mặt trong ảnh!',
                }
            
            # 3. Align face using landmarks (160x160 for FaceNet)
            aligned = align_face(processed, face['landmarks'], output_size=160)
            
            # 4. Preprocess for embedding
            aligned = preprocess_for_embedding(aligned)
            
            # 5. Generate augmented variants
            variants = augment_face(aligned, num_augmented=num_augmented)
            
            # 6. Extract embeddings for all variants (batch processing)
            all_embeddings = self.embedder.extract_batch(variants)
            
            if not all_embeddings:
                return {
                    'success': False,
                    'error': 'Không thể trích xuất đặc trưng khuôn mặt!',
                }
            
            # 7. Quality scoring & Selection (Instead of averaging)
            # Select top K diverse embeddings. For this implementation, we take the original 
            # and a few variants to preserve variance (e.g., 5 total).
            # In a real system, we would calculate blurriness/lighting quality score here.
            selected_embeddings = all_embeddings[:5]
            quality_scores = [face['confidence']] * len(selected_embeddings) # Use detection conf as baseline quality
            
            # Calculate average for backward compatibility with older parts of the system
            avg_embedding = np.mean(all_embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            return {
                'success': True,
                'embedding': avg_embedding,
                'all_embeddings': selected_embeddings,
                'quality_scores': quality_scores,
                'confidence': face['confidence'],
            }
            
        except Exception as e:
            logger.error(f"Registration failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Lỗi xử lý ảnh: {str(e)}',
            }
    
    def register_multiple_images(self, images: list, num_augmented: int = 10) -> dict:
        """
        Register from multiple uploaded images.
        
        Args:
            images: List of BGR numpy arrays
            num_augmented: Augmented variants per image (lower since we have more originals)
        
        Returns:
            Same as register_face() but with combined embeddings from all images
        """
        all_embeddings = []
        all_quality_scores = []
        detection_confidence = 0.0
        
        for img in images:
            result = self.register_face(img, num_augmented=num_augmented)
            if result['success']:
                all_embeddings.extend(result['all_embeddings'])
                all_quality_scores.extend(result['quality_scores'])
                detection_confidence = max(detection_confidence, result['confidence'])
        
        if not all_embeddings:
            return {
                'success': False,
                'error': 'Không phát hiện khuôn mặt trong bất kỳ ảnh nào!',
            }
        
        avg_embedding = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return {
            'success': True,
            'embedding': avg_embedding,
            'all_embeddings': all_embeddings,
            'quality_scores': all_quality_scores,
            'confidence': detection_confidence,
        }
    
    # ==================== IDENTIFICATION ====================
    
    def identify_face(self, image: np.ndarray) -> dict:
        """
        Full identification pipeline: detect → align → preprocess → embed → classify.
        
        Uses SVM classifier first (fast), then verifies with pgvector distance.
        Falls back to distance-only matching if classifier not trained.
        
        Args:
            image: BGR numpy array (from kiosk camera)
        
        Returns:
            dict with:
                - 'success': bool
                - 'user_id': matched user ID (or None)
                - 'confidence': float (0-100 percentage)
                - 'embedding': 512D array (for logging/debugging)
                - 'error': error message (if failed)
        """
        start_time = time.time()
        try:
            # 1. Preprocess for detection
            processed = preprocess_for_detection(image)
            
            # 2. Detect face (largest/closest for kiosk)
            face = self.detector.detect_largest(processed)
            
            if face is None:
                return {
                    'success': False,
                    'error': 'Không phát hiện khuôn mặt! Vui lòng đảm bảo khuôn mặt rõ ràng, đủ sáng.',
                }
            
            # 3. Align face (160x160 for FaceNet)
            aligned = align_face(processed, face['landmarks'], output_size=160)
            
            # 4. Preprocess for embedding
            aligned = preprocess_for_embedding(aligned)
            
            # 5. Extract embedding
            embedding = self.embedder.extract(aligned)
            
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Không thể mã hóa khuôn mặt!',
                }
            
            # 6. Recall Phase: Vector Search for top candidate
            # T2: Cosine Similarity Threshold (Environment based, defaulting to 55.0)
            db_result = self._search_database(embedding)
            
            if not db_result:
                logger.info("Double Validation: Vector search found no candidates above threshold.")
                return {
                    'success': False,
                    'error': 'Unknown',
                    'embedding': embedding,
                }
                
            vector_user_id = db_result['user_id']
            cosine_sim = db_result['confidence']
            
            # 7. Precision Phase: SVM Classifier Verification
            if self.classifier.is_trained:
                svm_user_id, svm_conf = self.classifier.predict(embedding)
                svm_conf = round(svm_conf * 100, 1)
                
                # Double Validation check
                # T1: SVM Confidence Threshold (e.g., 50.0)
                svm_threshold = 50.0 
                cosine_threshold = 55.0
                cosine_high_threshold = 65.0 # T2_high for fallback
                
                if svm_conf >= svm_threshold and cosine_sim >= cosine_threshold and svm_user_id == vector_user_id:
                    # Both layers agree with high confidence
                    latency = round((time.time() - start_time) * 1000, 2)
                    logger.info(json.dumps({
                        "event": "recognition_success",
                        "latency_ms": latency,
                        "cosine": cosine_sim,
                        "svm": svm_conf,
                        "result": vector_user_id,
                        "validation": "double"
                    }))
                    return {
                        'success': True,
                        'user_id': vector_user_id,
                        'confidence': max(cosine_sim, svm_conf), # Return highest confidence
                        'embedding': embedding,
                    }
                elif cosine_sim >= cosine_high_threshold:
                    # Fallback: SVM failed but Vector is VERY confident (trust embedding strongly)
                    latency = round((time.time() - start_time) * 1000, 2)
                    logger.info(json.dumps({
                        "event": "recognition_success",
                        "latency_ms": latency,
                        "cosine": cosine_sim,
                        "svm": svm_conf,
                        "result": vector_user_id,
                        "validation": "vector_fallback"
                    }))
                    return {
                        'success': True,
                        'user_id': vector_user_id,
                        'confidence': cosine_sim,
                        'embedding': embedding,
                    }
                else:
                    # Conflict or low confidence -> Prevent False Positive
                    latency = round((time.time() - start_time) * 1000, 2)
                    logger.warning(json.dumps({
                        "event": "recognition_failed",
                        "reason": "double_validation_failed",
                        "latency_ms": latency,
                        "cosine": cosine_sim,
                        "svm": svm_conf,
                        "predicted_vector": vector_user_id,
                        "predicted_svm": svm_user_id
                    }))
                    return {
                        'success': False,
                        'error': 'Unknown',
                        'feedback_ui': 'Không thể xác nhận danh tính, vui lòng đứng thẳng và nhìn vào camera.',
                        'embedding': embedding,
                    }
            else:
                # Fallback if classifier is not trained yet (e.g., early system state)
                cosine_threshold = 55.0
                latency = round((time.time() - start_time) * 1000, 2)
                
                if cosine_sim >= cosine_threshold:
                    logger.info(json.dumps({
                        "event": "recognition_success",
                        "latency_ms": latency,
                        "cosine": cosine_sim,
                        "svm": None,
                        "result": vector_user_id,
                        "validation": "vector_only"
                    }))
                    return {
                        'success': True,
                        'user_id': vector_user_id,
                        'confidence': cosine_sim,
                        'embedding': embedding,
                    }
                else:
                    logger.warning(json.dumps({
                        "event": "recognition_failed",
                        "reason": "low_cosine_similarity",
                        "latency_ms": latency,
                        "cosine": cosine_sim,
                        "svm": None,
                        "predicted_vector": vector_user_id
                    }))
                    return {
                        'success': False,
                        'error': 'Unknown',
                        'feedback_ui': 'Không nhận ra bạn, vui lòng tiến lại gần hơn chút nữa.',
                        'embedding': embedding,
                    }
            
        except Exception as e:
            latency = round((time.time() - start_time) * 1000, 2)
            logger.error(json.dumps({
                "event": "recognition_error",
                "latency_ms": latency,
                "error": str(e)
            }), exc_info=True)
            return {
                'success': False,
                'error': f'Lỗi hệ thống: {str(e)}',
                'feedback_ui': 'Hệ thống đang bận, vui lòng thử lại.',
            }
    
    def _search_database(self, embedding: np.ndarray) -> dict:
        """
        Search for matching face in the database.
        Uses pgvector if available, otherwise falls back to Python distance loop.
        
        Returns:
            dict with 'user_id' and 'confidence', or None
        """
        try:
            # Try pgvector search first
            from attendance.models import FaceEmbedding
            from pgvector.django import CosineDistance
            
            # Query Optimization: Fetch top 10 closest vectors
            results = FaceEmbedding.objects.annotate(
                distance=CosineDistance('embedding', embedding.tolist())
            ).order_by('distance')[:10]
            
            if results:
                best_user_id = None
                best_weighted_conf = 0.0
                actual_conf_of_best = 0.0
                
                for res in results:
                    # Cosine distance: 0 = identical, 2 = opposite
                    confidence = (1 - res.distance / 2) * 100
                    
                    # Weighted Similarity: Boost confidence slightly based on quality_score
                    # E.g., if quality_score is 1.0, boost by 5%. 
                    weighted_conf = confidence + (res.quality_score * 5.0)
                    
                    if weighted_conf > best_weighted_conf:
                        best_weighted_conf = weighted_conf
                        best_user_id = res.user_id
                        actual_conf_of_best = confidence
                
                if actual_conf_of_best >= 55.0:
                    return {
                        'user_id': best_user_id,
                        'confidence': round(actual_conf_of_best, 1),
                    }
        except Exception as e:
            logger.warning(f"pgvector search failed, falling back to Python: {e}")
        
        # Fallback: Python-based distance comparison
        return self._search_python_fallback(embedding)
    
    def _search_python_fallback(self, embedding: np.ndarray) -> dict:
        """
        Fallback face matching using Python loop over cached staff encodings.
        Used when pgvector is not available.
        """
        from attendance.services.cache_service import FaceCacheService
        
        all_cached_embeddings = FaceCacheService.get_all_embeddings()
        
        best_match_id = None
        best_similarity = 55.0  # Minimum confidence threshold in percent
        
        for known_encoding, user_id in all_cached_embeddings:
            # Cosine similarity between embeddings
                similarity = float(np.dot(embedding, known_encoding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_encoding) + 1e-8
                ))
                
                confidence = round(similarity * 100, 1)
                
                if confidence > best_similarity:
                    best_similarity = confidence
                    best_match_id = user_id
        
        if best_match_id is not None:
            return {
                'user_id': best_match_id,
                'confidence': best_similarity,
            }
        
        return None
    
    # ==================== CLASSIFIER MANAGEMENT ====================
    
    def retrain_classifier(self) -> bool:
        """
        Retrain the SVM classifier using all face embeddings in the database.
        
        Returns:
            True if training succeeded
        """
        try:
            from attendance.models import FaceEmbedding
            
            embeddings_qs = FaceEmbedding.objects.all().values_list('embedding', 'user_id')
            
            if not embeddings_qs:
                # Try fallback from User.face_encoding_text
                return self._retrain_from_user_encodings()
            
            embeddings = []
            user_ids = []
            
            for emb, uid in embeddings_qs:
                embeddings.append(np.array(emb))
                user_ids.append(uid)
            
            return self.classifier.train(embeddings, user_ids)
            
        except Exception as e:
            logger.error(f"Classifier retraining failed: {e}", exc_info=True)
            return self._retrain_from_user_encodings()
    
    def _retrain_from_user_encodings(self) -> bool:
        """Fallback: train classifier from User.face_encoding_text field."""
        from attendance.models import User
        
        staff = User.objects.filter(
            role=User.Role.STAFF
        ).exclude(face_encoding_text__isnull=True).exclude(face_encoding_text='')
        
        embeddings = []
        user_ids = []
        
        for s in staff:
            enc = s.get_encoding()
            if enc is not None:
                embeddings.append(enc)
                user_ids.append(s.id)
        
        if len(embeddings) < 2 or len(set(user_ids)) < 2:
            logger.warning("Not enough staff with face encodings to train classifier")
            return False
        
        return self.classifier.train(embeddings, user_ids)
