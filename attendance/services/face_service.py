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
            
            # 3. Align face using landmarks
            aligned = align_face(processed, face['landmarks'], output_size=112)
            
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
            
            # 7. Average embedding (robust representation)
            avg_embedding = np.mean(all_embeddings, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            return {
                'success': True,
                'embedding': avg_embedding,
                'all_embeddings': all_embeddings,
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
        detection_confidence = 0.0
        
        for img in images:
            result = self.register_face(img, num_augmented=num_augmented)
            if result['success']:
                all_embeddings.extend(result['all_embeddings'])
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
            
            # 3. Align face
            aligned = align_face(processed, face['landmarks'], output_size=112)
            
            # 4. Preprocess for embedding
            aligned = preprocess_for_embedding(aligned)
            
            # 5. Extract embedding
            embedding = self.embedder.extract(aligned)
            
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Không thể mã hóa khuôn mặt!',
                }
            
            # 6. Try SVM classifier first (fast)
            user_id = None
            confidence = 0.0
            
            if self.classifier.is_trained:
                user_id, confidence = self.classifier.predict(embedding)
                confidence = round(confidence * 100, 1)
                
                if confidence < 50.0:
                    # SVM not confident enough, will fall back to DB search
                    user_id = None
            
            # 7. If classifier failed or not trained, use pgvector DB search
            if user_id is None:
                db_result = self._search_database(embedding)
                if db_result:
                    user_id = db_result['user_id']
                    confidence = db_result['confidence']
            
            if user_id is None:
                return {
                    'success': False,
                    'error': 'Không nhận diện được! Khuôn mặt không khớp với bất kỳ nhân viên nào.',
                    'embedding': embedding,
                }
            
            return {
                'success': True,
                'user_id': user_id,
                'confidence': confidence,
                'embedding': embedding,
            }
            
        except Exception as e:
            logger.error(f"Identification failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Lỗi nhận diện: {str(e)}',
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
            
            results = FaceEmbedding.objects.annotate(
                distance=CosineDistance('embedding', embedding.tolist())
            ).order_by('distance')[:1]
            
            if results:
                best = results[0]
                # Cosine distance: 0 = identical, 2 = opposite
                # Convert to confidence: (1 - distance/2) * 100
                confidence = round((1 - best.distance / 2) * 100, 1)
                
                if confidence >= 55.0:
                    return {
                        'user_id': best.user_id,
                        'confidence': confidence,
                    }
        except Exception as e:
            logger.warning(f"pgvector search failed, falling back to Python: {e}")
        
        # Fallback: Python-based distance comparison
        return self._search_python_fallback(embedding)
    
    def _search_python_fallback(self, embedding: np.ndarray) -> dict:
        """
        Fallback face matching using Python loop over all staff encodings.
        Used when pgvector is not available.
        """
        from attendance.models import User
        
        staff_members = User.objects.filter(
            role=User.Role.STAFF
        ).exclude(face_encoding_text__isnull=True).exclude(face_encoding_text='')
        
        best_match = None
        best_similarity = 0.55  # Minimum threshold (55%)
        
        for staff in staff_members:
            known_encoding = staff.get_encoding()
            if known_encoding is not None:
                # Cosine similarity between embeddings
                similarity = float(np.dot(embedding, known_encoding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_encoding) + 1e-8
                ))
                
                confidence = round(similarity * 100, 1)
                
                if confidence > best_similarity:
                    best_similarity = confidence
                    best_match = staff
        
        if best_match is not None:
            return {
                'user_id': best_match.id,
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
