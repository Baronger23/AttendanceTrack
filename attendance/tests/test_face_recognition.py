from django.test import TestCase
from unittest.mock import patch, MagicMock
import numpy as np

from attendance.services.face_service import FaceService

class FaceServiceDoubleValidationTest(TestCase):
    def setUp(self):
        # Create a real FaceService instance but mock its internal components
        self.face_service = FaceService()
        self.face_service.detector = MagicMock()
        self.face_service.embedder = MagicMock()
        self.face_service.classifier = MagicMock()
        
        # Setup mock dummy face
        self.dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.dummy_embedding = np.random.rand(512).astype(np.float32)
        
        # Mock detection pipeline to always succeed
        self.face_service.detector.detect_largest.return_value = {
            'landmarks': [[1,2], [3,4], [5,6], [7,8], [9,10]],
            'confidence': 0.99
        }
        self.face_service.embedder.extract.return_value = self.dummy_embedding

    @patch('attendance.services.face_service.preprocess_for_detection')
    @patch('attendance.services.face_service.align_face')
    @patch('attendance.services.face_service.preprocess_for_embedding')
    @patch.object(FaceService, '_search_database')
    def test_identify_face_success(self, mock_db_search, mock_pre_emb, mock_align, mock_pre_det):
        """
        Test Double Validation: BOTH Vector Search AND SVM agree with high confidence.
        Should return the user ID.
        """
        # 1. Setup mock responses
        user_id = 1
        
        # Recall Phase (Vector Search) -> Returns User 1 with high confidence
        mock_db_search.return_value = {
            'user_id': user_id,
            'confidence': 60.0 # T2 = 55.0
        }
        
        # Precision Phase (SVM) -> Returns User 1 with high confidence
        self.face_service.classifier.is_trained = True
        self.face_service.classifier.predict.return_value = (user_id, 0.75) # T1 = 0.5
        
        # 2. Execute
        result = self.face_service.identify_face(self.dummy_image)
        
        # 3. Assert
        self.assertTrue(result['success'])
        self.assertEqual(result['user_id'], user_id)
        # Final confidence could be an average or just the SVM's
        # We will design the new logic to return something sensible
        
    @patch('attendance.services.face_service.preprocess_for_detection')
    @patch('attendance.services.face_service.align_face')
    @patch('attendance.services.face_service.preprocess_for_embedding')
    @patch.object(FaceService, '_search_database')
    def test_identify_face_false_positive_prevented(self, mock_db_search, mock_pre_emb, mock_align, mock_pre_det):
        """
        Test Double Validation: Vector Search finds User 1, but SVM predicts User 2.
        This is a conflict. Should return Unknown.
        """
        # Recall Phase (Vector Search) -> Returns User 1
        mock_db_search.return_value = {
            'user_id': 1,
            'confidence': 60.0
        }
        
        # Precision Phase (SVM) -> Predicts User 2! (Conflict)
        self.face_service.classifier.is_trained = True
        self.face_service.classifier.predict.return_value = (2, 0.8)
        
        # Execute
        result = self.face_service.identify_face(self.dummy_image)
        
        # Assert - Must fail to prevent False Positive
        self.assertFalse(result['success'])
        self.assertEqual(result.get('error'), 'Unknown')
        
    @patch('attendance.services.face_service.preprocess_for_detection')
    @patch('attendance.services.face_service.align_face')
    @patch('attendance.services.face_service.preprocess_for_embedding')
    @patch.object(FaceService, '_search_database')
    def test_identify_face_low_svm_confidence(self, mock_db_search, mock_pre_emb, mock_align, mock_pre_det):
        """
        Test Double Validation: Both agree on User 1, but SVM confidence is too low.
        Should return Unknown.
        """
        user_id = 1
        
        mock_db_search.return_value = {
            'user_id': user_id,
            'confidence': 60.0
        }
        
        # SVM predicts User 1, but confidence is weak (e.g., 0.4 < 0.5)
        self.face_service.classifier.is_trained = True
        self.face_service.classifier.predict.return_value = (user_id, 0.4)
        
        # Execute
        result = self.face_service.identify_face(self.dummy_image)
        
        # Assert
        self.assertFalse(result['success'])
        self.assertEqual(result.get('error'), 'Unknown')
        
    @patch('attendance.services.face_service.preprocess_for_detection')
    @patch('attendance.services.face_service.align_face')
    @patch('attendance.services.face_service.preprocess_for_embedding')
    @patch.object(FaceService, '_search_database')
    def test_identify_face_low_cosine_similarity(self, mock_db_search, mock_pre_emb, mock_align, mock_pre_det):
        """
        Test Double Validation: SVM predicts User 1 with high confidence, but Vector search is weak.
        Should return Unknown.
        """
        user_id = 1
        
        # Vector search finds User 1, but confidence is low (e.g., 40.0 < 55.0)
        mock_db_search.return_value = {
            'user_id': user_id,
            'confidence': 40.0
        }
        
        # SVM predicts User 1 with high confidence
        self.face_service.classifier.is_trained = True
        self.face_service.classifier.predict.return_value = (user_id, 0.8)
        
        # Execute
        result = self.face_service.identify_face(self.dummy_image)
        
        # Assert
        self.assertFalse(result['success'])
        self.assertEqual(result.get('error'), 'Unknown')

    @patch('attendance.services.face_service.preprocess_for_detection')
    @patch('attendance.services.face_service.align_face')
    @patch('attendance.services.face_service.preprocess_for_embedding')
    @patch.object(FaceService, '_search_database')
    def test_identify_face_unknown_person(self, mock_db_search, mock_pre_emb, mock_align, mock_pre_det):
        """
        Test Double Validation: Vector search returns no match (None).
        Should return Unknown immediately.
        """
        mock_db_search.return_value = None
        
        self.face_service.classifier.is_trained = True
        self.face_service.classifier.predict.return_value = (1, 0.2) # Doesn't matter
        
        # Execute
        result = self.face_service.identify_face(self.dummy_image)
        
        # Assert
        self.assertFalse(result['success'])
        self.assertEqual(result.get('error'), 'Unknown')


class FaceServiceFallbackSearchTest(TestCase):
    def test_python_fallback_requires_55_percent_confidence(self):
        face_service = FaceService.__new__(FaceService)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        known = np.array([0.54, np.sqrt(1 - 0.54 ** 2), 0.0], dtype=np.float32)

        with patch(
            'attendance.services.cache_service.FaceCacheService.get_all_embeddings',
            return_value=[(known, 123)],
        ):
            result = face_service._search_python_fallback(query)

        self.assertIsNone(result)

    def test_python_fallback_accepts_match_above_55_percent(self):
        face_service = FaceService.__new__(FaceService)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        known = np.array([0.56, np.sqrt(1 - 0.56 ** 2), 0.0], dtype=np.float32)

        with patch(
            'attendance.services.cache_service.FaceCacheService.get_all_embeddings',
            return_value=[(known, 123)],
        ):
            result = face_service._search_python_fallback(query)

        self.assertEqual(result['user_id'], 123)
