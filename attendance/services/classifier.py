"""
Face Classifier Service using SVM (Support Vector Machine).

Instead of manually comparing distances to every person in the database,
the SVM learns "decision boundaries" between staff members, making
classification much faster and more accurate.

The classifier is trained on embeddings from all registered staff.
"""

import os
import numpy as np
import joblib
import logging
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from django.conf import settings

logger = logging.getLogger(__name__)

# Directory to store trained classifier models
MODELS_DIR = os.path.join(settings.BASE_DIR, 'models')


class FaceClassifier:
    """SVM-based face classifier with probability estimates."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.model_path = os.path.join(MODELS_DIR, 'face_classifier.joblib')
        self.encoder_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')
        
        # Try to load existing model
        self._load_model()
    
    def train(self, embeddings: list, user_ids: list) -> bool:
        """
        Train the SVM classifier on face embeddings.
        
        Args:
            embeddings: List of 512D numpy arrays
            user_ids: List of user IDs (integer) corresponding to each embedding
        
        Returns:
            True if training succeeded
        """
        if len(embeddings) < 2 or len(set(user_ids)) < 2:
            logger.warning("Need at least 2 different people to train classifier")
            return False
        
        X = np.array(embeddings)
        y = np.array(user_ids)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train SVM with probability estimates
        self.model = SVC(
            kernel='rbf',
            probability=True,      # Enable probability estimates for confidence
            C=10.0,                 # Regularization parameter
            gamma='scale',          # Kernel coefficient
            class_weight='balanced', # Handle imbalanced classes
        )
        
        self.model.fit(X, y_encoded)
        
        # Save model
        self._save_model()
        
        num_classes = len(self.label_encoder.classes_)
        logger.info(f"Classifier trained: {len(X)} samples, {num_classes} classes")
        
        return True
    
    def predict(self, embedding: np.ndarray) -> tuple:
        """
        Classify a face embedding.
        
        Args:
            embedding: 512D numpy array
        
        Returns:
            (user_id, confidence) tuple, or (None, 0.0) if no model loaded
        """
        if self.model is None or self.label_encoder is None:
            logger.warning("No classifier model loaded — falling back to distance matching")
            return None, 0.0
        
        X = embedding.reshape(1, -1)
        
        # Predict class and probability
        y_pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        # Get the confidence for the predicted class
        confidence = float(proba[y_pred])
        
        # Decode label back to user_id
        user_id = int(self.label_encoder.inverse_transform([y_pred])[0])
        
        return user_id, confidence
    
    def _save_model(self):
        """Save trained model to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.label_encoder, self.encoder_path)
        
        logger.info(f"Classifier saved to {self.model_path}")
    
    def _load_model(self):
        """Load trained model from disk (if exists)."""
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            try:
                self.model = joblib.load(self.model_path)
                self.label_encoder = joblib.load(self.encoder_path)
                logger.info("Classifier loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
                self.model = None
                self.label_encoder = None
    
    @property
    def is_trained(self) -> bool:
        """Check if the classifier is ready for predictions."""
        return self.model is not None and self.label_encoder is not None
