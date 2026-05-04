"""
Face Detection Service using MTCNN (Multi-task Cascaded Convolutional Networks).

MTCNN detects faces even when partially occluded (masks) or at angles,
unlike HOG which only works well on frontal faces.

Returns: bounding boxes + 5-point facial landmarks (eyes, nose, mouth corners)
"""

import numpy as np
import cv2
import logging
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import warnings

# Suppress TF logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

try:
    # Monkeypatch for Keras 3.x compatibility with RetinaFace
    import tensorflow as tf
    try:
        from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
    except ImportError:
        try:
            from tensorflow.keras.layers import MaxPooling2D
        except ImportError:
            MaxPooling2D = None
            
    if MaxPooling2D is not None:
        original_init = MaxPooling2D.__init__
        def patched_init(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
            if isinstance(padding, str):
                padding = padding.lower()
            original_init(self, pool_size=pool_size, strides=strides, padding=padding, **kwargs)
        MaxPooling2D.__init__ = patched_init

    from retinaface import RetinaFace
    RETINA_AVAILABLE = True
except ImportError:
    RETINA_AVAILABLE = False

logger = logging.getLogger(__name__)


class FaceDetector:
    """Singleton MTCNN-based face detector."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.5, 0.6, 0.6],
            factor=0.709,
            select_largest=False,
            keep_all=True,
            device=self.device,
            post_process=False,
        )
        
        self.use_retina = RETINA_AVAILABLE
        self._initialized = True
        logger.info(f"Detector initialized (RetinaFace: {self.use_retina}, Fallback MTCNN: {self.device})")
    
    def detect(self, image: np.ndarray):
        """
        Detect faces in an image.
        
        Args:
            image: BGR numpy array (from cv2.imread or cv2.imdecode)
        
        Returns:
            list of dicts, each containing:
                - 'box': [x1, y1, x2, y2] bounding box
                - 'confidence': float detection confidence
                - 'landmarks': np.array of shape (5, 2) — 
                  [left_eye, right_eye, nose, mouth_left, mouth_right]
        """
        if image is None:
            return []
            
        results = []
        
        if self.use_retina:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = RetinaFace.detect_faces(rgb_image)
                if isinstance(faces, dict):
                    for key, face in faces.items():
                        score = face.get("score", 0.0)
                        if score < 0.5: continue
                        
                        box = face["facial_area"]  # [x1, y1, x2, y2]
                        landmarks_dict = face.get("landmarks", {})
                        
                        # Extract 5 points
                        if len(landmarks_dict) == 5:
                            landmark = np.array([
                                landmarks_dict["left_eye"],
                                landmarks_dict["right_eye"],
                                landmarks_dict["nose"],
                                landmarks_dict["mouth_left"],
                                landmarks_dict["mouth_right"]
                            ], dtype=np.float32)
                        else:
                            landmark = None
                            
                        results.append({
                            'box': box,
                            'confidence': float(score),
                            'landmarks': landmark
                        })
                    
                    if results:
                        results.sort(key=lambda x: x['confidence'], reverse=True)
                        return results
            except Exception as e:
                logger.error(f"RetinaFace detection failed, falling back to MTCNN: {e}")
        
        # Fallback to MTCNN
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
            
            if boxes is not None and probs is not None:
                for i in range(len(boxes)):
                    if probs[i] is None or probs[i] < 0.5:
                        continue
                    
                    box = boxes[i].astype(int).tolist()
                    landmark = landmarks[i] if landmarks is not None else None
                    if landmark is not None and len(landmark) != 5:
                        landmark = None
                        
                    results.append({
                        'box': box,
                        'confidence': float(probs[i]),
                        'landmarks': landmark,
                    })
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results
    
    def detect_largest(self, image: np.ndarray):
        """
        Detect the largest (closest) face in the image.
        Best for kiosk check-in where only 1 person is expected.
        
        Returns:
            dict with 'box', 'confidence', 'landmarks' or None
        """
        faces = self.detect(image)
        
        if not faces:
            return None
            
        img_h, img_w = image.shape[:2]
        center_x, center_y = img_w / 2, img_h / 2
        
        # Select best face by Area and Distance to Center
        def evaluate_face(face):
            x1, y1, x2, y2 = face['box']
            area = max(0, x2 - x1) * max(0, y2 - y1)
            
            face_cx = (x1 + x2) / 2
            face_cy = (y1 + y2) / 2
            
            # Distance from center of image
            dist_to_center = np.sqrt((face_cx - center_x)**2 + (face_cy - center_y)**2)
            
            # Combine area and distance (heuristic: prioritize area, but penalize distance heavily if multiple faces)
            score = area - (dist_to_center * 50)
            return score
            
        return max(faces, key=evaluate_face)
