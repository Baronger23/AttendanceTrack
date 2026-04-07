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
        
        # MTCNN with tuned parameters for attendance system
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,           # Minimum face size in pixels (lowered for webcam)
            thresholds=[0.5, 0.6, 0.6], # P-Net, R-Net, O-Net thresholds (lowered for robustness)
            factor=0.709,               # Scale factor for image pyramid
            select_largest=False,       # Don't auto-select, we handle it
            keep_all=True,              # Return all detected faces
            device=self.device,
            post_process=False,         # Don't normalize (we do it ourselves)
        )
        
        self._initialized = True
        logger.info(f"MTCNN initialized on {self.device}")
    
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
        
        # Convert BGR numpy to PIL Image (RGB) — required by facenet-pytorch MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Detect faces
        try:
            boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return []
        
        if boxes is None or probs is None:
            return []
        
        results = []
        for i in range(len(boxes)):
            if probs[i] is None or probs[i] < 0.5:
                continue
            
            box = boxes[i].astype(int).tolist()
            landmark = landmarks[i] if landmarks is not None else None
            
            results.append({
                'box': box,  # [x1, y1, x2, y2]
                'confidence': float(probs[i]),
                'landmarks': landmark,  # (5, 2) array
            })
        
        # Sort by confidence (highest first)
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
        
        # Find the face with largest bounding box area
        def box_area(face):
            x1, y1, x2, y2 = face['box']
            return max(0, x2 - x1) * max(0, y2 - y1)
        
        return max(faces, key=box_area)
