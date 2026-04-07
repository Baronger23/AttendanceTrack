"""
Face Embedding Service using InceptionResnetV1 (VGGFace2 pre-trained).

Produces 512-dimensional face embeddings (vectors) that encode facial identity.
This is Transfer Learning: the model was trained on millions of faces and we
use it as-is to extract features, then train our own classifier on top.
"""

import numpy as np
import cv2
import torch
import logging
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Singleton face embedding extractor using InceptionResnetV1."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # InceptionResnetV1 pretrained on VGGFace2
        # Outputs 512D embeddings
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self._initialized = True
        logger.info(f"InceptionResnetV1 (VGGFace2) loaded on {self.device}")
    
    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract 512D face embedding from an aligned face image.
        
        Args:
            aligned_face: BGR numpy array, ideally 112x112 or 160x160 aligned face
        
        Returns:
            512D numpy array (L2-normalized embedding)
        """
        if aligned_face is None:
            return None
        
        # Convert BGR → RGB
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (160x160 for InceptionResnetV1)
        rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
        
        # Normalize pixel values to [-1, 1] (standard for this model)
        tensor = torch.FloatTensor(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 160, 160)
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(tensor)
        
        # Convert to numpy and L2-normalize
        embedding = embedding.cpu().numpy().flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def extract_batch(self, aligned_faces: list) -> list:
        """
        Extract embeddings for multiple faces efficiently (batch processing).
        
        Args:
            aligned_faces: List of BGR numpy arrays
        
        Returns:
            List of 512D numpy arrays
        """
        if not aligned_faces:
            return []
        
        tensors = []
        for face in aligned_faces:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
            tensor = torch.FloatTensor(rgb).permute(2, 0, 1)
            tensor = (tensor - 127.5) / 128.0
            tensors.append(tensor)
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(batch)
        
        embeddings = embeddings.cpu().numpy()
        
        # L2-normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return [emb for emb in embeddings]
