"""
Image Preprocessing Service.

Advanced preprocessing pipeline:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) — superior to basic equalizeHist
2. LAB colorspace brightness normalization
3. Resize standardization
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def preprocess_for_detection(image: np.ndarray, max_dimension: int = 1000) -> np.ndarray:
    """
    Preprocess image before face detection.
    Resize very large images and normalize brightness.
    
    Args:
        image: BGR numpy array
        max_dimension: Maximum width/height
    
    Returns:
        Preprocessed BGR image
    """
    if image is None:
        return None
    
    # Resize if too large (keeps aspect ratio)
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE for brightness normalization
    image = apply_clahe(image)
    
    return image


def preprocess_for_embedding(aligned_face: np.ndarray, clahe_threshold: float = 90.0) -> np.ndarray:
    """
    Preprocess an aligned face before embedding extraction.
    Only applies CLAHE if the face is too dark (average brightness < clahe_threshold)
    to preserve FaceNet features in well-lit conditions.
    
    Args:
        aligned_face: BGR numpy array, typically 112x112
        clahe_threshold: Brightness threshold below which CLAHE is applied
    
    Returns:
        Preprocessed BGR image ready for embedding model
    """
    if aligned_face is None:
        return None
    
    # Calculate average brightness
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Apply CLAHE conditionally
    if mean_brightness < clahe_threshold:
        logger.debug(f"Applying CLAHE (Brightness: {mean_brightness:.1f} < {clahe_threshold})")
        result = apply_clahe(aligned_face)
    else:
        logger.debug(f"Skipping CLAHE (Brightness: {mean_brightness:.1f} >= {clahe_threshold})")
        result = aligned_face
        
    return result


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE is superior to basic equalizeHist because:
    - It divides the image into tiles and equalizes each independently
    - Clip limit prevents over-amplification of noise
    - Works well in both dark and bright environments
    
    Args:
        image: BGR numpy array
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
    
    Returns:
        CLAHE-enhanced BGR image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel (luminance)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return result


def normalize_brightness(image: np.ndarray, target_mean: float = 127.0) -> np.ndarray:
    """
    Normalize overall image brightness to a target mean.
    
    Args:
        image: BGR numpy array
        target_mean: Target mean brightness (0-255)
    
    Returns:
        Brightness-normalized image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_mean = np.mean(gray)
    
    if current_mean == 0:
        return image
    
    # Scale brightness
    factor = target_mean / current_mean
    result = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    return result
