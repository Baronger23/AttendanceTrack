"""
Face Alignment Service.

Uses 5-point facial landmarks (from MTCNN) to perform affine transformation,
aligning the face so both eyes are horizontal. This standardized 112x112 crop
dramatically improves embedding consistency.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# Standard landmark positions for 112x112 aligned face
# These are the "ideal" positions where eyes, nose, and mouth should land
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # mouth left
    [70.7299, 92.2041],   # mouth right
], dtype=np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """
    Align a face using 5-point landmarks via similarity transform.
    
    Args:
        image: BGR numpy array
        landmarks: (5, 2) array of facial landmark coordinates
            [left_eye, right_eye, nose, mouth_left, mouth_right]
        output_size: Output image size (default 112 for ArcFace-compatible models)
    
    Returns:
        Aligned face image of shape (output_size, output_size, 3)
    """
    if landmarks is None or len(landmarks) < 5:
        logger.warning("Not enough landmarks for alignment, using center crop")
        return _center_crop(image, output_size)
    
    src_pts = np.array(landmarks, dtype=np.float32)
    dst_pts = REFERENCE_LANDMARKS_112.copy()
    
    # Scale reference landmarks if output_size != 112
    if output_size != 112:
        scale = output_size / 112.0
        dst_pts *= scale
    
    # Estimate similarity transform (rotation + scale + translation)
    transform_matrix = _estimate_similarity_transform(src_pts, dst_pts)
    
    # Apply the affine transformation
    aligned = cv2.warpAffine(
        image,
        transform_matrix,
        (output_size, output_size),
        borderMode=cv2.BORDER_REPLICATE,
    )
    
    return aligned


def _estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate a 2D similarity transform (rotation, scale, translation)
    from source points to destination points using least squares.
    
    Returns:
        2x3 affine transformation matrix
    """
    num = src.shape[0]
    
    # Build system of equations: [X] * [params] = [Y]
    # Where params = [a, b, tx, ty] for the transform:
    #   x' = a*x - b*y + tx
    #   y' = b*x + a*y + ty
    X = np.zeros((num * 2, 4))
    Y = np.zeros(num * 2)
    
    for i in range(num):
        X[2*i]     = [src[i, 0], -src[i, 1], 1, 0]
        X[2*i + 1] = [src[i, 1],  src[i, 0], 0, 1]
        Y[2*i]     = dst[i, 0]
        Y[2*i + 1] = dst[i, 1]
    
    # Solve using least squares
    params, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    a, b, tx, ty = params
    
    # Build 2x3 affine matrix
    M = np.array([
        [a, -b, tx],
        [b,  a, ty],
    ], dtype=np.float64)
    
    return M


def _center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Fallback: center crop and resize when landmarks are unavailable."""
    h, w = image.shape[:2]
    min_dim = min(h, w)
    
    # Center crop to square
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped = image[top:top + min_dim, left:left + min_dim]
    
    # Resize to target size
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
