"""
Data Augmentation Service.

From 1 registration image, generates multiple variants to create a robust
set of embeddings. This compensates for having very few training images.

Augmentations applied:
- Horizontal flip
- Random rotation (±15°)
- Brightness/contrast adjustments
- Slight Gaussian blur
- Gamma correction
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def augment_face(aligned_face: np.ndarray, num_augmented: int = 20) -> list:
    """
    Generate augmented variants of an aligned face image.
    
    Args:
        aligned_face: BGR numpy array (aligned face, e.g. 112x112)
        num_augmented: Number of augmented variants to generate
    
    Returns:
        List of BGR numpy arrays (original + augmented variants)
    """
    if aligned_face is None:
        return []
    
    variants = [aligned_face.copy()]  # Include original
    
    h, w = aligned_face.shape[:2]
    rng = np.random.RandomState(42)  # Reproducible augmentation
    
    for i in range(num_augmented):
        img = aligned_face.copy()
        
        # 1. Random horizontal flip (50% chance)
        if rng.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # 2. Random rotation (±12 degrees)
        angle = rng.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # 3. Random brightness adjustment (±25%)
        brightness_factor = rng.uniform(0.75, 1.25)
        img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
        
        # 4. Random contrast adjustment
        if rng.random() > 0.5:
            contrast_factor = rng.uniform(0.8, 1.2)
            mean = np.mean(img)
            img = np.clip((img - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # 5. Random slight blur (20% chance)
        if rng.random() > 0.8:
            kernel_size = rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # 6. Random gamma correction (30% chance)
        if rng.random() > 0.7:
            gamma = rng.uniform(0.7, 1.3)
            inv_gamma = 1.0 / gamma
            table = np.array([((j / 255.0) ** inv_gamma) * 255
                              for j in range(256)]).astype("uint8")
            img = cv2.LUT(img, table)
        
        # 7. Random noise (10% chance)
        if rng.random() > 0.9:
            noise = rng.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        variants.append(img)
    
    logger.info(f"Generated {len(variants)} face variants (1 original + {num_augmented} augmented)")
    return variants
