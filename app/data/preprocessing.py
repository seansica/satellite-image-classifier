from typing import Optional
import numpy as np
import cv2

from ..core.types import ImageArray

def preprocess_image(
    image: ImageArray,
    normalize: bool = True,
    enhance_contrast: bool = True
) -> ImageArray:
    """Apply preprocessing steps to an image.
    
    This function applies several preprocessing steps that can improve
    model performance on satellite imagery:
    1. Optional contrast enhancement
    2. Optional pixel value normalization
    
    Args:
        image: Input image
        normalize: Whether to normalize pixel values to [0, 1]
        enhance_contrast: Whether to apply contrast enhancement
        
    Returns:
        Preprocessed image
    """
    # Work on a copy to avoid modifying the original
    img = image.copy()
    
    if enhance_contrast:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        
        # Convert back to RGB
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    if normalize:
        # Normalize to [0, 1] range
        img = img.astype(np.float32) / 255.0
    
    return img
