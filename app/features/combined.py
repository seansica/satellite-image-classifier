import numpy as np

from .base import FeatureExtractor
from .hog import HOGFeatureExtractor
from .hsv import HSVHistogramExtractor
from .lbp import LBPFeatureExtractor


@FeatureExtractor.register("combined")
class CombinedFeatureExtractor(FeatureExtractor):
    """Combines multiple feature extractors for richer image representation.

    This extractor concatenates HOG, HSV histogram, and LBP features to capture
    shape, color, and texture information simultaneously. Each feature type
    provides complementary information that can help in classification:
    - HOG: Captures overall shape and gradient patterns
    - HSV: Captures color distribution and illumination
    - LBP: Captures local texture patterns
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize individual feature extractors
        self.hog_extractor = HOGFeatureExtractor()
        self.hsv_extractor = HSVHistogramExtractor()
        self.lbp_extractor = LBPFeatureExtractor()

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract combined features from an image.

        Args:
            image: RGB image as numpy array

        Returns:
            Concatenated feature vector of HOG, HSV, and LBP features
        """
        # Extract individual features
        hog_features = self.hog_extractor.extract(image)
        hsv_features = self.hsv_extractor.extract(image)
        lbp_features = self.lbp_extractor.extract(image)

        # Normalize each feature set independently
        hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-7)
        hsv_features = hsv_features / (np.linalg.norm(hsv_features) + 1e-7)
        lbp_features = lbp_features / (np.linalg.norm(lbp_features) + 1e-7)

        # Concatenate all features
        return np.concatenate([hog_features, hsv_features, lbp_features])

    @property
    def name(self) -> str:
        return "Combined"
