from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2hsv, rgb2gray
import numpy as np
import cv2
from .base import FeatureExtractor


@FeatureExtractor.register("hsv")
class HSVHistogramExtractor(FeatureExtractor):
    """Extracts color information using HSV color space histograms.

    HSV color space is particularly useful for satellite imagery because it separates
    color information (Hue) from illumination (Value), making it more robust to the
    extreme lighting conditions encountered in space photography.
    """

    def __init__(
        self,
        h_bins: int = 30,  # More bins for hue to capture subtle color differences
        s_bins: int = 32,  # Balanced bins for saturation
        v_bins: int = 32,  # Balanced bins for value
        **kwargs
    ):
        super().__init__(h_bins=h_bins, s_bins=s_bins, v_bins=v_bins, **kwargs)
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract HSV histogram features from an image.

        Args:
            image: RGB image as numpy array

        Returns:
            Concatenated HSV histogram features
        """
        # Convert to HSV color space using OpenCV for better performance
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create histogram ranges for each channel
        h_range = (0, 180)  # OpenCV uses 0-180 for Hue
        s_range = (0, 256)  # OpenCV uses 0-255 for Saturation
        v_range = (0, 256)  # OpenCV uses 0-255 for Value

        # Compute histograms for each channel using OpenCV
        h_hist = cv2.calcHist([hsv], [0], None, [self.h_bins], h_range)
        s_hist = cv2.calcHist([hsv], [1], None, [self.s_bins], s_range)
        v_hist = cv2.calcHist([hsv], [2], None, [self.v_bins], v_range)

        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()

        # Concatenate histograms
        return np.concatenate([h_hist, s_hist, v_hist])

    @property
    def name(self) -> str:
        return "HSV_Histogram"
