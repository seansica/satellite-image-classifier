from skimage.feature import hog
from skimage.color import rgb2gray

from .base import FeatureExtractor
from ..core.types import ImageArray, FeatureArray

@FeatureExtractor.register('hog')
class HOGFeatureExtractor(FeatureExtractor):
    """Histogram of Oriented Gradients (HOG) feature extractor."""
    
    def __init__(
        self,
        orientations: int = 8,
        pixels_per_cell: tuple[int, int] = (16, 16),
        cells_per_block: tuple[int, int] = (2, 2),
        **kwargs
    ):
        super().__init__(
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            **kwargs
        )
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def extract(self, image: ImageArray) -> FeatureArray:
        """Extract HOG features from an image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            HOG feature vector as numpy array
        """
        # Convert to grayscale for HOG feature extraction
        gray = rgb2gray(image)
        
        # Extract HOG features
        features = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            feature_vector=True
        )
        
        return features
    
    @property
    def name(self) -> str:
        return "HOG"