from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

from ..core.types import ImageArray, FeatureArray
from ..core.base import RegistryMixin

class FeatureExtractor(RegistryMixin, ABC):
    """Base class for all feature extractors."""
    
    def __init__(self, **kwargs):
        """Initialize feature extractor with optional parameters."""
        self.params = kwargs
    
    @abstractmethod
    def extract(self, image: ImageArray) -> FeatureArray:
        """Extract features from a single image."""
        pass
    
    def extract_batch(self, images: List[ImageArray]) -> List[FeatureArray]:
        """Extract features from a batch of images."""
        return [self.extract(img) for img in images]
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the feature extractor."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used by this feature extractor."""
        return self.params.copy()