from typing import List, Type, Dict
from .base import FeatureExtractor

# Import implementations to ensure decorators are executed
from . import hog  # This ensures the HOG implementation is registered
from . import hsv
from . import lbp
from . import resnet
from . import combined

def get_feature_extractor(name: str, **kwargs) -> FeatureExtractor:
    """Get a feature extractor instance by name.
    
    Args:
        name: Name of the registered feature extractor
        **kwargs: Parameters to pass to the feature extractor
        
    Returns:
        Instantiated feature extractor
        
    Raises:
        KeyError: If no feature extractor is registered with the given name
    """
    try:
        extractor_cls = FeatureExtractor.get_registry().get(name)
        return extractor_cls(**kwargs)
    except KeyError:
        available = FeatureExtractor.get_registry().list()
        raise KeyError(
            f"No feature extractor registered as '{name}'. "
            f"Available extractors: {available}"
        )

def list_feature_extractors() -> List[str]:
    """List all registered feature extractors."""
    return FeatureExtractor.get_registry().list()
