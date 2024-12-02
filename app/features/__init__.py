"""Feature extraction components."""

from .base import FeatureExtractor
from .hog import HOGFeatureExtractor  # Import concrete implementation
from .registry import get_feature_extractor, list_feature_extractors

__all__ = [
    'FeatureExtractor',
    'HOGFeatureExtractor',
    'get_feature_extractor',
    'list_feature_extractors'
]