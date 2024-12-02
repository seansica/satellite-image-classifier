"""Feature extraction components."""

from .base import FeatureExtractor
from .hog import HOGFeatureExtractor
from .hsv import HSVHistogramExtractor
from .lbp import LBPFeatureExtractor
from .resnet import ResNet50FeatureExtractor
from .combined import CombinedFeatureExtractor

from .registry import get_feature_extractor, list_feature_extractors

__all__ = [
    "FeatureExtractor",
    "HOGFeatureExtractor",
    "HSVHistogramExtractor",
    "LBPFeatureExtractor",
    "ResNet50FeatureExtractor",
    "CombinedFeatureExtractor",
    "get_feature_extractor",
    "list_feature_extractors",
]
