from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from torch import Tensor

from ..core.base import RegistryMixin


class FeatureExtractor(RegistryMixin, ABC):
    """Base class for all feature extractors."""

    def __init__(self, **kwargs):
        """Initialize feature extractor with optional parameters."""
        self.params = kwargs
        self.device = torch.device("cpu")  # Default device

    @abstractmethod
    def extract(self, image: Tensor) -> Tensor:
        """Extract features from a batch of images.

        Args:
            image: Batch of images as torch.Tensor of shape (B, C, H, W)
                  where B is batch size, C is channels (3 for RGB)

        Returns:
            Tensor of shape (B, F) where F is the feature dimension
        """
        pass

    def to(self, device: torch.device) -> "FeatureExtractor":
        """Move the feature extractor to the specified device.

        Args:
            device: Device to move to (e.g., 'cuda', 'mps', or 'cpu')

        Returns:
            self for method chaining
        """
        self.device = device
        return self

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the feature extractor."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used by this feature extractor."""
        return self.params.copy()
