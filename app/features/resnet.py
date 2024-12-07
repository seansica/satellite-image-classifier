import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import functional as TF

from .base import FeatureExtractor


class Identity(nn.Module):
    """Identity module to replace the final classification layer."""

    def forward(self, x):
        return x


@FeatureExtractor.register("resnet50")
class ResNet50FeatureExtractor(FeatureExtractor):
    """Extracts deep features using ResNet50 pre-trained on ImageNet.

    Features are extracted from the final pooling layer before classification,
    providing a 2048-dimensional representation that captures high-level visual patterns.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load pre-trained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove the final classification layer and replace with Identity
        self.model.fc = Identity()

        # Set model to evaluation mode
        self.model.eval()

        # Store output dimension
        self._output_dim = 2048  # ResNet50's feature dimension

        # Get the default preprocessing transforms
        self.weights = ResNet50_Weights.IMAGENET1K_V2

        # Move model to the correct device
        self.model = self.model.to(self.device)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract deep features from a batch of images using ResNet50.

        Args:
            images: Batch of RGB images as torch.Tensor of shape (B, C, H, W)
                   with values in range [0, 1]

        Returns:
            Tensor of shape (B, 2048) containing L2-normalized features
        """
        # Move model to correct device if needed
        self.model = self.model.to(self.device)

        # Ensure images are on the correct device
        images = images.to(self.device)

        # Preprocess images
        preprocessed = self._preprocess_images(images)

        # Extract features
        with torch.no_grad():
            features = self.model(preprocessed)

        # L2 normalize the features
        features = F.normalize(features, dim=1)

        return features

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for ResNet50.

        Args:
            images: Tensor of shape (B, C, H, W) with values in [0, 1]

        Returns:
            Preprocessed images following ImageNet normalization
        """
        # Resize to 224x224 if needed
        if images.shape[-2:] != (224, 224):
            images = TF.resize(images, [224, 224], antialias=True)

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        normalized = (images - mean) / std

        return normalized

    def to(self, device: torch.device) -> "ResNet50FeatureExtractor":
        """Move the feature extractor to the specified device."""
        super().to(device)
        self.model = self.model.to(device)
        return self

    @property
    def output_dim(self) -> int:
        """Get the output feature dimension."""
        return self._output_dim

    @property
    def name(self) -> str:
        return "ResNet50"
