import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import FeatureExtractor


class HOGLayerTorch(nn.Module):
    """PyTorch implementation of HOG feature extraction."""

    def __init__(
        self,
        nbins: int = 9,
        pool: int = 8,
        max_angle: float = math.pi,
    ):
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.max_angle = max_angle

        # Sobel filter initialization
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:, None, :, :])

        # Pooling layer
        self.pooler = nn.AvgPool2d(
            pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing HOG features."""
        if x.size(1) > 1:
            x = x.mean(dim=1)[:, None, :, :]

        # 1. Compute gradients
        gxy = F.conv2d(x, self.weight, None, stride=1, padding=1)

        # 2. Compute magnitude and phase
        mag = gxy.norm(dim=1)
        norm = mag[:, None, :, :]
        phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

        # 3. Bin magnitudes into orientation bins
        phase_int = phase / self.max_angle * self.nbins
        phase_int = phase_int[:, None, :, :]

        # Efficient binning using scatter operations
        n, _, h, w = gxy.shape
        out = torch.zeros(
            (n, self.nbins, h, w),
            dtype=torch.float,
            device=x.device,  # Use same device as input
        )

        # Linear interpolation between bins
        out.scatter_(1, phase_int.floor().long() % self.nbins, norm)
        out.scatter_add_(1, phase_int.ceil().long() % self.nbins, 1 - norm)

        # 4. Block normalization via average pooling
        return self.pooler(out)


@FeatureExtractor.register('hog')
class HOGFeatureExtractor(FeatureExtractor):
    """HOG feature extractor using optimized PyTorch implementation."""

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (3, 3),
        **kwargs
    ):
        super().__init__(
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            **kwargs
        )

        self.hog = HOGLayerTorch(
            nbins=orientations,
            pool=pixels_per_cell[0],  # Assuming square cells
            max_angle=math.pi,
        )

        # Initialize device
        self.device = torch.device("cpu")

        # Calculate output dimension
        self._output_dim = None  # Will be set on first feature extraction

    def _calculate_output_dim(self, height: int, width: int) -> int:
        """Calculate the output feature dimension.

        Args:
            height: Input image height
            width: Input image width

        Returns:
            Number of features in the HOG descriptor
        """
        # Calculate number of cells
        n_cells_y = height // self.params["pixels_per_cell"][0]
        n_cells_x = width // self.params["pixels_per_cell"][1]

        # Calculate number of orientations per cell
        n_orient = self.params["orientations"]

        # Calculate total feature dimension
        # Each spatial position after pooling will have n_orient features
        return n_cells_y * n_cells_x * n_orient

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract HOG features from a batch of images."""
        # Move HOG module to correct device if needed
        self.hog = self.hog.to(self.device)

        # Ensure images are on correct device
        images = images.to(self.device)

        # Extract HOG features
        features = self.hog(images)

        # Calculate output dimension if not already set
        if self._output_dim is None:
            # Update output_dim based on actual feature size
            self._output_dim = features.shape[1] * features.shape[2] * features.shape[3]

        # Flatten spatial dimensions
        return features.flatten(1)

    def to(self, device: torch.device) -> "HOGFeatureExtractor":
        """Move the feature extractor to the specified device."""
        self.device = device
        self.hog = self.hog.to(device)
        return self

    @property
    def output_dim(self) -> int:
        """Get the output feature dimension.

        Note: This will return None until the first call to extract()
        as we need to see the actual feature size.
        """
        if self._output_dim is None:
            raise RuntimeError(
                "Output dimension not yet calculated. "
                "Call extract() first with a sample input."
            )
        return self._output_dim

    @property
    def name(self) -> str:
        return "HOG"
