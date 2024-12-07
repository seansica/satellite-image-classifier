import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FeatureExtractor


def compute_lbp(x: torch.Tensor) -> torch.Tensor:
    """Compute LBP codes using PyTorch operations.

    Args:
        x: Input tensor of shape (B, 1, H, W)

    Returns:
        LBP codes of shape (B, 1, H-2, W-2)
    """
    # Pad image for 3x3 mask size
    x = F.pad(x, pad=[1, 1, 1, 1], mode="reflect")

    # Get shape
    B, C, M, N = x.shape

    # Select elements within 3x3 mask
    y00 = x[:, :, 0 : M - 2, 0 : N - 2]
    y01 = x[:, :, 0 : M - 2, 1 : N - 1]
    y02 = x[:, :, 0 : M - 2, 2:N]
    y10 = x[:, :, 1 : M - 1, 0 : N - 2]
    y11 = x[:, :, 1 : M - 1, 1 : N - 1]  # Center pixel
    y12 = x[:, :, 1 : M - 1, 2:N]
    y20 = x[:, :, 2:M, 0 : N - 2]
    y21 = x[:, :, 2:M, 1 : N - 1]
    y22 = x[:, :, 2:M, 2:N]

    # Compute LBP code
    code = torch.zeros_like(y11)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=x.device)

    # Compare with neighbors and compute binary code
    neighbors = [y01, y02, y12, y22, y21, y20, y10, y00]
    for bit, neighbor in enumerate(neighbors):
        code = code + (neighbor >= y11).float() * powers[bit]

    return code


@FeatureExtractor.register("lbp")
class LBPFeatureExtractor(FeatureExtractor):
    """Extracts texture information using Local Binary Patterns."""

    def __init__(
        self,
        grid_x: int = 4,  # Number of grid cells in x direction
        grid_y: int = 4,  # Number of grid cells in y direction
        n_bins: int = 256,  # Number of histogram bins (2^8 for standard LBP)
        **kwargs
    ):
        super().__init__(grid_x=grid_x, grid_y=grid_y, n_bins=n_bins, **kwargs)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.n_bins = n_bins
        self._output_dim = n_bins * grid_x * grid_y

    def compute_grid_histograms(self, lbp_codes: torch.Tensor) -> torch.Tensor:
        """Compute histograms for grid cells.

        Args:
            lbp_codes: Tensor of shape (B, 1, H, W)

        Returns:
            Grid cell histograms of shape (B, grid_y * grid_x * n_bins)
        """
        B, C, H, W = lbp_codes.shape
        device = lbp_codes.device

        # Calculate grid cell sizes
        cell_h = H // self.grid_y
        cell_w = W // self.grid_x

        # Initialize output tensor
        features = torch.zeros(
            B, self.grid_y * self.grid_x * self.n_bins, device=device
        )

        # Process each grid cell
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                # Extract cell
                cell = lbp_codes[
                    :, :, i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                ]

                # Compute histogram for each batch item
                for b in range(B):
                    # Use bincount for histogram
                    hist = torch.bincount(
                        cell[b].flatten().long(), minlength=self.n_bins
                    ).float()

                    # Normalize
                    hist = hist / hist.sum().clamp(min=1e-8)

                    # Store in output tensor
                    start_idx = (i * self.grid_x + j) * self.n_bins
                    features[b, start_idx : start_idx + self.n_bins] = hist

        return features

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract LBP features from a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W)

        Returns:
            LBP features of shape (B, output_dim)
        """
        # Convert to grayscale if needed
        if images.size(1) > 1:
            gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
            gray = gray.unsqueeze(1)
        else:
            gray = images

        # Compute LBP codes
        lbp_codes = compute_lbp(gray)

        # Compute grid cell histograms
        features = self.compute_grid_histograms(lbp_codes)

        return features

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def name(self) -> str:
        return "LBP"
