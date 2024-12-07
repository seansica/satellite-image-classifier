import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FeatureExtractor


class LBPLayer(nn.Module):
    """PyTorch implementation of Local Binary Pattern feature extraction."""

    def __init__(self, n_points: int = 24, radius: int = 3):
        super().__init__()
        self.n_points = n_points
        self.radius = radius

        # Precompute sampling coordinates
        angles = torch.arange(0, 2 * torch.pi, 2 * torch.pi / n_points)
        self.register_buffer("sample_x", radius * torch.cos(angles))
        self.register_buffer("sample_y", radius * torch.sin(angles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LBP features.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            LBP codes of shape (B, 1, H, W)
        """
        # Get dimensions
        B, _, H, W = x.shape
        device = x.device

        # Create sampling grid
        xx, yy = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        # Initialize LBP codes
        lbp = torch.zeros((B, 1, H, W), device=device)

        # For each sampling point
        for i in range(self.n_points):
            # Calculate sampling coordinates
            sample_x = xx + self.sample_x[i]
            sample_y = yy + self.sample_y[i]

            # Grid sample requires coordinates in [-1, 1]
            grid_x = (2.0 * sample_x / (W - 1)) - 1.0
            grid_y = (2.0 * sample_y / (H - 1)) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.expand(B, -1, -1, -1)

            # Sample values
            sampled = F.grid_sample(
                x, grid, mode="bilinear", padding_mode="border", align_corners=True
            )

            # Compare with center pixel
            bit = (sampled >= x).float()

            # Add to LBP code
            lbp = lbp + (bit * (2**i))

        return lbp


@FeatureExtractor.register("lbp")
class LBPFeatureExtractor(FeatureExtractor):
    """Extracts texture information using Local Binary Patterns."""

    def __init__(
        self,
        n_points: int = 24,  # Number of points in the circular pattern
        radius: int = 3,  # Radius of the circular pattern
        grid_x: int = 4,  # Number of grid cells in x direction
        grid_y: int = 4,  # Number of grid cells in y direction
        **kwargs
    ):
        super().__init__(
            n_points=n_points, radius=radius, grid_x=grid_x, grid_y=grid_y, **kwargs
        )

        self.lbp_layer = LBPLayer(n_points=n_points, radius=radius)
        self.grid_x = grid_x
        self.grid_y = grid_y

        # Number of histogram bins for uniform LBP
        self.n_bins = n_points + 2

        # Calculate output dimension
        self._output_dim = self.n_bins * grid_x * grid_y

    def compute_grid_histograms(self, lbp_codes: torch.Tensor) -> torch.Tensor:
        """Compute histograms for grid cells.

        Args:
            lbp_codes: Tensor of shape (B, 1, H, W)

        Returns:
            Grid cell histograms of shape (B, grid_y * grid_x * n_bins)
        """
        B, _, H, W = lbp_codes.shape

        # Compute cell dimensions
        cell_h = H // self.grid_y
        cell_w = W // self.grid_x

        # Initialize histograms
        histograms = []

        # For each grid cell
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                # Extract cell
                cell = lbp_codes[
                    :, :, i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                ]

                # Compute histogram
                hist = torch.histc(
                    cell.float(), bins=self.n_bins, min=0, max=self.n_bins - 1
                )
                hist = hist / hist.sum().clamp(min=1e-8)  # Normalize
                histograms.append(hist)

        # Concatenate all histograms
        return torch.cat(histograms, dim=0)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract LBP features from a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W)

        Returns:
            LBP features of shape (B, grid_y * grid_x * n_bins)
        """
        # Convert to grayscale if needed
        if images.size(1) > 1:
            gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
            gray = gray.unsqueeze(1)
        else:
            gray = images

        # Move LBP layer to correct device if needed
        self.lbp_layer = self.lbp_layer.to(self.device)

        # Compute LBP codes
        lbp_codes = self.lbp_layer(gray)

        # Compute grid cell histograms
        features = self.compute_grid_histograms(lbp_codes)

        return features

    @property
    def output_dim(self) -> int:
        """Get the output feature dimension."""
        return self._output_dim

    @property
    def name(self) -> str:
        return "LBP"
