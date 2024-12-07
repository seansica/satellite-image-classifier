import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FeatureExtractor


@FeatureExtractor.register("hsv")
class HSVHistogramExtractor(FeatureExtractor):
    """Extracts color information using HSV color space histograms.

    HSV color space is particularly useful for satellite imagery because it separates
    color information (Hue) from illumination (Value), making it more robust to the
    extreme lighting conditions encountered in space photography.
    """

    def __init__(
        self,
        h_bins: int = 30,  # More bins for hue to capture subtle color differences
        s_bins: int = 32,  # Balanced bins for saturation
        v_bins: int = 32,  # Balanced bins for value
        **kwargs
    ):
        super().__init__(h_bins=h_bins, s_bins=s_bins, v_bins=v_bins, **kwargs)
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins

        # Calculate output dimension
        self._output_dim = h_bins + s_bins + v_bins

    def rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV color space.

        Args:
            rgb: Tensor of shape (B, 3, H, W) with values in range [0, 1]

        Returns:
            Tensor of shape (B, 3, H, W) with HSV values
        """
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

        max_rgb, _ = torch.max(rgb, dim=1)
        min_rgb, _ = torch.min(rgb, dim=1)
        diff = max_rgb - min_rgb

        # Hue calculation
        hue = torch.zeros_like(max_rgb)

        # If R is maximum
        mask_r = max_rgb == r
        hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)

        # If G is maximum
        mask_g = max_rgb == g
        hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)

        # If B is maximum
        mask_b = max_rgb == b
        hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)

        # Normalize hue to [0, 1]
        hue = hue / 360.0

        # Saturation calculation
        saturation = torch.zeros_like(max_rgb)
        mask = max_rgb != 0
        saturation[mask] = diff[mask] / max_rgb[mask]

        # Value calculation
        value = max_rgb

        return torch.stack([hue, saturation, value], dim=1)

    def compute_histogram(self, channel: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Compute histogram for a single channel using differentiable operations.

        Args:
            channel: Tensor of shape (B, H, W) with values in [0, 1]
            num_bins: Number of histogram bins

        Returns:
            Tensor of shape (B, num_bins) containing normalized histogram
        """
        # Scale values to [0, num_bins-1]
        scaled = channel * (num_bins - 1)

        # Compute bin assignments using soft assignments
        bin_assignments = torch.floor(scaled).long()
        weights = 1 - (scaled - bin_assignments)

        # Create one-hot encodings
        shape = bin_assignments.shape
        one_hot = torch.zeros(
            (shape[0], num_bins, shape[1], shape[2]), device=channel.device
        )

        # Distribute weights to adjacent bins
        one_hot.scatter_(1, bin_assignments.unsqueeze(1), weights.unsqueeze(1))
        next_bin = (bin_assignments + 1).clamp(max=num_bins - 1).unsqueeze(1)
        one_hot.scatter_add_(1, next_bin, (1 - weights).unsqueeze(1))

        # Sum over spatial dimensions and normalize
        hist = one_hot.sum(dim=(2, 3))
        hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return hist

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract HSV histogram features from a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W) with RGB values in [0, 1]

        Returns:
            Tensor of shape (B, h_bins + s_bins + v_bins) containing concatenated histograms
        """
        # Convert to HSV
        hsv = self.rgb_to_hsv(images)

        # Compute histograms for each channel
        h_hist = self.compute_histogram(hsv[:, 0], self.h_bins)
        s_hist = self.compute_histogram(hsv[:, 1], self.s_bins)
        v_hist = self.compute_histogram(hsv[:, 2], self.v_bins)

        # Concatenate histograms
        return torch.cat([h_hist, s_hist, v_hist], dim=1)

    @property
    def output_dim(self) -> int:
        """Get the output feature dimension."""
        return self._output_dim

    @property
    def name(self) -> str:
        return "HSV_Histogram"
