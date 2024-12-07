import torch
import torch.nn as nn

from .base import Model


class HingeLoss(nn.Module):
    """Multi-class hinge loss with margin."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the hinge loss.

        Args:
            output: Predictions of shape (N, C)
            target: Target labels of shape (N,)

        Returns:
            torch.Tensor: Average hinge loss
        """
        # Ensure output is properly shaped
        if len(output.shape) != 2:
            raise ValueError(f"Expected output shape (N, C), got {output.shape}")

        # Get correct class scores
        target_scores = output[torch.arange(output.size(0)), target]

        # Compute margins for all classes
        margins = output - target_scores.unsqueeze(1) + self.margin

        # Zero out margin for correct class
        margins[torch.arange(margins.size(0)), target] = 0

        # Take hinge and mean
        return torch.mean(torch.clamp(margins, min=0))


@Model.register('svm')
class SVMModel(Model):
    def __init__(self, input_dim: int, num_classes: int, margin: float = 1.0, **kwargs):
        self._margin = margin  # Store before super().__init__
        super().__init__(input_dim, num_classes, **kwargs)

    def build(self) -> None:
        """Build the SVM model architecture."""
        self.linear = nn.Linear(self.input_dim, self.num_classes)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SVM.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        # Ensure input is properly shaped
        if len(x.shape) != 2:
            x = x.flatten(1)

        return self.linear(x)

    def get_criterion(self) -> nn.Module:
        """Get the hinge loss criterion."""
        return HingeLoss(margin=self._margin)

    @property
    def name(self) -> str:
        return "SVM"
