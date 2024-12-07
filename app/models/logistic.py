import torch
import torch.nn as nn

from .base import Model


@Model.register('logistic')
class LogisticRegressionModel(Model):
    """Logistic Regression classifier."""

    def __init__(
        self, input_dim: int, num_classes: int, learning_rate: float = 0.001, **kwargs
    ):
        super().__init__(input_dim, num_classes, **kwargs)
        self.learning_rate = learning_rate

    def build(self) -> None:
        """Build the logistic regression model."""
        self.linear = nn.Linear(self.input_dim, self.num_classes)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.linear(x)

    def get_criterion(self) -> nn.Module:
        """Get the loss criterion."""
        return nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "LogisticRegression"
