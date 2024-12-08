import torch
import torch.nn as nn
from typing import Optional

from .base import Model


@Model.register('logistic')
class LogisticRegressionModel(Model):
    """Logistic Regression classifier supporting both binary and multi-class cases."""

    def __init__(self, input_dim: int, num_classes: int, bias: bool = True, **kwargs):
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self._use_bias = bias
        super().__init__(input_dim, num_classes, **kwargs)

    def build(self) -> None:
        """Build the logistic regression model."""
        # For binary classification, we need only one output
        out_features = 1 if self.num_classes == 2 else self.num_classes

        self.linear = nn.Linear(self.input_dim, out_features, bias=self._use_bias)

        # Traditional initialization
        nn.init.zeros_(self.linear.weight)  # or small random values
        if self._use_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Probabilities tensor of shape (batch_size, num_classes)
            For binary classification, shape is (batch_size, 1)
        """
        if len(x.shape) != 2:
            x = x.view(-1, self.input_dim)

        logits = self.linear(x)

        if self.num_classes == 2:
            # Binary case: apply sigmoid
            return torch.sigmoid(logits)
        else:
            # Multi-class case: apply softmax
            return torch.softmax(logits, dim=1)

    def get_criterion(self) -> nn.Module:
        """Get the appropriate loss criterion based on number of classes."""
        if self.num_classes == 2:
            return nn.BCELoss()
        else:
            return nn.CrossEntropyLoss()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        with torch.no_grad():
            return self.forward(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        probs = self.predict_proba(x)
        if self.num_classes == 2:
            return (probs > 0.5).float()
        else:
            return probs.argmax(dim=1)

    @property
    def name(self) -> str:
        return "LogisticRegression"
