import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from typing import Tuple

from .base import Model

@Model.register('gbm')
class GradientBoostingModel(Model):
    """Gradient Boosting classifier using sklearn backend with PyTorch interface."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,  # Called 'learning_rate' in sklearn
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,  # For stochastic gradient boosting
        max_features: str = "sqrt",  # Traditional default for classification
        **kwargs
    ):
        # Store GBM-specific parameters before parent class initialization
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features

        # Initialize parent class
        super().__init__(input_dim, num_classes, **kwargs)

    def build(self) -> None:
        """Initialize the sklearn GradientBoostingClassifier."""
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features,
            random_state=42,
        )

        # Add dummy parameter for PyTorch compatibility
        self.dummy_param = nn.Parameter(torch.zeros(1))

        # Initialize criterion for validation
        self.criterion_fn = nn.CrossEntropyLoss()

        # Track if model has been fitted
        self.is_fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class probabilities.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Class probabilities tensor of shape (batch_size, num_classes)
        """
        if not self.is_fitted:
            # Return uniform probabilities if not fitted
            batch_size = x.shape[0]
            return (
                torch.ones(batch_size, self.num_classes, device=x.device)
                / self.num_classes
            )

        # Convert to numpy for sklearn
        x_np = x.detach().cpu().numpy()

        # Get predictions
        probas = self.model.predict_proba(x_np)

        # Convert back to torch tensor
        return torch.from_numpy(probas).float().to(x.device)

    def train_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        """Train using sklearn's fit method.

        Args:
            batch: Tuple of (inputs, targets)
            optimizer: Unused, kept for pipeline compatibility

        Returns:
            0.0 as loss value (actual training handled by sklearn)
        """
        X, y = batch

        # Convert to numpy
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # Fit the model
        self.model.fit(X_np, y_np)
        self.is_fitted = True

        # Handle optimizer step with dummy parameter to maintain pipeline compatibility
        optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True, device=self.dummy_param.device)
        loss.backward()
        optimizer.step()

        return 0.0

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Compute validation loss using CrossEntropy.

        Args:
            batch: Tuple of (inputs, targets)

        Returns:
            Validation loss value
        """
        X, y = batch
        outputs = self(X)
        return self.criterion_fn(outputs, y).item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predicted class labels tensor of shape (batch_size,)
        """
        if not self.is_fitted:
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Convert to numpy for sklearn
        x_np = x.detach().cpu().numpy()

        # Get predictions
        preds = self.model.predict(x_np)

        # Convert back to torch tensor
        return torch.from_numpy(preds).to(x.device)

    def get_criterion(self) -> nn.Module:
        """Get the loss criterion (CrossEntropyLoss for classification)."""
        return self.criterion_fn

    @property
    def name(self) -> str:
        return "GradientBoosting"
