from pathlib import Path
import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from typing import Optional, Tuple

import yaml

from .base import Model

@Model.register('gbm')
class GradientBoostingModel(Model):
    """Gradient Boosting classifier using sklearn backend with PyTorch interface."""

    model_type = "sklearn"  # Override model type

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

    def save(self, path: Path) -> None:
        """Save both sklearn model and torch components."""
        # Save sklearn model
        joblib.dump(self.model, path / "sklearn_model.joblib")
        # Save torch components
        torch.save({"dummy_param": self.dummy_param}, path / "torch_state.pt")

    @classmethod
    def load(
        cls, path: Path, map_location: Optional[torch.device] = None
    ) -> "GradientBoostingModel":
        """Load saved model components."""
        # Load architecture
        with open(path / "architecture.yaml", "r") as f:
            architecture = yaml.safe_load(f)

        # Create instance
        model = cls(
            input_dim=architecture["input_dim"], num_classes=architecture["num_classes"]
        )

        # Load sklearn model
        model.model = joblib.load(path / "sklearn_model.joblib")
        model.is_fitted = True

        # Load torch state
        torch_state = torch.load(path / "torch_state.pt", map_location=map_location)
        model.dummy_param.data = torch_state["dummy_param"]

        return model

    def build(self) -> None:
        """Initialize the sklearn GradientBoostingClassifier."""
        self.classes_ = np.arange(self.num_classes)
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
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # Add dummy samples for missing classes
        unique_classes = np.unique(y_np)
        missing_classes = np.setdiff1d(self.classes_, unique_classes)

        if len(missing_classes) > 0:
            # Create one dummy sample per missing class
            dummy_X = np.zeros((len(missing_classes), X_np.shape[1]))
            X_np = np.vstack([X_np, dummy_X])
            y_np = np.concatenate([y_np, missing_classes])

        self.model.fit(X_np, y_np)
        self.is_fitted = True

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
