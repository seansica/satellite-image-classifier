import torch
import torch.nn as nn

from .base import Model

@Model.register('gbm')
class GradientBoostingModel(Model):
    """Neural Gradient Boosting classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_estimators: int = 100,
        max_depth: int = 5,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(input_dim, num_classes, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.dropout = dropout
        self.hidden_dim = max(input_dim // 2, 10)

    def _create_weak_learner(self) -> nn.Module:
        """Create a single weak learner network."""
        layers = []
        curr_dim = self.input_dim

        for _ in range(self.max_depth):
            layers.extend(
                [
                    nn.Linear(curr_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            curr_dim = self.hidden_dim

        layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        return nn.Sequential(*layers)

    def build(self) -> None:
        """Build the gradient boosting model."""
        self.estimators = nn.ModuleList(
            [self._create_weak_learner() for _ in range(self.n_estimators)]
        )

        # Initialize learner weights
        self.weights = nn.Parameter(torch.ones(self.n_estimators) / self.n_estimators)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted combination of weak learners."""
        estimator_outputs = []
        for estimator in self.estimators:
            estimator_outputs.append(estimator(x))

        weighted_sum = torch.sum(
            torch.stack(estimator_outputs) * self.weights.view(-1, 1, 1), dim=0
        )
        return weighted_sum

    def get_criterion(self) -> nn.Module:
        """Get the loss criterion."""
        return nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "GradientBoosting"
