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
        n_estimators: int = 10,  # Reduced default for memory efficiency
        max_depth: int = 3,  # Reduced default for memory efficiency
        hidden_dim: int = None,  # Allow override via CLI
        dropout: float = 0.1,
        **kwargs
    ):
        # Store model-specific parameters before calling parent's __init__
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._dropout = dropout

        # Set hidden dimension with optional override
        if hidden_dim is not None:
            self._hidden_dim = hidden_dim
        else:
            self._hidden_dim = max(input_dim // 4, 10)  # More memory efficient default

        # Call parent's __init__ after setting model-specific parameters
        super().__init__(input_dim, num_classes, **kwargs)

    def _create_weak_learner(self) -> nn.Module:
        """Create a single weak learner network."""
        layers = []
        curr_dim = self.input_dim

        for _ in range(self._max_depth):
            # Ensure linear layer parameters require gradients
            linear = nn.Linear(curr_dim, self._hidden_dim)
            linear.weight.requires_grad_(True)
            linear.bias.requires_grad_(True)

            layers.extend(
                [
                    linear,
                    nn.ReLU(),
                    nn.Dropout(self._dropout),
                ]
            )
            curr_dim = self._hidden_dim

        # Output layer
        output_layer = nn.Linear(self._hidden_dim, self.num_classes)
        output_layer.weight.requires_grad_(True)
        output_layer.bias.requires_grad_(True)
        layers.append(output_layer)

        return nn.Sequential(*layers)

    def build(self) -> None:
        """Build the gradient boosting model."""
        self.estimators = nn.ModuleList(
            [self._create_weak_learner() for _ in range(self._n_estimators)]
        )

        # Initialize learner weights with gradients enabled
        self.weights = nn.Parameter(torch.ones(self._n_estimators) / self._n_estimators)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted combination of weak learners."""
        # Ensure input requires gradients
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # Process estimators in chunks to save memory
        chunk_size = 5
        all_outputs = []

        for i in range(0, self._n_estimators, chunk_size):
            chunk_estimators = list(self.estimators[i : i + chunk_size])
            chunk_weights = self.weights[i : i + chunk_size]

            chunk_outputs = []
            for estimator in chunk_estimators:
                chunk_outputs.append(estimator(x))

            # Weight and sum the chunk
            chunk_stack = torch.stack(chunk_outputs)
            weighted_chunk = chunk_stack * chunk_weights.view(-1, 1, 1)
            all_outputs.append(torch.sum(weighted_chunk, dim=0))

        # Sum all chunks
        return torch.sum(torch.stack(all_outputs), dim=0)

    def get_criterion(self) -> nn.Module:
        """Get the loss criterion."""
        return nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "GradientBoosting"
