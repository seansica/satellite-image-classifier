import torch
import torch.nn as nn

from .base import Model

@Model.register('rf')
class RandomForestModel(Model):
    """Memory-efficient Neural Random Forest classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_estimators: int = 10,
        max_depth: int = 3,
        hidden_dim: int = None,
        dropout: float = 0.1,
        **kwargs
    ):
        # Store model-specific parameters
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._dropout = dropout

        # More conservative hidden dimension calculation
        if hidden_dim is not None:
            self._hidden_dim = hidden_dim
        else:
            self._hidden_dim = max(input_dim // 4, 10)

        # Call parent's __init__ after setting model-specific parameters
        super().__init__(input_dim, num_classes, **kwargs)

    def _create_tree(self) -> nn.Module:
        """Create a single decision tree network with reduced complexity."""
        layers = []
        curr_dim = self.input_dim

        for depth in range(self._max_depth):
            out_dim = (
                self._hidden_dim if depth < self._max_depth - 1 else self.num_classes
            )

            # Ensure linear layer parameters require gradients
            linear = nn.Linear(curr_dim, out_dim)
            linear.weight.requires_grad_(True)
            linear.bias.requires_grad_(True)

            layers.extend(
                [
                    linear,
                    nn.ReLU() if depth < self._max_depth - 1 else nn.Identity(),
                    (
                        nn.Dropout(self._dropout)
                        if depth < self._max_depth - 1
                        else nn.Identity()
                    ),
                ]
            )
            curr_dim = out_dim

        return nn.Sequential(*layers)

    def build(self) -> None:
        """Build the random forest model."""
        self.trees = nn.ModuleList(
            [self._create_tree() for _ in range(self._n_estimators)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient forward pass."""
        # Ensure input requires gradients
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # Process trees in chunks to save memory
        chunk_size = 5  # Process 5 trees at a time
        outputs = []

        for i in range(0, len(self.trees), chunk_size):
            chunk_trees = list(self.trees[i : i + chunk_size])
            chunk_outputs = []

            for tree in chunk_trees:
                chunk_outputs.append(tree(x))

            # Average the chunk
            chunk_avg = torch.mean(torch.stack(chunk_outputs), dim=0)
            outputs.append(chunk_avg)

        # Final average across all chunks
        return torch.mean(torch.stack(outputs), dim=0)

    def get_criterion(self) -> nn.Module:
        """Get the loss criterion."""
        return nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "RandomForest"
