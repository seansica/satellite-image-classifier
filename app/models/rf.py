import torch
import torch.nn as nn

from .base import Model

@Model.register('rf')
class RandomForestModel(Model):
    """Neural Random Forest classifier."""

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

    def _create_tree(self) -> nn.Module:
        """Create a single decision tree network."""
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
        """Build the random forest model."""
        self.trees = nn.ModuleList(
            [self._create_tree() for _ in range(self.n_estimators)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass averaging all tree predictions."""
        tree_outputs = []
        for tree in self.trees:
            tree_outputs.append(tree(x))
        return torch.mean(torch.stack(tree_outputs), dim=0)

    def get_criterion(self) -> nn.Module:
        """Get the loss criterion."""
        return nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "RandomForest"
