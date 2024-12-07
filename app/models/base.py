from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Type, Optional, Tuple

from ..core.base import Registry


class Model(nn.Module, ABC):
    """Base class for all models, combining nn.Module functionality with registration."""

    # Class-level registry
    _registry: Optional[Registry] = None

    def __init__(self, input_dim: int, num_classes: int, **kwargs):
        """Initialize the model.

        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            **kwargs: Additional model parameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.params = kwargs

        # Track device
        self._device = torch.device("cpu")

        # Initialize model architecture
        self.build()

    @classmethod
    def register(cls, name: str):
        """Decorator for registering model implementations."""
        if cls._registry is None:
            cls._registry = Registry()

        def decorator(impl: Type["Model"]) -> Type["Model"]:
            cls._registry.register(name, impl)
            return impl

        return decorator

    @classmethod
    def get_registry(cls) -> Registry:
        """Get the model registry."""
        if cls._registry is None:
            cls._registry = Registry()
        return cls._registry

    @abstractmethod
    def build(self) -> None:
        """Build the model architecture."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        """Get the loss criterion for this model."""
        pass

    def train_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        """Perform a single training step."""
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.get_criterion()(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Perform a single validation step."""
        inputs, targets = batch
        with torch.no_grad():
            outputs = self(inputs)
            loss = self.get_criterion()(outputs, targets)
        return loss.item()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels for input data."""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            predictions = outputs.argmax(dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for input data."""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities

    def to(self, device: torch.device) -> "Model":
        """Move the model to the specified device."""
        self._device = device
        return super().to(device)

    @property
    def device(self) -> torch.device:
        """Get the current device of the model."""
        return self._device

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model."""
        pass
