from typing import List, Type, Optional, Dict
from .base import Model

# Import implementations to ensure decorators are executed
from . import svm, logistic, gbm, rf


class LazyModel:
    """Wrapper for lazy model initialization."""

    def __init__(self, name: str, model_cls: Type[Model], **kwargs):
        self.name = name
        self.model_cls = model_cls
        self.kwargs = kwargs

    def create(
        self, input_dim: int, num_classes: int, model_params: Optional[Dict] = None
    ) -> Model:
        """Create the model when dimensions are known.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            model_params: Dictionary of model-specific parameters, keyed by model name

        Returns:
            Instantiated model
        """
        # Start with base kwargs from initialization
        params = self.kwargs.copy()

        # Update with model-specific params if provided
        if model_params and self.name in model_params:
            params.update(model_params[self.name])

        return self.model_cls(input_dim=input_dim, num_classes=num_classes, **params)


def get_model(name: str, **kwargs) -> LazyModel:
    """Get a lazy-initialized model by name.

    Args:
        name: Name of the registered model
        **kwargs: Default parameters to pass to the model

    Returns:
        LazyModel that can create the model when dimensions are known

    Raises:
        KeyError: If no model is registered with the given name
    """
    try:
        model_cls = Model.get_registry().get(name)
        return LazyModel(name, model_cls, **kwargs)
    except KeyError:
        available = Model.get_registry().list()
        raise KeyError(
            f"No model registered as '{name}'. "
            f"Available models: {available}"
        )


def list_models() -> List[str]:
    """List all registered models."""
    return Model.get_registry().list()
