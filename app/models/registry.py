from typing import List, Type
from .base import Model

# Import implementations to ensure decorators are executed
from . import svm, logistic, gbm, rf


class LazyModel:
    """Wrapper for lazy model initialization."""

    def __init__(self, model_cls: Type[Model], **kwargs):
        self.model_cls = model_cls
        self.kwargs = kwargs

    def create(self, input_dim: int, num_classes: int) -> Model:
        """Create the model when dimensions are known."""
        return self.model_cls(
            input_dim=input_dim, num_classes=num_classes, **self.kwargs
        )


def get_model(name: str, **kwargs) -> LazyModel:
    """Get a lazy-initialized model by name.

    Args:
        name: Name of the registered model
        **kwargs: Parameters to pass to the model

    Returns:
        LazyModel that can create the model when dimensions are known

    Raises:
        KeyError: If no model is registered with the given name
    """
    try:
        model_cls = Model.get_registry().get(name)
        return LazyModel(model_cls, **kwargs)
    except KeyError:
        available = Model.get_registry().list()
        raise KeyError(
            f"No model registered as '{name}'. "
            f"Available models: {available}"
        )


def list_models() -> List[str]:
    """List all registered models."""
    return Model.get_registry().list()
