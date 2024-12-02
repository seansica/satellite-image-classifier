from typing import List, Type, Dict
from .base import Model

# Import implementations to ensure decorators are executed
from . import svm, logistic  # This ensures all implementations are registered

def get_model(name: str, **kwargs) -> Model:
    """Get a model instance by name.
    
    Args:
        name: Name of the registered model
        **kwargs: Parameters to pass to the model
        
    Returns:
        Instantiated model
        
    Raises:
        KeyError: If no model is registered with the given name
    """
    try:
        model_cls = Model.get_registry().get(name)
        return model_cls(**kwargs)
    except KeyError:
        available = Model.get_registry().list()
        raise KeyError(
            f"No model registered as '{name}'. "
            f"Available models: {available}"
        )

def list_models() -> List[str]:
    """List all registered models."""
    return Model.get_registry().list()