from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generic, List, Optional
import numpy as np

from .types import T, ImageArray

@dataclass
class DatasetMetadata:
    """Metadata about a dataset including paths and labels."""
    name: str
    class_names: List[str]
    n_classes: int
    n_samples: int
    data_path: Path

class Dataset:
    """Container for dataset samples and metadata."""
    def __init__(
        self,
        images: List[ImageArray],
        labels: List[str],
        metadata: DatasetMetadata
    ):
        self.images = images
        self.labels = labels
        self.metadata = metadata
    
    @property
    def n_samples(self) -> int:
        return len(self.images)
    
    @property
    def class_names(self) -> List[str]:
        return self.metadata.class_names

class Registry(Generic[T]):
    """Generic registry for managing named components."""
    def __init__(self):
        self._registry: Dict[str, T] = {}
    
    def register(self, name: str, item: T) -> None:
        """Register a new item with the given name."""
        if name in self._registry:
            raise ValueError(f"Item with name '{name}' already registered")
        self._registry[name] = item
    
    def get(self, name: str) -> T:
        """Retrieve a registered item by name."""
        if name not in self._registry:
            raise KeyError(f"No item registered with name '{name}'")
        return self._registry[name]
    
    def list(self) -> List[str]:
        """List all registered item names."""
        return list(self._registry.keys())

class RegistryMixin:
    """Mixin class to add registration capability to components."""
    _registry: Optional[Registry] = None
    
    @classmethod
    def register(cls, name: str):
        """Decorator for registering implementations."""
        def decorator(impl):
            if cls._registry is None:
                cls._registry = Registry()
            cls._registry.register(name, impl)
            return impl
        return decorator
    
    @classmethod
    def get_registry(cls) -> Registry:
        """Get the component registry."""
        if cls._registry is None:
            cls._registry = Registry()
        return cls._registry

@dataclass
class EvaluationResult:
    """Container for model evaluation results."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    roc_curves: Dict[str, Dict[str, np.ndarray]]  # {class_name: {'fpr': array, 'tpr': array}}
    training_time: float