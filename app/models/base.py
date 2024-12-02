from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from ..core.types import FeatureArray, LabelArray
from ..core.base import RegistryMixin

class Model(RegistryMixin, ABC):
    """Base class for all classification models."""
    
    def __init__(self, **kwargs):
        """Initialize model with optional parameters."""
        self.params = kwargs
        self._model = self._create_model()
    
    @abstractmethod
    def _create_model(self):
        """Create and return the underlying model instance."""
        pass
    
    @abstractmethod
    def fit(self, X: FeatureArray, y: LabelArray) -> None:
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: FeatureArray) -> LabelArray:
        """Make predictions for the given features."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: FeatureArray) -> np.ndarray:
        """Get prediction probabilities for the given features."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used by this model."""
        return self.params.copy()