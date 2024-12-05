from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV

from ..core.base import RegistryMixin
from ..core.types import FeatureArray, LabelArray


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
    
    def tune_hyperparameters(
        self,
        X: FeatureArray,
        y: LabelArray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tune SVM hyperparameters using GridSearchCV.
        """
        grid_search = GridSearchCV(
            estimator=self._model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        grid_search.fit(X, y)
        self.params.update(grid_search.best_params_)
        self._model.set_params(**grid_search.best_params_)
        
        return grid_search