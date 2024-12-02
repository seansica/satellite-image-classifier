from typing import Optional
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
import numpy as np

from .base import Model
from ..core.types import FeatureArray, LabelArray

@Model.register('logistic')
class LogisticRegressionModel(Model):
    """Logistic Regression classifier wrapper."""
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        )
    
    def _create_model(self) -> SKLogisticRegression:
        return SKLogisticRegression(
            C=self.params['C'],
            max_iter=self.params['max_iter'],
            class_weight=self.params['class_weight'],
            random_state=self.params['random_state'],
            n_jobs=-1  # Use all available cores
        )
    
    def fit(self, X: FeatureArray, y: LabelArray) -> None:
        self._model.fit(X, y)
    
    def predict(self, X: FeatureArray) -> LabelArray:
        return self._model.predict(X)
    
    def predict_proba(self, X: FeatureArray) -> np.ndarray:
        return self._model.predict_proba(X)
    
    @property
    def name(self) -> str:
        return "LogisticRegression"