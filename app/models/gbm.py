from typing import Optional
from sklearn.ensemble import GradientBoostingClassifier as SKGradientBoostingClassifier
import numpy as np

from .base import Model
from ..core.types import FeatureArray, LabelArray

@Model.register('gbm')
class GradientBoostingModel(Model):
    """Gradient Boosting classifier wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            **kwargs
        )
    
    def _create_model(self) -> SKGradientBoostingClassifier:
        return SKGradientBoostingClassifier(
            n_estimators=self.params['n_estimators'],
            learning_rate=self.params['learning_rate'],
            max_depth=self.params['max_depth'],
            min_samples_split=self.params['min_samples_split'],
            min_samples_leaf=self.params['min_samples_leaf'],
            subsample=self.params['subsample'],
            random_state=self.params['random_state']
        )
    
    def fit(self, X: FeatureArray, y: LabelArray) -> None:
        self._model.fit(X, y)
    
    def predict(self, X: FeatureArray) -> LabelArray:
        return self._model.predict(X)
    
    def predict_proba(self, X: FeatureArray) -> np.ndarray:
        return self._model.predict_proba(X)
    
    @property
    def name(self) -> str:
        return "GradientBoosting"
