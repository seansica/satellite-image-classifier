from typing import Optional
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
import numpy as np

from .base import Model
from ..core.types import FeatureArray, LabelArray

@Model.register('rf')
class RandomForestModel(Model):
    """Random Forest classifier wrapper."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        )
    
    def _create_model(self) -> SKRandomForestClassifier:
        return SKRandomForestClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            min_samples_split=self.params['min_samples_split'],
            min_samples_leaf=self.params['min_samples_leaf'],
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
        return "RandomForest"