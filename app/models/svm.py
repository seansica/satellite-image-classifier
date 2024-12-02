from typing import Optional
from sklearn.svm import SVC
import numpy as np

from .base import Model
from ..core.types import FeatureArray, LabelArray

@Model.register('svm')
class SVMModel(Model):
    """Support Vector Machine classifier wrapper."""
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs
        )
    
    def _create_model(self) -> SVC:
        return SVC(
            kernel=self.params['kernel'],
            C=self.params['C'],
            gamma=self.params['gamma'],
            class_weight=self.params['class_weight'],
            random_state=self.params['random_state'],
            probability=True
        )
    
    def fit(self, X: FeatureArray, y: LabelArray) -> None:
        self._model.fit(X, y)
    
    def predict(self, X: FeatureArray) -> LabelArray:
        return self._model.predict(X)
    
    def predict_proba(self, X: FeatureArray) -> np.ndarray:
        return self._model.predict_proba(X)
    
    @property
    def name(self) -> str:
        return "SVM"