from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from ..core.types import FeatureArray, LabelArray

def create_train_test_split(
    features: FeatureArray,
    labels: LabelArray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[FeatureArray, FeatureArray, LabelArray, LabelArray]:
    """Split features and labels into training and test sets.
    
    Args:
        features: Feature array
        labels: Label array
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Maintain class distribution in splits
    )