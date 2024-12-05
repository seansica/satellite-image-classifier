from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)

from ..core.base import EvaluationResult
from ..core.types import LabelArray
from ..models.base import Model

def evaluate_model(
    model: Model,
    model_name: str,
    X_test: np.ndarray,
    y_test: LabelArray,
    class_names: List[str],
    training_time: float
) -> EvaluationResult:
    """Evaluate a trained model and compute all metrics.
    
    This function computes a comprehensive set of evaluation metrics:
    - Accuracy
    - Precision (weighted average)
    - Recall (weighted average)
    - F1 Score (weighted average)
    - Confusion Matrix
    - ROC Curves and AUC for each class
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: True labels
        class_names: List of class names
        training_time: Time taken to train the model
        
    Returns:
        EvaluationResult containing all computed metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average='weighted'
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curves for each class
    roc_curves = {}
    for i, class_name in enumerate(class_names):
        # Calculate ROC curve for this class
        fpr, tpr, _ = roc_curve(
            y_test == i,  # Binary indicator for this class
            y_pred_proba[:, i]  # Probability predictions for this class
        )
        roc_auc = auc(fpr, tpr)
        
        roc_curves[class_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
    
    return EvaluationResult(
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=cm,
        roc_curves=roc_curves,
        training_time=training_time
    )
