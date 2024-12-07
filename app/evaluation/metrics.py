from typing import Dict, List, Optional, Union
import numpy as np
import torch
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
    class_names: List[str],
    training_time: float,
    dataset_split: str = "test",
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
    features: Optional[torch.Tensor] = None,  # Changed from predictions to features
    labels: Optional[torch.Tensor] = None,
) -> EvaluationResult:
    """Evaluate a trained model and compute all metrics."""
    device = model.device

    # Handle input types
    if features is not None and labels is not None:
        # Get predictions from features
        features = features.to(dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = model(features)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
            y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
        y_true = labels.cpu().numpy()

    elif X_test is not None and y_test is not None:
        # Make predictions from features
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        if isinstance(y_test, np.ndarray):
            y_test = torch.from_numpy(y_test)

        features = X_test.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(features)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
            y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()

        y_true = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    else:
        raise ValueError("Must provide either (X_test, y_test) or (features, labels)")

    # Ensure predictions and labels are the right type
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate ROC curves for each class
    roc_curves = {}
    for i, class_name in enumerate(class_names):
        # Calculate ROC curve for this class
        fpr, tpr, _ = roc_curve(
            y_true == i,  # Binary indicator for this class
            y_pred_proba[:, i],  # Probability predictions for this class
        )
        roc_auc = auc(fpr, tpr)

        roc_curves[class_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }

    return EvaluationResult(
        model_name=model.name,
        dataset_split=dataset_split,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=cm,
        roc_curves=roc_curves,
        training_time=training_time,
    )
