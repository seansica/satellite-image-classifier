from typing import Dict, List, Optional, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from ..core.base import EvaluationResult
from ..core.types import LabelArray
from ..models.base import Model


def batch_evaluate(
    predictions: torch.Tensor, targets: torch.Tensor, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute predictions and probabilities in batches to avoid memory spikes."""
    with torch.no_grad():
        probs = torch.softmax(predictions, dim=1)
        preds = predictions.argmax(dim=1)

    return preds.cpu().numpy(), probs.cpu().numpy()


def evaluate_model(
    model: Model,
    class_names: List[str],
    training_time: float,
    dataset_split: str = "test",
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
    features: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    batch_size: int = 1024,  # Added batch size parameter
) -> EvaluationResult:
    """Evaluate a trained model and compute all metrics."""
    device = model.device
    n_classes = len(class_names)

    # Initialize arrays for predictions and ground truth
    if features is not None and labels is not None:
        # Convert labels to numpy once
        y_true = labels.cpu().numpy()

        # Process in batches to avoid memory issues
        n_samples = features.shape[0]
        y_pred_list = []
        y_pred_proba_list = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_features = features[i : i + batch_size].to(
                    device=device, dtype=torch.float32
                )
                outputs = model(batch_features)
                batch_preds, batch_probs = batch_evaluate(
                    outputs, labels[i : i + batch_size], n_classes
                )
                y_pred_list.append(batch_preds)
                y_pred_proba_list.append(batch_probs)

        # Concatenate results
        y_pred = np.concatenate(y_pred_list)
        y_pred_proba = np.concatenate(y_pred_proba_list)

    elif X_test is not None and y_test is not None:
        # Convert input features to torch tensor if needed
        if isinstance(X_test, np.ndarray):
            features = torch.from_numpy(X_test).float()
        else:
            features = X_test

        # Convert labels to numpy once
        y_true = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

        # Process in batches
        n_samples = features.shape[0]
        y_pred_list = []
        y_pred_proba_list = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_features = features[i : i + batch_size].to(
                    device=device, dtype=torch.float32
                )
                outputs = model(batch_features)
                batch_preds, batch_probs = batch_evaluate(
                    outputs, torch.from_numpy(y_true[i : i + batch_size]), n_classes
                )
                y_pred_list.append(batch_preds)
                y_pred_proba_list.append(batch_probs)

        # Concatenate results
        y_pred = np.concatenate(y_pred_list)
        y_pred_proba = np.concatenate(y_pred_proba_list)
    else:
        raise ValueError("Must provide either (X_test, y_test) or (features, labels)")

    # Ensure predictions and labels are the right type
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)

    # Calculate metrics (now using numpy arrays)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate ROC curves efficiently
    roc_curves = {}
    for i, class_name in enumerate(class_names):
        # Binary indicator for this class
        class_true = y_true == i
        class_proba = y_pred_proba[:, i]

        # Calculate ROC curve for this class
        fpr, tpr, _ = roc_curve(class_true, class_proba)
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
