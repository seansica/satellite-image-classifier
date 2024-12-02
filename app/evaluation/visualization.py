from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from ..core.base import EvaluationResult

def plot_confusion_matrix(
    result: EvaluationResult,
    class_names: List[str],
    output_path: Path,
    normalize: bool = True
) -> None:
    """Plot and save confusion matrix visualization.
    
    Args:
        result: Evaluation result containing confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
        normalize: Whether to show proportions instead of counts
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize if requested
    cm = result.confusion_matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(f'Confusion Matrix - {result.model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(
        output_path / f"{result.model_name}_confusion_matrix.png",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_roc_curves(
    result: EvaluationResult,
    output_path: Path
) -> None:
    """Plot and save ROC curves for all classes.
    
    Args:
        result: Evaluation result containing ROC curves
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Define a color cycle for different classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                   'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    # Plot ROC curve for each class
    for class_name, color in zip(result.roc_curves.keys(), colors):
        curves = result.roc_curves[class_name]
        plt.plot(
            curves['fpr'],
            curves['tpr'],
            color=color,
            lw=2,
            label=f'{class_name} (AUC = {curves["auc"]:.2f})'
        )
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {result.model_name}')
    plt.legend(loc="lower right")
    
    # Save plot
    plt.savefig(
        output_path / f"{result.model_name}_roc_curves.png",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def save_metrics_summary(
    result: EvaluationResult,
    output_path: Path
) -> None:
    """Save evaluation metrics to a text file.
    
    Args:
        result: Evaluation result containing metrics
        output_path: Path to save the summary
    """
    summary_path = output_path / f"{result.model_name}_metrics_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write(f"Model: {result.model_name}\n")
        f.write("-" * 50 + "\n\n")
        
        # Basic metrics
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy:  {result.accuracy:.4f}\n")
        f.write(f"Precision: {result.precision:.4f}\n")
        f.write(f"Recall:    {result.recall:.4f}\n")
        f.write(f"F1 Score:  {result.f1_score:.4f}\n\n")
        
        # Training time
        f.write(f"Training Time: {result.training_time:.2f} seconds\n\n")
        
        # ROC AUC scores
        f.write("ROC AUC Scores:\n")
        for class_name, curves in result.roc_curves.items():
            f.write(f"{class_name}: {curves['auc']:.4f}\n")