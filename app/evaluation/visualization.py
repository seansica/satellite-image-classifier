from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import cycle
import os

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


def save_grid_search_results(results, model_name: str, output_path: Path) -> None:
    """
    Save the grid search results to a CSV file in the specified output path.
    If the output directory does not exist, it will be created.

    Args:
        results (dict): Dictionary containing grid search results
        model_name (str): Name of the model used in grid search
        output_path (Path): Path where the results will be saved
    """
    # Ensure the directory exists
    output_path.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
    summary_path = output_path / f"{model_name}_grid_search_results.csv"
    # Convert grid search results to a pandas DataFrame
    grid_results = pd.DataFrame(results)
    # Save the results to a CSV file
    grid_results.to_csv(summary_path, index=False)
    print(f"Grid search results saved to: {summary_path}")

def plot_grid_search_results(results, model_name, output_path):
    """
    Plot grid search results for different models.

    Args:
        results (dict): Dictionary containing grid search results
        model_name (str): Name of the model (SVM or LogisticRegression)
        output_path (Path): Path where the plot will be saved
    """
    if model_name == "SVM":
        plot_grid_search_svm(results, model_name, output_path)
    if model_name == "LogisticRegression":
        plot_grid_search_lgr(results, model_name, output_path)
    else:
        print(f"{model_name} visualization not implemented yet")




def plot_grid_search_svm(results, model_name, output_path) -> None:
    """
    Plot grid search results for Support Vector Machine (SVM).

    Args:
        results (dict): Dictionary containing grid search results
        model_name (str): Name of the model (SVM)
        output_path (Path): Path where the plot will be saved
    """
    # Extract scores
    scores = results['mean_test_score']
    Cs = [param['C'] for param in results['params']]
    kernels = [param['kernel'] for param in results['params']]
    # Create a 2D grid for visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    for kernel in ['linear', 'rbf', 'poly']:
        kernel_scores = [scores[i] for i in range(len(scores)) if kernels[i] == kernel]
        kernel_Cs = [Cs[i] for i in range(len(scores)) if kernels[i] == kernel]
        ax.plot(kernel_Cs, kernel_scores, marker='o', label=f'Kernel: {kernel}')
        ax.set_xscale('log')  # Log scale for C
        ax.set_xlabel('C (Regularization Parameter)', fontsize=12)
        ax.set_ylabel('Mean Accuracy (CV)', fontsize=12)
        ax.set_title('SVM Performance with Different Kernels and C Values', fontsize=14)
        ax.legend()
        plt.grid()
    # Save the plot
    plot_path = output_path / f"{model_name}_grid_search_plot.png"
    plt.savefig(plot_path)

def plot_grid_search_lgr(results, model_name, output_path) -> None:
    """
    Plot grid search results for Logistic Regression.

    Args:
        results (dict): Dictionary containing grid search results
        model_name (str): Name of the model (LogisticRegression)
        output_path (Path): Path where the plot will be saved
    """
    # Extract scores, C values, and penalty types
    scores = results['mean_test_score']
    Cs = [param['C'] for param in results['params']]
    penalties = [param['solver'] for param in results['params']]
    # Create a 2D grid for visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot for each penalty type (L1, L2)
    for penalty in  ["lbfgs", "sag"]:
        penalty_scores = [scores[i] for i in range(len(scores)) if penalties[i] == penalty]
        penalty_Cs = [Cs[i] for i in range(len(scores)) if penalties[i] == penalty]
        # Plot the results for the current penalty type
        ax.plot(penalty_Cs, penalty_scores, marker='o', label=f'solver: {penalty}')
        # Set the x-axis to a logarithmic scale for better visualization of C
        ax.set_xscale('log')
        ax.set_xlabel('C (Regularization Parameter)', fontsize=12)
        ax.set_ylabel('Mean Accuracy (CV)', fontsize=12)
        ax.set_title(f'{model_name} Performance with Different Penalties and C Values', fontsize=14)
        ax.legend()
        plt.grid()
    # Save the plot
    plot_path = output_path / f"{model_name}_logreg_grid_search_plot.png"
    plt.savefig(plot_path)