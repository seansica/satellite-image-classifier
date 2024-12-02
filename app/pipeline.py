# app/satellite_classifier/pipeline.py
from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional
import numpy as np

from .core.base import Dataset, EvaluationResult
from .data.dataset import DatasetLoader
from .data.splits import create_train_test_split
from .features.base import FeatureExtractor
from .models.base import Model
from .evaluation.metrics import evaluate_model
from .evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    save_metrics_summary
)

@dataclass
class PipelineConfig:
    """Configuration for the classification pipeline."""
    data_path: Path
    output_path: Path
    feature_extractor: FeatureExtractor
    models: List[Model]
    samples_per_class: Optional[int] = None
    test_size: float = 0.2
    target_size: tuple[int, int] = (128, 128)
    random_seed: int = 42

class Pipeline:
    """Main pipeline for satellite image classification.
    
    This class orchestrates the entire classification process:
    1. Data loading and preprocessing
    2. Feature extraction
    3. Model training and evaluation
    4. Results visualization and saving
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration containing all necessary parameters
        """
        self.config = config
        
        # Create output directories
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        (self.config.output_path / "plots").mkdir(exist_ok=True)
        (self.config.output_path / "metrics").mkdir(exist_ok=True)
    
    def run(self) -> List[EvaluationResult]:
        """Execute the complete classification pipeline.
        
        Returns:
            List of evaluation results for each model
        """
        print("Starting satellite image classification pipeline...")
        
        # Load dataset
        print("\nLoading dataset...")
        dataset = self._load_dataset()
        print(f"Loaded {dataset.n_samples} images from {len(dataset.class_names)} classes")
        
        # Extract features
        print("\nExtracting features...")
        features = self._extract_features(dataset)
        print(f"Extracted features with shape: {features.shape}")
        
        # Prepare labels
        labels = self._encode_labels(dataset.labels)
        
        # Split data
        print("\nSplitting data into train/test sets...")
        X_train, X_test, y_train, y_test = create_train_test_split(
            features,
            labels,
            test_size=self.config.test_size,
            random_state=self.config.random_seed
        )
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Train and evaluate models
        results = []
        for model in self.config.models:
            print(f"\nTraining {model.name}...")
            
            # Train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate the model
            print(f"Evaluating {model.name}...")
            result = evaluate_model(
                model,
                X_test,
                y_test,
                dataset.class_names,
                training_time
            )
            
            # Save evaluation results
            self._save_results(result, dataset.class_names)
            
            results.append(result)
            
            print(f"{model.name} Results:")
            print(f"Accuracy:  {result.accuracy:.4f}")
            print(f"Precision: {result.precision:.4f}")
            print(f"Recall:    {result.recall:.4f}")
            print(f"F1 Score:  {result.f1_score:.4f}")
        
        return results
    
    def _load_dataset(self) -> Dataset:
        """Load and preprocess the image dataset."""
        loader = DatasetLoader(
            data_path=self.config.data_path,
            target_size=self.config.target_size,
            samples_per_class=self.config.samples_per_class,
            random_seed=self.config.random_seed
        )
        return loader.load()
    
    def _extract_features(self, dataset: Dataset) -> np.ndarray:
        """Extract features from all images in the dataset."""
        features = []
        for image in dataset.images:
            features.append(self.config.feature_extractor.extract(image))
        return np.array(features)
    
    def _encode_labels(self, labels: List[str]) -> np.ndarray:
        """Convert string labels to numerical indices."""
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        return encoder.fit_transform(labels)
    
    def _save_results(
        self,
        result: EvaluationResult,
        class_names: List[str]
    ) -> None:
        """Save evaluation results and visualizations."""
        # Create visualizations
        plot_confusion_matrix(
            result,
            class_names,
            self.config.output_path / "plots"
        )
        
        plot_roc_curves(
            result,
            self.config.output_path / "plots"
        )
        
        # Save metrics summary
        save_metrics_summary(
            result,
            self.config.output_path / "metrics"
        )