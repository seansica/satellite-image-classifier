# app/satellite_classifier/pipeline.py
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .core.base import Dataset, EvaluationResult
from .data.dataset import DatasetLoader
from .data.splits import create_train_test_split
from .evaluation.metrics import evaluate_model
from .evaluation.visualization import (plot_confusion_matrix,
                                       plot_grid_search_results,
                                       plot_roc_curves,
                                       save_grid_search_results,
                                       save_metrics_summary)
from .features.base import FeatureExtractor
from .models.base import Model


@dataclass
class PipelineConfig:
    """Configuration for the classification pipeline."""
    data_path: Path
    output_path: Path
    feature_extractor: FeatureExtractor
    models: List[Model]
    samples_per_class: Optional[int] = None
    valid_size: float = 0.2
    test_size: float = 0.2
    target_size: tuple[int, int] = (128, 128)
    random_seed: int = 42
    param_grids: Optional[dict] = None

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
        X_train, X_rem, y_train, y_rem = create_train_test_split(
            features,
            labels,
            test_size=self.config.test_size + self.config.valid_size,
            random_state=self.config.random_seed
        )
        X_valid, X_test, y_valid, y_test = create_train_test_split(
            X_rem,
            y_rem,
            test_size=self.config.test_size / (self.config.test_size + self.config.valid_size),
            random_state=self.config.random_seed
        )

        print(f"Train set size: {len(X_train)}, Valid set size: {len(X_valid)}, Test set size: {len(X_test)}")

        # Load parameter grids if provided
        param_grids = None
        if self.config.param_grids:
            import json
            with open(self.config.param_grids, 'r') as f:
                param_grids = json.load(f)
        
        # Train and evaluate models
        results = []
        for model in self.config.models:
            print(f"\nProcessing model: {model.name}")

            training_time = -1.0
            # Check for grid search 
            if param_grids:
                param_grid = param_grids.get(model.name, {})

                if param_grid:
                    print("Tuning hyperparameters...")

                    # Start grid search
                    start_time = time.time()
                    gs_results = model.tune_hyperparameters(
                        X_train, y_train, param_grid
                    )
                    training_time = time.time() - start_time
                    print("Done Grid Search")
                    print(f"Best parameters for {model.name}: {gs_results.best_params_}")

                    # Save results
                    self._save_grid_search_results(gs_results.cv_results_, model.name)

                    # Use the trained best estimator for evaluation
                    tuned_model = gs_results.best_estimator_
                else:
                    print(f"No Grid Search Parameters given for {model.name}")
            
            # Use either tuned_model or the original model
            if tuned_model:
                print(f"Evaluating {model.name} (best estimator)...")
                eval_model = tuned_model
            else:
                print(f"\nTraining {model.name}...")
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                eval_model = model
                print(f"Training time: {training_time:.2f}s")

            # Evaluate the model
            print(f"Evaluating {model.name} train metrics...")
            train_result = evaluate_model(
                eval_model,
                model.name,
                X_train,
                y_train,
                dataset.class_names,
                training_time,
                'train'
            )

            # Save evaluation results
            self._save_results(train_result, dataset.class_names)

            results.append(train_result)

            print(f"{model.name} Train Results:")
            print(f"Train Accuracy:  {train_result.accuracy:.4f}")
            print(f"Train Precision: {train_result.precision:.4f}")
            print(f"Train Recall:    {train_result.recall:.4f}")
            print(f"Train F1 Score:  {train_result.f1_score:.4f}")
            
            print(f"Evaluating {model.name} valid metrics...")
            valid_result = evaluate_model(
                eval_model,
                model.name,
                X_valid,
                y_valid,
                dataset.class_names,
                training_time,
                'valid'
            )

            # Save evaluation results
            self._save_results(valid_result, dataset.class_names)

            results.append(valid_result)

            print(f"{model.name} Valid Results:")
            print(f"Valid Accuracy:  {valid_result.accuracy:.4f}")
            print(f"Valid Precision: {valid_result.precision:.4f}")
            print(f"Valid Recall:    {valid_result.recall:.4f}")
            print(f"Valid F1 Score:  {valid_result.f1_score:.4f}")

            print(f"Evaluating {model.name} train metrics...")
            test_result = evaluate_model(
                eval_model,
                model.name,
                X_test,
                y_test,
                dataset.class_names,
                training_time,
                'test'
            )

            # Save evaluation results
            self._save_results(test_result, dataset.class_names)

            results.append(test_result)
            
            print(f"{model.name} Test Results:")
            print(f"Test Accuracy:  {test_result.accuracy:.4f}")
            print(f"Test Precision: {test_result.precision:.4f}")
            print(f"Test Recall:    {test_result.recall:.4f}")
            print(f"Test F1 Score:  {test_result.f1_score:.4f}")
              
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


    def _save_grid_search_results(
        self,
        results,
        model_name
    ) -> None:
        save_grid_search_results(results, model_name, self.config.output_path / "grid_search")
        plot_grid_search_results(results, model_name, self.config.output_path / "grid_search")
