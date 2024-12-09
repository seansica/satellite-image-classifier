from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from sys import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import platform

from .models.svm import SVMModel
from .models.rf import RandomForestModel
from .models.gbm import GradientBoostingModel
from .models.logistic import LogisticRegressionModel

from .models.registry import LazyModel
from .core.base import EvaluationResult
from .data.dataset import Dataset, DatasetMetadata
from .features.base import FeatureExtractor
from .models.base import Model
from .evaluation.metrics import evaluate_model
from .evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    save_metrics_summary,
)


@dataclass
class PipelineConfig:
    """Configuration for the classification pipeline."""
    data_path: Path
    feature_extractor: FeatureExtractor
    models: List["LazyModel"]
    device: torch.device
    train_ratio: float = 1.0
    test_ratio: float = 1.0
    target_size: tuple[int, int] = (128, 128)
    random_seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 0.001
    epochs: int = 1
    patience: int = 10

    # Model-specific parameters
    model_params: dict = field(default_factory=dict)  # Store model-specific parameters


@dataclass
class ExperimentMetadata:
    """Metadata about the experiment run."""

    timestamp: datetime
    config: Dict[str, Any]
    system_info: Dict[str, str]

    @classmethod
    def create(cls, config: PipelineConfig) -> "ExperimentMetadata":
        """Create metadata from pipeline config."""
        # Create a clean dictionary of config, excluding model weights
        config_dict = {
            "batch_size": config.batch_size,
            "data_path": str(config.data_path),
            "device": str(config.device),
            "epochs": config.epochs,
            "feature_extractor": {
                "type": config.feature_extractor.__class__.__name__,
                "output_dim": getattr(config.feature_extractor, "_output_dim", None),
            },
            "train_ratio": config.train_ratio,
            "test_ratio": config.test_ratio,
            "target_size": config.target_size,
            "learning_rate": config.learning_rate,
            "model_params": config.model_params,
            "models": [m.name for m in config.models],
        }

        return cls(
            timestamp=datetime.now(),
            config=config_dict,
            system_info={
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else None
                ),
            },
        )

    def save(self, path: Path) -> None:
        """Save metadata to disk."""
        with open(path / "experiment_config.yaml", "w") as f:
            yaml.dump(
                asdict(self), f, indent=2, sort_keys=False, default_flow_style=False
            )


def generate_output_dir(config: PipelineConfig) -> Path:
    """Generate deterministic output directory name.

    Format: YYYY-MM-DD_HH-MM-SS_MODEL_FEATURE_trainXpYY
    Example: 2024-12-08_14-35-22_SVM_ResNet50_train0p80

    Args:
        config: Pipeline configuration containing model and feature extractor info

    Returns:
        Path: Directory path for experiment outputs
    """
    # Format datetime
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H-%M-%S")

    # Get model names (e.g., "SVM" or "SVM-RF" for multiple)
    models_str = "-".join(sorted(m.name.upper() for m in config.models))

    # Get feature extractor name without "FeatureExtractor" suffix
    feature_name = config.feature_extractor.__class__.__name__
    if feature_name.endswith("FeatureExtractor"):
        feature_name = feature_name[:-15]  # Remove "FeatureExtractor"

    # Format train ratio (e.g., 0.05 -> "0p05", 1.00 -> "1p00")
    train_ratio = f"train{config.train_ratio:0.2f}".replace(".", "p")

    # Construct directory name
    dir_name = f"{date_str}_{time_str}_{models_str}_{feature_name}__{train_ratio}"

    return Path("results") / dir_name


class Pipeline:

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with config and optional CLI args."""
        self.config = config

        # Generate output directory
        self.output_dir = generate_output_dir(config)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.models_dir = self.output_dir / "models"
        self.metrics_dir = self.output_dir / "metrics"

        for directory in [self.plots_dir, self.models_dir, self.metrics_dir]:
            directory.mkdir(exist_ok=True)

        # Save experiment metadata
        metadata = ExperimentMetadata.create(config)
        metadata.save(self.output_dir)

        # Add cache for extracted features
        self.feature_cache = {}

        # Move feature extractor to device
        self.config.feature_extractor.to(self.config.device)

        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)

    def _extract_and_cache_features(
        self, loader: DataLoader, split_name: str
    ) -> torch.Tensor:
        """Extract features once and cache them for reuse."""
        if split_name in self.feature_cache:
            return self.feature_cache[split_name]

        all_features = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(
                loader, desc=f"Extracting features for {split_name}"
            ):
                inputs = inputs.to(self.config.device)
                features = self.config.feature_extractor.extract(inputs)

                # Keep features on device if possible
                all_features.append(features)
                all_labels.append(labels.to(self.config.device))

        # Concatenate once instead of repeatedly
        features = torch.cat(all_features)
        labels = torch.cat(all_labels)

        self.feature_cache[split_name] = (features, labels)
        return features, labels

    def _create_evaluation_loader(self, dataset: Dataset) -> DataLoader:
        """Create a loader optimized for evaluation."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size * 2,  # Double batch size for eval
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def run(self) -> List[EvaluationResult]:
        # Load datasets
        train_dataset, val_dataset, test_dataset = self._load_dataset()

        # Create loaders with optimized batch sizes
        train_loader = self._create_loader(train_dataset, shuffle=True)
        val_loader = self._create_evaluation_loader(val_dataset)
        test_loader = self._create_evaluation_loader(test_dataset)

        # Extract features once for each split
        print("Extracting features for all splits...")
        train_features, train_labels = self._extract_and_cache_features(
            train_loader, "train"
        )
        val_features, val_labels = self._extract_and_cache_features(val_loader, "val")
        test_features, test_labels = self._extract_and_cache_features(
            test_loader, "test"
        )

        all_results = []  # Store results from all models

        for model_wrapper in self.config.models:
            model = model_wrapper.create(
                input_dim=train_features.shape[1],
                num_classes=len(train_dataset.class_names),
                model_params=self.config.model_params,
            ).to(self.config.device)

            print(f"\nTraining {model.name}...")
            start_time = time.time()

            # Train using cached features
            self._train_model_with_features(
                model,
                train_features,
                train_labels,
                (val_features, val_labels) if val_loader else None,
            )
            training_time = time.time() - start_time

            # Save trained model
            model_path = self.models_dir / model.name
            model_path.mkdir(exist_ok=True)
            self._save_model(model, model_path)

            # Evaluate on all splits using cached features
            splits = [
                ("train", (train_features, train_labels)),
                ("val", (val_features, val_labels)),
                ("test", (test_features, test_labels)),
            ]

            # Collect results for all splits for this model
            model_results = []
            for split_name, (features, labels) in splits:
                result = evaluate_model(
                    model=model,
                    class_names=train_dataset.class_names,
                    training_time=training_time,
                    dataset_split=split_name,
                    features=features,
                    labels=labels,
                )
                model_results.append(result)

            # Save results for this model
            self._save_results(model_results, train_dataset.class_names)

            # Add to overall results
            all_results.extend(model_results)

        return all_results

    def _train_model_with_features(
        self,
        model: Model,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """Train model using pre-extracted features."""
        optimizer = self._configure_optimizer(model)
        best_val_loss = float("inf")
        patience_counter = 0

        # Calculate number of batches
        batch_size = self.config.batch_size
        n_samples = train_features.shape[0]
        indices = torch.arange(n_samples)

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            # Shuffle indices for each epoch
            shuffled_indices = indices[torch.randperm(n_samples)]

            # Process in batches
            for i in tqdm(
                range(0, n_samples, batch_size),
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            ):
                batch_indices = shuffled_indices[i : i + batch_size]
                batch_features = train_features[batch_indices]
                batch_labels = train_labels[batch_indices]

                loss = model.train_step((batch_features, batch_labels), optimizer)
                train_loss += loss

            avg_train_loss = train_loss / (n_samples // batch_size)

            # Validation phase
            if val_data is not None:
                val_features, val_labels = val_data
                val_loss = self._validate_model_with_features(
                    model, val_features, val_labels
                )
                print(
                    f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        print(f"Early stopping after {epoch + 1} epochs")
                        break
            else:
                print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

    def _validate_model_with_features(
        self,
        model: Model,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> float:
        """Validate model using pre-extracted features."""
        model.eval()
        total_loss = 0.0
        batch_size = self.config.batch_size * 2  # Larger batch size for validation
        n_samples = val_features.shape[0]

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_features = val_features[i : i + batch_size]
                batch_labels = val_labels[i : i + batch_size]

                loss = model.validation_step((batch_features, batch_labels))
                total_loss += loss

        return total_loss / (n_samples // batch_size)

    def _create_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for a dataset."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def _configure_optimizer(self, model: Model) -> torch.optim.Optimizer:
        """Configure optimizer with appropriate regularization per model type.

        Args:
            model: The model instance to configure optimization for

        Returns:
            Configured optimizer instance with appropriate regularization

        Raises:
            ValueError: If model type is not supported
        """
        base_lr = self.config.learning_rate

        if isinstance(model, SVMModel):
            # SVM typically benefits from L2 regularization via weight decay
            # Often uses slightly higher weight decay than other linear models
            return torch.optim.SGD(
                model.parameters(),
                lr=base_lr,
                weight_decay=0.01,  # Higher L2 penalty typical for SVMs
                momentum=0.9,  # Momentum helps with margin optimization
            )

        elif isinstance(model, LogisticRegressionModel):
            # Logistic regression typically uses L2 regularization
            # But usually with a lighter touch than SVM
            return torch.optim.SGD(
                model.parameters(),
                lr=base_lr,
                weight_decay=0.001,  # Lighter L2 penalty
                momentum=0.9,
            )

        elif isinstance(model, (RandomForestModel, GradientBoostingModel)):
            # For sklearn-based models, use minimal optimizer since they handle their own training
            return torch.optim.SGD(
                model.parameters(),  # Only contains dummy parameter
                lr=0.0,  # Learning rate doesn't matter since these models don't use gradient updates
            )

        else:
            supported_models = [
                "SVMModel",
                "LogisticRegressionModel",
                "RandomForestModel",
                "GradientBoostingModel",
            ]
            raise ValueError(
                f"Unsupported model type: {type(model)}. "
                f"Supported models are: {', '.join(supported_models)}"
            )

    def _train_model(
        self,
        model: Model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Train a model using the configured parameters."""
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}"
            )

            for batch in train_batches:
                # Move batch to device and extract features
                inputs, targets = batch
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                # Extract features and ensure they require gradients
                features = self.config.feature_extractor.extract(inputs)
                if not features.requires_grad:
                    features = features.detach().requires_grad_(True)

                # Training step
                loss = model.train_step((features, targets), optimizer)
                train_loss += loss

                # Update progress bar
                train_batches.set_postfix({"loss": f"{loss:.4f}"})

            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_model(model, val_loader)
                print(
                    f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        print(f"Early stopping after {epoch + 1} epochs")
                        break
            else:
                print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

    def _validate_model(self, model: Model, val_loader: DataLoader) -> float:
        """Validate a model and return average validation loss."""
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device and extract features
                inputs, targets = batch
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                # Extract features
                features = self.config.feature_extractor.extract(inputs)

                # Validation step
                loss = model.validation_step((features, targets))
                total_loss += loss

        return total_loss / len(val_loader)

    def _get_predictions(
        self, model: Model, loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get model predictions for a dataset."""
        model.eval()
        all_predictions = []
        all_labels = []
        all_features = []  # Store features for evaluation

        with torch.no_grad():
            for inputs, labels in loader:
                # Move input to device and extract features
                inputs = inputs.to(self.config.device)
                features = self.config.feature_extractor.extract(inputs)
                all_features.append(features.cpu())  # Store features

                # Get predictions
                predictions = model(features).argmax(dim=1)

                # Collect results
                all_predictions.append(predictions.cpu())
                all_labels.append(labels)

        # Concatenate all batches
        all_features = torch.cat(all_features)  # Features for evaluation
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        return all_features, all_labels  # Return features instead of predictions

    def _load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and prepare the datasets with debugging information."""
        # Create the transform
        transform = transforms.Compose(
            [
                transforms.Resize(self.config.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load directories
        train_dir = self.config.data_path / "train_rgb"
        val_dir = self.config.data_path / "validate_rgb"
        test_dir = self.config.data_path / "test_rgb"

        # Get class names from train directory for consistent ordering
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        print(f"Found classes: {class_names}")
        print(f"Number of classes: {len(class_names)}")

        # Load splits with debugging
        train_images, train_labels = self._load_split(train_dir, "train")
        val_images, val_labels = self._load_split(val_dir, "val")
        test_images, test_labels = self._load_test_split(test_dir, class_names)

        # Debug class distribution
        self._debug_class_distribution(train_labels, "Training")
        self._debug_class_distribution(val_labels, "Validation")
        self._debug_class_distribution(test_labels, "Test")

        # Create metadata
        metadata = DatasetMetadata(
            name=self.config.data_path.name,
            class_names=class_names,
            n_classes=len(class_names),
            n_samples=len(train_images) + len(val_images) + len(test_images),
            data_path=self.config.data_path,
        )

        # Create datasets
        train_dataset = Dataset(train_images, train_labels, metadata, transform)
        val_dataset = Dataset(val_images, val_labels, metadata, transform)
        test_dataset = Dataset(test_images, test_labels, metadata, transform)

        return train_dataset, val_dataset, test_dataset

    def _load_split(
        self, split_dir: Path, split_name: str
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Load images and labels from a directory split with debugging."""
        images = []
        labels = []

        print(f"\nLoading {split_name} split from {split_dir}")

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            # Get image files
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            print(f"Found {len(image_files)} images in class {class_dir.name}")

            # Apply train_ratio if less than 1.0
            if self.config.train_ratio < 1.0:
                original_count = len(image_files)
                num_samples = int(len(image_files) * self.config.train_ratio)
                if num_samples > 0:
                    rng = np.random.RandomState(self.config.random_seed)
                    image_files = rng.choice(
                        image_files, size=num_samples, replace=False
                    ).tolist()
                else:
                    image_files = image_files[:1]
                print(
                    f"Applied train_ratio {self.config.train_ratio}: {original_count} -> {len(image_files)} images"
                )

            for img_path in tqdm(image_files, desc=f"Loading {class_dir.name}"):
                images.append(img_path)
                labels.append(class_dir.name)

        return images, labels

    def _debug_class_distribution(self, labels: List[str], split_name: str):
        """Print class distribution information."""
        from collections import Counter

        distribution = Counter(labels)
        print(f"\n{split_name} set class distribution:")
        for class_name, count in sorted(distribution.items()):
            print(f"{class_name}: {count}")

    def _load_test_split(
        self, test_dir: Path, class_names: List[str]
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Load test images and labels with class mapping."""
        images = []
        labels = []

        # Create index to class name mapping
        idx_to_class = {idx: name for idx, name in enumerate(class_names)}

        print(f"\nLoading test split from {test_dir}")
        print(f"Class index mapping: {idx_to_class}")

        # Load test labels CSV
        labels_file = self.config.data_path / "test_labels.csv"
        df = pd.read_csv(labels_file)

        print(f"Unique class indices in test CSV: {sorted(df['class'].unique())}")

        valid_samples = []
        for _, row in df.iterrows():
            class_idx = int(row["class"])
            if class_idx not in idx_to_class:
                print(f"WARNING: Found invalid class index {class_idx} in test labels")
                continue

            class_name = idx_to_class[class_idx]
            img_path_jpg = test_dir / f"image_{str(row['id']).zfill(5)}_img.jpg"
            if img_path_jpg.exists():
                valid_samples.append((img_path_jpg, class_name))

        print(f"Found {len(valid_samples)} valid test samples")

        if self.config.test_ratio < 1.0:
            original_count = len(valid_samples)
            num_samples = int(len(valid_samples) * self.config.test_ratio)
            if num_samples > 0:
                rng = np.random.RandomState(self.config.random_seed)
                valid_samples = [
                    valid_samples[i]
                    for i in rng.choice(
                        len(valid_samples), size=num_samples, replace=False
                    )
                ]
            else:
                valid_samples = valid_samples[:1]
            print(
                f"Applied test_ratio {self.config.test_ratio}: {original_count} -> {len(valid_samples)} samples"
            )

        images, labels = zip(*valid_samples) if valid_samples else ([], [])
        return images, labels

    def _get_class_names(self, train_dir: Path) -> List[str]:
        """Get sorted list of class names from training directory."""
        return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    def _process_dataset(
        self, dataset: Dataset, desc: str = "Processing dataset"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a dataset to extract features and get labels."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Keep order consistent for feature extraction
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=desc):
                # Extract features
                features = self.config.feature_extractor.extract(images)

                # Move to CPU and convert to numpy
                features = features.cpu().numpy()
                labels = labels.cpu().numpy()

                all_features.append(features)
                all_labels.append(labels)

        # Concatenate all batches
        return (np.concatenate(all_features), np.concatenate(all_labels))

    def _save_results(
        self, results: List[EvaluationResult], class_names: List[str]
    ) -> None:
        """Save evaluation results and plots for all splits.

        Args:
            results: List of evaluation results from all splits for a model
            class_names: List of class names for visualization
        """
        # Save metrics summary with all splits
        save_metrics_summary(results, self.metrics_dir)

        # Generate plots for all splits
        for result in results:
            # Save confusion matrix for each split
            plot_confusion_matrix(result, class_names, self.plots_dir)

            # Only generate ROC curves for test split
            if result.dataset_split == "test":
                plot_roc_curves(result, self.plots_dir)

    def _save_model(self, model: Model, path: Path) -> None:
        """Save model weights and architecture summary."""
        # Save model weights
        torch.save(model.state_dict(), path / "model.pt")

        # Save model architecture summary
        summary = {
            "name": model.name,
            "type": model.__class__.__name__,
            "input_dim": model.input_dim,
            "num_classes": model.num_classes,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "hyperparameters": model.params,
        }

        with open(path / "architecture.yaml", "w") as f:
            yaml.dump(summary, f, indent=2, sort_keys=False, default_flow_style=False)
