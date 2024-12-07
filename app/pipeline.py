from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    output_path: Path
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


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        (self.config.output_path / "plots").mkdir(exist_ok=True)
        (self.config.output_path / "metrics").mkdir(exist_ok=True)

        # Move feature extractor to device
        self.config.feature_extractor.to(self.config.device)

        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)

    def run(self) -> List[EvaluationResult]:
        print("Starting satellite image classification pipeline...")

        # Load datasets
        train_dataset, val_dataset, test_dataset = self._load_dataset()
        print(
            f"Loaded datasets - Train: {len(train_dataset)}, "
            f"Val: {len(val_dataset)}, Test: {len(test_dataset)} images"
        )

        # Create data loaders
        train_loader = self._create_loader(train_dataset, shuffle=True)
        val_loader = self._create_loader(val_dataset, shuffle=False)
        test_loader = self._create_loader(test_dataset, shuffle=False)

        # Get a sample batch to determine feature dimension
        sample_inputs, _ = next(iter(train_loader))
        sample_inputs = sample_inputs.to(self.config.device)
        sample_features = self.config.feature_extractor.extract(sample_inputs)
        input_dim = sample_features.shape[1]

        print(f"Feature dimension: {input_dim}")

        results = []
        for model_wrapper in self.config.models:
            # Initialize model with correct dimensions
            model = model_wrapper.create(
                input_dim=input_dim,
                num_classes=len(train_dataset.class_names),
                model_params=self.config.model_params,
            ).to(self.config.device)

            print(f"\nTraining {model.name}...")
            start_time = time.time()

            # Train the model
            self._train_model(model, train_loader, val_loader)
            training_time = time.time() - start_time

            # Evaluate on all splits
            splits = [
                ("train", train_loader),
                ("val", val_loader),
                ("test", test_loader),
            ]

            for split_name, loader in splits:
                predictions, labels = self._get_predictions(model, loader)

                result = evaluate_model(
                    model=model,
                    class_names=train_dataset.class_names,
                    training_time=training_time,
                    dataset_split=split_name,
                    features=predictions,
                    labels=labels,
                )
                self._save_results(result, train_dataset.class_names)
                results.append(result)

        return results

    def _create_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for a dataset."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
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
        """Load and prepare the datasets."""
        # Create the transform for loading and preprocessing images
        transform = transforms.Compose(
            [
                transforms.Resize(self.config.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load all splits
        train_data_dir = self.config.data_path / "train_rgb"
        val_data_dir = self.config.data_path / "validate_rgb"
        test_data_dir = self.config.data_path / "test_rgb"

        # Load images and labels for each split
        train_images, train_labels = self._load_split(train_data_dir)
        val_images, val_labels = self._load_split(val_data_dir)
        test_images, test_labels = self._load_test_split(test_data_dir)

        # Create metadata
        metadata = DatasetMetadata(
            name=self.config.data_path.name,
            class_names=self._get_class_names(train_data_dir),
            n_classes=len(self._get_class_names(train_data_dir)),
            n_samples=len(train_images) + len(val_images) + len(test_images),
            data_path=self.config.data_path,
        )

        # Create dataset objects
        train_dataset = Dataset(train_images, train_labels, metadata, transform)
        val_dataset = Dataset(val_images, val_labels, metadata, transform)
        test_dataset = Dataset(test_images, test_labels, metadata, transform)

        # Move to device
        return (
            train_dataset.to(self.config.device),
            val_dataset.to(self.config.device),
            test_dataset.to(self.config.device),
        )

    def _load_split(self, split_dir: Path) -> Tuple[List[torch.Tensor], List[str]]:
        """Load images and labels from a directory split."""
        images = []
        labels = []

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            # Get image files
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

            # Apply train_ratio if less than 1.0
            if self.config.train_ratio < 1.0:
                num_samples = int(len(image_files) * self.config.train_ratio)
                if num_samples > 0:
                    rng = np.random.RandomState(self.config.random_seed)
                    image_files = rng.choice(
                        image_files, size=num_samples, replace=False
                    ).tolist()
                else:
                    # Ensure at least one sample per class
                    image_files = image_files[:1]

            for img_path in tqdm(image_files, desc=f"Loading {class_dir.name}"):
                images.append(img_path)
                labels.append(class_dir.name)

        return images, labels

    def _load_test_split(self, test_dir: Path) -> Tuple[List[torch.Tensor], List[str]]:
        """Load test images and labels.

        The test set is organized differently from train/val:
        - Images are in a flat directory without class subdirectories
        - Labels are in a CSV file with numeric class indices
        - File naming follows pattern 'image_XXXXX_img.jpg'
        """
        images = []
        labels = []

        # Get ordered list of class names from training directory for index mapping
        class_names = self._get_class_names(self.config.data_path / "train_rgb")

        # Create index to class name mapping
        idx_to_class = {idx: name for idx, name in enumerate(class_names)}

        # Load test labels
        labels_file = self.config.data_path / "test_labels.csv"
        df = pd.read_csv(labels_file)

        # First collect all valid images and labels
        valid_samples = []
        for _, row in df.iterrows():
            # Map numeric class index to class name
            class_idx = int(row["class"])
            if class_idx not in idx_to_class:
                continue

            class_name = idx_to_class[class_idx]

            # Handle both .png and .jpg extensions
            img_path_png = test_dir / f"image_{str(row['id']).zfill(5)}_img.png"
            img_path_jpg = test_dir / f"image_{str(row['id']).zfill(5)}_img.jpg"

            if img_path_png.exists():
                valid_samples.append((img_path_png, class_name))
            elif img_path_jpg.exists():
                valid_samples.append((img_path_jpg, class_name))

        # Apply test_ratio if less than 1.0
        if self.config.test_ratio < 1.0 and valid_samples:
            num_samples = int(len(valid_samples) * self.config.test_ratio)
            if num_samples > 0:
                rng = np.random.RandomState(self.config.random_seed)
                indices = rng.choice(
                    len(valid_samples), size=num_samples, replace=False
                )
                valid_samples = [valid_samples[i] for i in indices]
            else:
                # Ensure at least one sample if possible
                valid_samples = valid_samples[:1]

        # Unzip the samples into separate lists
        images, labels = zip(*valid_samples) if valid_samples else ([], [])

        if not images:
            raise ValueError(
                f"No test images found in {test_dir}. "
                "Check that the test directory and labels file are properly structured."
            )

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

    def _save_results(self, result: EvaluationResult, class_names: List[str]) -> None:
        """Save evaluation results and plots."""
        plot_confusion_matrix(result, class_names, self.config.output_path / "plots")
        plot_roc_curves(result, self.config.output_path / "plots")
        save_metrics_summary(result, self.config.output_path / "metrics")
