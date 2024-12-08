import argparse
import logging
import sys
from pathlib import Path
from typing import List
import torch

from .features.registry import get_feature_extractor, list_feature_extractors
from .models.registry import get_model, list_models
from .pipeline import Pipeline, PipelineConfig
from .utils.logging import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Satellite Image Classification System"
    )

    # Required arguments
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the ICIP dataset directory",
    )

    # Optional arguments
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device to use (default: auto-select best available)",
    )

    parser.add_argument(
        "--feature-extractor",
        choices=list_feature_extractors(),
        default="hog",
        help="Feature extraction method to use (default: hog)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list_models(),
        default=list_models(),
        help="Models to evaluate (default: all available models)",
    )

    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to use (default: 1)"
    )

    # Add model-specific hyperparameters
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=10,  # Reduced from 100
        help="Number of trees in Random Forest (default: 10)",
    )

    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=3,  # Reduced from 5
        help="Maximum depth of trees in Random Forest (default: 3)",
    )

    parser.add_argument(
        "--rf-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for Random Forest trees (default: input_dim//4 or 10)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Ratio of training data to use (between 0 and 1, default: 1.0)",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=1.0,
        help="Ratio of test data to use (between 0 and 1, default: 1.0)",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("WIDTH", "HEIGHT"),
        help="Target image size as width height (default: 128 128)"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )

    return parser


def get_compute_device(device_arg: str) -> torch.device:
    """Determine the compute device to use.

    Args:
        device_arg: Device argument from CLI ('auto', 'cuda', 'mps', or 'cpu')

    Returns:
        torch.device: Selected compute device
    """
    logger = logging.getLogger(__name__)

    if device_arg != "auto":
        # If user specified a device, try to use it
        if device_arg == "cuda":
            if not torch.cuda.is_available():
                logger.error("CUDA device requested but CUDA is not available")
                sys.exit(1)
            device = torch.device("cuda")
            logger.info("Using specified device: NVIDIA GPU (CUDA)")

        elif device_arg == "mps":
            if not torch.backends.mps.is_available():
                logger.warning(
                    "MPS device requested but MPS is not available. Falling back to CPU."
                )
                device = torch.device("cpu")
                logger.info("Using device: CPU (MPS unavailable)")
            else:
                device = torch.device("mps")
                logger.info("Using specified device: Apple Metal (MPS)")

        else:  # cpu
            device = torch.device("cpu")
            logger.info("Using specified device: CPU")

    else:
        # Auto-select best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Auto-selected device: NVIDIA GPU (CUDA)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-selected device: Apple Metal (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Auto-selected device: CPU")

    return device


def main() -> None:
    """Main entry point for the classification system."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # Validate ratios
    if not 0 < args.train_ratio <= 1:
        parser.error("--train-ratio must be between 0 and 1")
    if not 0 < args.test_ratio <= 1:
        parser.error("--test-ratio must be between 0 and 1")

    try:
        # Get compute device
        device = get_compute_device(args.device)

        # Initialize feature extractor
        feature_extractor = get_feature_extractor(args.feature_extractor)

        # Collect model-specific parameters
        model_params = {
            "rf": {  # Parameters for Random Forest
                "n_estimators": args.rf_n_estimators,
                "max_depth": args.rf_max_depth,
                "hidden_dim": args.rf_hidden_dim,
            }
            # TODO add other model-specific parameters here
        }

        # Initialize selected models
        models = [get_model(name) for name in args.models]

        # Create pipeline configuration
        config = PipelineConfig(
            data_path=args.data_path,
            feature_extractor=feature_extractor,
            models=models,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            target_size=tuple(args.image_size),
            random_seed=args.random_seed,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_params=model_params,  # Pass model parameters to config
        )

        # Create and run pipeline
        pipeline = Pipeline(config)
        results = pipeline.run()

        # Log final results
        logger.info("\nClassification Results Summary:")
        for result in results:
            logger.info(
                f"\n{result.model_name} ({result.dataset_split}):"
                f"\n  Accuracy:  {result.accuracy:.4f}"
                f"\n  Precision: {result.precision:.4f}"
                f"\n  Recall:    {result.recall:.4f}"
                f"\n  F1 Score:  {result.f1_score:.4f}"
                f"\n"
                f"\n  Time To Train: {result.training_time:.2f}s"
                f"\n"
            )

        logger.info(
            "\nEvaluation completed successfully. "
            f"Results saved in: {pipeline.output_dir}"
        )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
