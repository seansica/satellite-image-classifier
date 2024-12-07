import argparse
from pathlib import Path
import logging
import sys
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
        "--output-path",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory for saving results (default: evaluation_results)"
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
        "--samples-per-class",
        type=str,
        default="max",
        help="Number of samples to use per class ('max' for all available)",
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

    # Convert samples_per_class if not 'max'
    if args.samples_per_class != "max":
        try:
            args.samples_per_class = int(args.samples_per_class)
            if args.samples_per_class <= 0:
                raise ValueError
        except ValueError:
            parser.error("--samples-per-class must be 'max' or a positive integer")

    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    try:
        # Get compute device
        device = get_compute_device(args.device)

        # Initialize feature extractor
        feature_extractor = get_feature_extractor(args.feature_extractor)

        # Initialize selected models
        models = [get_model(name) for name in args.models]

        # Create pipeline configuration
        config = PipelineConfig(
            data_path=args.data_path,
            output_path=args.output_path,
            feature_extractor=feature_extractor,
            models=models,
            samples_per_class=args.samples_per_class,
            target_size=tuple(args.image_size),
            random_seed=args.random_seed,
            device=device,  # Pass device to pipeline config
        )

        # Create and run pipeline
        pipeline = Pipeline(config)
        results = pipeline.run()

        # Log final results
        logger.info("\nClassification Results Summary:")
        for result in results:
            logger.info(
                f"\n{result.model_name}:"
                f"\n  Accuracy:  {result.accuracy:.4f}"
                f"\n  Precision: {result.precision:.4f}"
                f"\n  Recall:    {result.recall:.4f}"
                f"\n  F1 Score:  {result.f1_score:.4f}"
            )

        logger.info("\nEvaluation completed successfully. "
                   f"Results saved in: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
