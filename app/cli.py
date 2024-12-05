# app/satellite_classifier/cli.py
import argparse
import logging
import sys
from pathlib import Path
from typing import List

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
        help="Path to the image dataset directory"
    )
    
    # Optional arguments
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
        help="Models to evaluate (default: all available models)"
    )
    
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Number of samples to use per class (default: use all available)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2)"
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

    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Enable hyperparameter tuning for models."
    )
    
    parser.add_argument(
        "--param-grids",
        type=str,
        help="Path to JSON file specifying parameter grids for each model."
    )
    
    return parser

def main() -> None:
    """Main entry point for the classification system."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize feature extractor
        feature_extractor = get_feature_extractor(args.feature_extractor)
        
        # Initialize selected models
        models = [get_model(name) for name in args.models]
        

        # Load parameter grids if provided
        param_grids = None
        if args.param_grids:
            import json
            with open(args.param_grids, 'r') as f:
                param_grids = json.load(f)
        print(param_grids)

        # Create pipeline configuration
        config = PipelineConfig(
            data_path=args.data_path,
            output_path=args.output_path,
            feature_extractor=feature_extractor,
            models=models,
            samples_per_class=args.samples_per_class,
            test_size=args.test_size,
            target_size=tuple(args.image_size),
            random_seed=args.random_seed,
            tune_hyperparameters=args.tune_hyperparameters,
            param_grids=param_grids,
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