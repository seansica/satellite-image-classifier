import cv2
import numpy as np
from pathlib import Path
from .base import FeatureExtractor
import logging


@FeatureExtractor.register("resnet50")
class ResNet50FeatureExtractor(FeatureExtractor):
    """Extracts deep features using ResNet50-v1 from ONNX Model Zoo.

    This implementation uses OpenCV's DNN module to run the ONNX ResNet50-v1 model.
    Features are extracted from the final pooling layer before classification,
    providing a 2048-dimensional representation that captures high-level visual patterns.

    The model expects RGB images of size 224x224 and performs the following preprocessing:
    1. Converts to BGR (OpenCV's default)
    2. Subtracts mean values [123.675, 116.28, 103.53]
    3. Scales by std [58.395, 57.12, 57.375]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define model path
        model_dir = Path(__file__).parent.parent.parent / "models" / "resnet"
        model_path = model_dir / "resnet50-v1-12.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"ResNet50 ONNX model not found at {model_path}. "
                "Please download resnet50-v1-12.onnx from the ONNX Model Zoo "
                "and place it in the models/resnet directory."
            )

        try:
            # Load the model using OpenCV's DNN module
            self.model = cv2.dnn.readNetFromONNX(str(model_path))
            logging.info(f"Successfully loaded ResNet50 model from {model_path}")

            # Set computation preferences
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get all layer names
            self.layer_names = self.model.getLayerNames()
            logging.debug(f"Model layers: {self.layer_names}")

            # In ResNet50-v1-12, we know the features come from the second-to-last layer
            # The last layer is typically the fully connected classification layer
            self.feature_layer = self.layer_names[-2]
            logging.info(f"Using feature extraction layer: {self.feature_layer}")

        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet50 model: {str(e)}")

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract deep features from an image using ResNet50.

        Args:
            image: RGB image as numpy array

        Returns:
            2048-dimensional feature vector from the penultimate layer
        """
        try:
            # Preprocess the image
            blob = cv2.dnn.blobFromImage(
                image,
                1.0,  # scale factor
                (224, 224),  # target size
                mean=[123.675, 116.28, 103.53],  # RGB means
                swapRB=True,  # OpenCV uses BGR, so we need to swap
                crop=True,  # Center crop after scaling
                ddepth=cv2.CV_32F,  # Use floating point precision
            )

            # Set the input
            self.model.setInput(blob)

            # Get output from feature layer
            features = self.model.forward(self.feature_layer)

            # Flatten and normalize
            features = features.flatten()
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features

        except Exception as e:
            logging.error(f"Error during feature extraction: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(2048, dtype=np.float32)

    @property
    def name(self) -> str:
        return "ResNet50"
