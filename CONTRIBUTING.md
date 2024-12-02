# Contributing to the Satellite Image Classification Framework

We welcome contributions to extend and improve the framework. This guide explains how to add new components and ensure they integrate smoothly with the existing system.

## Understanding the Base Classes

The framework uses abstract base classes to ensure consistency across implementations. When adding new components, you'll need to work with two primary base classes: `Model` and `FeatureExtractor`. Let's understand what these require:

### The Model Base Class

Any new classification model must inherit from the `Model` base class and implement several required methods:

```python
class Model(ABC):
    @abstractmethod
    def _create_model(self):
        """Create and return the underlying model instance.
        
        This method should instantiate your actual model (e.g., a scikit-learn classifier).
        It will be called during initialization.
        """
        pass
    
    @abstractmethod
    def fit(self, X, y):
        """Train the model on the given data.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions for the given features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Get prediction probabilities for the given features.
        
        This method is required for ROC curve generation and must return
        probability estimates for each class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, n_classes) containing class probabilities
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model.
        
        This name will be used in results and visualizations.
        """
        pass
```

When implementing a new model, you must provide concrete implementations for all these methods. The abstract methods ensure that your model will work properly with the framework's evaluation and visualization systems.

### The FeatureExtractor Base Class

Similarly, any new feature extraction method must inherit from the `FeatureExtractor` base class and implement its required methods:

```python
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image.
        
        This is the core method that transforms an image into a feature vector.
        Your implementation must handle any necessary preprocessing and ensure
        the output is a 1D numpy array of consistent size.
        
        Args:
            image: Input image as a numpy array (RGB format)
            
        Returns:
            1D numpy array containing the extracted features
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the feature extractor.
        
        This name will be used in configuration and logging.
        """
        pass
```

The `extract` method is particularly important as it defines how your feature extractor processes images. Your implementation must:
1. Accept an RGB image as a numpy array
2. Process the image to extract meaningful features
3. Return a 1D numpy array of consistent length (the same size for any input image)

### Important Implementation Considerations

When implementing either base class, keep in mind:

1. Type Safety: Use proper type hints to ensure compatibility
2. Error Handling: Handle edge cases gracefully (e.g., unusually sized images)
3. Performance: Consider memory usage and processing time
4. Documentation: Provide clear docstrings explaining parameters and behavior
5. Consistency: Ensure output formats match the framework's expectations

## Adding New Models

The framework uses a registry pattern that makes it easy to add new classification models. Here's how to add a new model:

1. Create a new file in the `app/models` directory (e.g., `random_forest.py`):

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from .base import Model

@Model.register('random_forest')
class RandomForestModel(Model):
    """Random Forest classifier implementation."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    def _create_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            random_state=self.params['random_state']
        )
    
    def fit(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, X):
        return self._model.predict(X)
    
    def predict_proba(self, X):
        return self._model.predict_proba(X)
    
    @property
    def name(self) -> str:
        return "RandomForest"
```

2. Import the new model in `app/models/registry.py`:

```python
# Import implementations to ensure decorators are executed
from . import svm, logistic, random_forest  # This ensures all implementations are registered
```

3. Import the new model in `app/models/__init__.py`:

```python
from .random_forest import RandomForestModel

__all__ = [
    # ... existing models ...
    'RandomForestModel',
]
```

Your model will now be automatically available through the command-line interface.

## Adding New Feature Extractors

Adding a new feature extraction method follows a similar pattern. Here's how to add a new feature extractor:

1. Create a new file in the `app/features` directory (e.g., `sift.py`):

```python
from .base import FeatureExtractor
import cv2
import numpy as np

@FeatureExtractor.register('sift')
class SIFTFeatureExtractor(FeatureExtractor):
    """SIFT feature extractor implementation."""
    
    def __init__(self, n_keypoints: int = 100, **kwargs):
        super().__init__(n_keypoints=n_keypoints, **kwargs)
        self.sift = cv2.SIFT_create()
        self.n_keypoints = n_keypoints
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect and compute SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Handle cases with few or no keypoints
        if descriptors is None or len(keypoints) < self.n_keypoints:
            return np.zeros((self.n_keypoints * 128,))
        
        # Select top keypoints
        scores = [kp.response for kp in keypoints]
        indices = np.argsort(scores)[-self.n_keypoints:]
        descriptors = descriptors[indices]
        
        # Flatten to consistent size
        return descriptors.flatten()
    
    @property
    def name(self) -> str:
        return "SIFT"
```

2. Import the new feature extractor in `app/features/registry.py`:

```python
# Import implementations to ensure decorators are executed
from . import sift  # This ensures the SIFTFeatureExtractor implementation is registered
```

3. Import the new feature extractor in `app/features/__init__.py`:

```python
from .sift import SIFTFeatureExtractor

__all__ = [
    # ... existing feature extractors ...
    'SIFTFeatureExtractor',
]
```

## Adding New Preprocessing Techniques

To add new preprocessing techniques:

1. Add your preprocessing function to `app/data/preprocessing.py`:

```python
def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply contrast enhancement to an image."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
```

2. Update the `preprocess_image` function to include your new technique:

```python
def preprocess_image(
    image: np.ndarray,
    enhance_contrast: bool = False,  # Add parameter
    **kwargs
) -> np.ndarray:
    """Apply preprocessing steps to an image."""
    img = image.copy()
    
    if enhance_contrast:
        img = enhance_contrast(img)
    
    return img
```