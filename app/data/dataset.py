from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import cv2

from ..core.base import Dataset, DatasetMetadata
from ..core.types import ImageArray
from .preprocessing import preprocess_image

class DatasetLoader:
    """Handles loading and preprocessing of satellite image datasets.
    
    This class manages the entire data loading pipeline, including:
    - Reading images from disk
    - Applying preprocessing steps
    - Creating balanced class distributions
    - Splitting data into train/test sets
    """
    
    def __init__(
        self,
        data_path: Path,
        target_size: Tuple[int, int] = (128, 128),
        samples_per_class: Optional[int] = None,
        random_seed: int = 42
    ):
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.samples_per_class = samples_per_class
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
    
    def load(self) -> Dataset:
        """Load the dataset from disk.
        
        This method:
        1. Discovers all image files and their classes
        2. Loads and preprocesses images
        3. Creates a balanced dataset if requested
        4. Returns a Dataset object with all data and metadata
        
        Returns:
            Dataset object containing images and labels
        """
        # Get all class directories
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        class_names = [d.name for d in class_dirs]
        
        print(f"Found {len(class_names)} classes: {', '.join(class_names)}")
        
        images = []
        labels = []
        
        # Load images for each class
        for class_dir in tqdm(class_dirs, desc="Loading classes"):
            # Get all image files for this class
            image_files = list(class_dir.glob('*.jpg'))
            
            # If samples_per_class is set, randomly sample that many images
            if self.samples_per_class is not None:
                if len(image_files) > self.samples_per_class:
                    image_files = np.random.choice(
                        image_files,
                        size=self.samples_per_class,
                        replace=False
                    ).tolist()
            
            # Load and preprocess each image
            for img_path in tqdm(
                image_files,
                desc=f"Loading {class_dir.name}",
                leave=False
            ):
                try:
                    # Load and preprocess the image
                    img = load_image(img_path, self.target_size)
                    img = preprocess_image(img)
                    
                    images.append(img)
                    labels.append(class_dir.name)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
        
        # Create dataset metadata
        metadata = DatasetMetadata(
            name=self.data_path.name,
            class_names=class_names,
            n_classes=len(class_names),
            n_samples=len(images),
            data_path=self.data_path
        )
        
        return Dataset(images, labels, metadata)

def load_image(path: Path, target_size: Tuple[int, int]) -> ImageArray:
    """Load and resize an image from disk.
    
    Args:
        path: Path to the image file
        target_size: Desired output size as (width, height)
        
    Returns:
        Preprocessed image as numpy array
        
    Raises:
        ValueError: If the image cannot be loaded
    """
    # Read image in BGR format
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    
    return img