from typing import Dict, List, Optional, Union, TypeVar, Generic
from pathlib import Path
import numpy as np
import numpy.typing as npt

# Type definitions for improved code clarity and type checking
ImageArray = npt.NDArray[np.uint8]  # RGB image arrays
FeatureArray = npt.NDArray[np.float64]  # Extracted feature arrays
LabelArray = npt.NDArray[np.int64]  # Label arrays

# Generic type for registry keys
T = TypeVar('T')