from skimage.feature import local_binary_pattern
import numpy as np
import cv2
from .base import FeatureExtractor


@FeatureExtractor.register("lbp")
class LBPFeatureExtractor(FeatureExtractor):
    """Extracts texture information using Local Binary Patterns.

    LBP is particularly useful for satellite classification as it captures surface
    texture patterns that can help distinguish different spacecraft components
    like solar panels, antennas, and body structures.
    """

    def __init__(
        self,
        n_points: int = 24,  # Number of points in the circular pattern
        radius: int = 3,  # Radius of the circular pattern
        grid_x: int = 4,  # Number of grid cells in x direction
        grid_y: int = 4,  # Number of grid cells in y direction
        **kwargs
    ):
        super().__init__(
            n_points=n_points, radius=radius, grid_x=grid_x, grid_y=grid_y, **kwargs
        )
        self.n_points = n_points
        self.radius = radius
        self.grid_x = grid_x
        self.grid_y = grid_y

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract LBP features using a spatial pyramid approach.

        Args:
            image: RGB image as numpy array

        Returns:
            Concatenated LBP histogram features from grid cells
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Compute LBP
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method="uniform")

        n_bins = self.n_points + 2  # Number of bins for uniform LBP

        # Initialize list for grid cell histograms
        grid_histograms = []

        # Compute cell size
        cell_height = lbp.shape[0] // self.grid_y
        cell_width = lbp.shape[1] // self.grid_x

        # Compute histograms for each grid cell
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                # Extract cell
                cell = lbp[
                    i * cell_height : (i + 1) * cell_height,
                    j * cell_width : (j + 1) * cell_width,
                ]

                # Compute histogram for cell
                hist, _ = np.histogram(
                    cell.ravel(), bins=n_bins, range=(0, n_bins), density=True
                )

                grid_histograms.append(hist)

        # Concatenate all grid histograms
        return np.concatenate(grid_histograms)

    @property
    def name(self) -> str:
        return "LBP"
