from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""

    name: str
    class_names: List[str]
    n_classes: int
    n_samples: int
    data_path: Path


class Dataset(TorchDataset):
    """PyTorch Dataset for satellite image classification with lazy loading."""

    def __init__(
        self,
        images: Sequence[Union[torch.Tensor, Path]],
        labels: Sequence[str],
        metadata: DatasetMetadata,
        transform: Optional[callable] = None,
    ):
        if len(images) != len(labels):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of labels ({len(labels)})"
            )

        self.images = images
        self.labels = labels
        self.metadata = metadata
        self.transform = transform
        self.device = torch.device("cpu")  # Default device

        # Create label encoder
        self._label_to_idx = {
            label: idx for idx, label in enumerate(metadata.class_names)
        }
        self._idx_to_label = {idx: label for label, idx in self._label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, image_source: Union[torch.Tensor, Path]) -> torch.Tensor:
        """Load an image from a path or return the tensor.

        Important: Always returns tensor on CPU, device transfer happens later."""
        if isinstance(image_source, torch.Tensor):
            return image_source.cpu()  # Ensure CPU
        else:
            # Load image from path
            with Image.open(image_source) as img:
                if self.transform:
                    img_tensor = self.transform(img)
                else:
                    # Default transformation if none provided
                    from torchvision import transforms

                    default_transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                        ]
                    )
                    img_tensor = default_transform(img)
                return img_tensor.cpu()  # Ensure CPU

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        image_source = self.images[idx]
        label = self.labels[idx]

        # Load image on CPU
        image = self._load_image(image_source)

        # Convert string label to index and create tensor
        label_idx = self._label_to_idx[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        # Note: We don't move to self.device here - that's handled by DataLoader
        return image, label_tensor

    def get_label_name(self, idx: int) -> str:
        return self._idx_to_label[idx]

    @property
    def class_names(self) -> List[str]:
        return self.metadata.class_names

    @property
    def num_classes(self) -> int:
        return self.metadata.n_classes

    def to(self, device: torch.device) -> "Dataset":
        """Update the target device for the dataset.

        Note: Actual device transfer happens in DataLoader via pin_memory and device transfer.
        """
        self.device = device
        return self

    def subset(self, indices: Sequence[int]) -> "Dataset":
        images = [self.images[i] for i in indices]
        labels = [self.labels[i] for i in indices]

        new_metadata = DatasetMetadata(
            name=self.metadata.name,
            class_names=self.metadata.class_names,
            n_classes=self.metadata.n_classes,
            n_samples=len(indices),
            data_path=self.metadata.data_path,
        )

        return Dataset(
            images=images,
            labels=labels,
            metadata=new_metadata,
            transform=self.transform,
        )
