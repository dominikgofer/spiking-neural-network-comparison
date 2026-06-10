from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata about a dataset used by the experiment pipeline.

    `encoding` controls how SNN inputs are encoded:
      - "rate":       Bernoulli rate coding over `time_steps` (default for image MNIST-like data).
      - "sequential": treat 28 rows of the image as a 28-step sequence (Sequential MNIST).

    `snn_input_size` is the per-timestep input dimensionality for SpikingMLP.
    """

    name: str
    display_name: str
    cls: type[torch.utils.data.Dataset]
    num_classes: int
    encoding: str
    snn_input_size: int
    ann_input_size: int
    image_shape: tuple[int, int, int]


def _mnist_like(cls: type[torch.utils.data.Dataset], display_name: str, name: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        display_name=display_name,
        cls=cls,
        num_classes=10,
        encoding="rate",
        snn_input_size=28 * 28,
        ann_input_size=28 * 28,
        image_shape=(1, 28, 28),
    )


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "mnist": _mnist_like(datasets.MNIST, "MNIST", "mnist"),
    "fashion-mnist": _mnist_like(datasets.FashionMNIST, "Fashion-MNIST", "fashion-mnist"),
    "kmnist": _mnist_like(datasets.KMNIST, "KMNIST", "kmnist"),
    "seq-mnist": DatasetSpec(
        name="seq-mnist",
        display_name="Sequential MNIST",
        cls=datasets.MNIST,
        num_classes=10,
        encoding="sequential",
        snn_input_size=28,
        ann_input_size=28 * 28,
        image_shape=(1, 28, 28),
    ),
}


def list_datasets() -> list[str]:
    return list(DATASET_REGISTRY.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    return DATASET_REGISTRY[name]


def _create_subset(dataset: torch.utils.data.Dataset, subset_size: int, seed: int) -> Subset:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return Subset(dataset, indices.tolist())


def create_dataloaders(
    dataset_name: str,
    batch_size: int,
    train_subset_size: int,
    test_subset_size: int,
    seed: int,
    data_root: Path | str = "data",
    transform: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DatasetSpec]:
    spec = get_dataset_spec(dataset_name)
    if transform is None:
        transform = transforms.ToTensor()

    train_dataset = spec.cls(root=str(data_root), train=True, download=True, transform=transform)
    test_dataset = spec.cls(root=str(data_root), train=False, download=True, transform=transform)

    train_subset = _create_subset(train_dataset, train_subset_size, seed)
    test_subset = _create_subset(test_dataset, test_subset_size, seed + 1)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, spec
