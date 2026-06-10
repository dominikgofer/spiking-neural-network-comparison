from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from models.ann import ANNClassifier, evaluate_ann
from models.cnn import CNNClassifier, evaluate_cnn
from models.snn import SpikingMLP, evaluate_snn


class NoisyLoader:
    """Wraps a DataLoader so each batch's images are perturbed with Gaussian noise.

    Designed for short test sweeps; deterministic given a seed.
    """

    def __init__(self, base_loader: DataLoader, sigma: float, seed: int) -> None:
        self._base_loader = base_loader
        self._sigma = sigma
        self._seed = seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self._seed)
        for images, labels in self._base_loader:
            if self._sigma > 0:
                noise = torch.randn(images.shape, generator=generator) * self._sigma
                images = (images + noise).clamp(0.0, 1.0)
            yield images, labels

    def __len__(self) -> int:
        return len(self._base_loader)


def noise_robustness_sweep(
    models: dict[str, torch.nn.Module],
    base_test_loader: DataLoader,
    device: torch.device,
    sigmas: tuple[float, ...] | list[float],
    *,
    time_steps: int,
    encoding: str,
    seed: int,
) -> list[dict[str, object]]:
    """Evaluate each model under several Gaussian noise levels and return rows for plotting."""
    rows: list[dict[str, object]] = []
    for sigma in sigmas:
        noisy_loader = NoisyLoader(base_test_loader, sigma=sigma, seed=seed)
        for name, model in models.items():
            if isinstance(model, SpikingMLP):
                res = evaluate_snn(model, noisy_loader, device, time_steps, encoding=encoding)
            elif isinstance(model, CNNClassifier):
                res = evaluate_cnn(model, noisy_loader, device)
            elif isinstance(model, ANNClassifier):
                res = evaluate_ann(model, noisy_loader, device)
            else:
                continue
            rows.append(
                {
                    "model": name,
                    "sigma": float(sigma),
                    "accuracy": float(res["accuracy"]),
                    "loss": float(res["loss"]),
                }
            )
    return rows
