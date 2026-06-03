from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class Config:
    batch_size: int = 128
    epochs: int = 3
    hidden_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 7
    train_subset_size: int = 12000
    test_subset_size: int = 2000
    time_steps: int = 20
    beta: float = 0.95


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_results_dir() -> Path:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def create_subset(dataset: torch.utils.data.Dataset, subset_size: int, seed: int) -> Subset:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return Subset(dataset, indices.tolist())


def create_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_subset = create_subset(train_dataset, config.train_subset_size, config.seed)
    test_subset = create_subset(test_dataset, config.test_subset_size, config.seed + 1)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader


def save_history_csv(history: list[dict[str, object]], path: Path) -> None:
    if not history:
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_results(history: list[dict[str, object]], output_dir: Path) -> None:
    ann_train = [entry for entry in history if entry["model"] == "ANN" and entry["split"] == "train"]
    ann_eval = [entry for entry in history if entry["model"] == "ANN" and entry["split"] == "test"]
    snn_train = [entry for entry in history if entry["model"] == "SNN" and entry["split"] == "train"]
    snn_eval = [entry for entry in history if entry["model"] == "SNN" and entry["split"] == "test"]

    epochs = [entry["epoch"] for entry in ann_train]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [entry["accuracy"] for entry in ann_train], label="ANN train")
    axes[0].plot(epochs, [entry["accuracy"] for entry in ann_eval], label="ANN test")
    axes[0].plot(epochs, [entry["accuracy"] for entry in snn_train], label="SNN train")
    axes[0].plot(epochs, [entry["accuracy"] for entry in snn_eval], label="SNN test")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, [entry["loss"] for entry in ann_train], label="ANN train")
    axes[1].plot(epochs, [entry["loss"] for entry in ann_eval], label="ANN test")
    axes[1].plot(epochs, [entry["loss"] for entry in snn_train], label="SNN train")
    axes[1].plot(epochs, [entry["loss"] for entry in snn_eval], label="SNN test")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_curves.png", dpi=150)
    plt.close(fig)


def save_summary(summary: dict[str, object], history: list[dict[str, object]], output_dir: Path) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_history_csv(history, output_dir / "history.csv")

    with (output_dir / "comparison.md").open("w", encoding="utf-8") as file:
        file.write("# Porownanie ANN i SNN na MNIST\n\n")
        file.write("| Model | Test accuracy | Test loss |\n")
        file.write("| --- | ---: | ---: |\n")
        file.write(f"| ANN | {summary['ann_test_accuracy']:.4f} | {summary['ann_test_loss']:.4f} |\n")
        file.write(f"| SNN | {summary['snn_test_accuracy']:.4f} | {summary['snn_test_loss']:.4f} |\n")


def save_results(summary: dict[str, object], history: list[dict[str, object]], output_dir: Path) -> None:
    save_summary(summary, history, output_dir)
    plot_results(history, output_dir)