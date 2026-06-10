from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class Config:
    batch_size: int = 128
    epochs: int = 15
    hidden_size: int = 128
    learning_rate: float = 1e-3
    train_subset_size: int = 12000
    test_subset_size: int = 2000
    time_steps: int = 20
    beta: float = 0.95
    target_accuracy: float = 0.90
    noise_sigmas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    seeds: tuple[int, ...] = (7, 13, 42)
    benchmark_batch_sizes: tuple[int, ...] = (1, 8, 32, 128, 512)
    benchmark_repeats: int = 5
    benchmark_warmup_batches: int = 2


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_results_dir(dataset_name: str) -> Path:
    results_dir = Path("results") / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def count_parameters(model: torch.nn.Module) -> dict[str, float]:
    """Return parameter count and model size in MB (assuming float32)."""
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = params * 4 / (1024 * 1024)
    return {"params": params, "trainable_params": trainable, "size_mb": size_mb}


def energy_proxy_ann(params: int) -> float:
    """Rough energy proxy for ANN: ~2 FLOPs per parameter per forward pass."""
    return 2.0 * params


def energy_proxy_snn(params: int, spike_rate: float, time_steps: int) -> float:
    """Energy proxy for SNN: only neurons that spike trigger downstream MACs.

    Approximation: per timestep, fraction `spike_rate` of upstream activations are 1,
    so on average `spike_rate * params` MACs per timestep, summed over `time_steps`.
    """
    return spike_rate * params * time_steps


def classification_metrics(predictions: list[int], labels: list[int]) -> dict[str, object]:
    """Compute F1 / precision / recall (macro) and confusion matrix using scikit-learn."""
    preds = np.asarray(predictions)
    labs = np.asarray(labels)
    return {
        "f1_macro": float(f1_score(labs, preds, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labs, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labs, preds, average="macro", zero_division=0)),
        "report": classification_report(labs, preds, zero_division=0, digits=4),
        "confusion_matrix": confusion_matrix(labs, preds).tolist(),
    }


def epoch_to_target_accuracy(history: list[dict[str, object]], model_name: str, target: float) -> int | None:
    """First epoch in which `model_name` test accuracy >= target. None if never reached."""
    for entry in history:
        if entry.get("model") != model_name or entry.get("split") != "test":
            continue
        if float(entry.get("accuracy", 0.0)) >= target:
            return int(entry["epoch"])
    return None


def time_to_target_accuracy(
    history: list[dict[str, object]], model_name: str, target: float
) -> float | None:
    """Cumulative train time (seconds) until `model_name` first reaches target test accuracy."""
    cumulative = 0.0
    for entry in history:
        if entry.get("model") != model_name:
            continue
        if entry.get("split") == "train":
            cumulative += float(entry.get("train_time_s", 0.0))
        elif entry.get("split") == "test" and float(entry.get("accuracy", 0.0)) >= target:
            return cumulative
    return None


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


def save_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, default=str)


def plot_training_curves(history: list[dict[str, object]], output_dir: Path, title_suffix: str = "") -> None:
    """Accuracy and loss curves per model, averaged over seeds.

    Expects history rows with keys: epoch, model, split, loss, accuracy, seed.
    """
    if not history:
        return
    models = sorted({entry["model"] for entry in history})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for model_name in models:
        for split in ("train", "test"):
            rows = [e for e in history if e["model"] == model_name and e["split"] == split]
            if not rows:
                continue
            epochs = sorted({int(e["epoch"]) for e in rows})
            acc_mean = [
                float(np.mean([float(e["accuracy"]) for e in rows if int(e["epoch"]) == ep]))
                for ep in epochs
            ]
            loss_mean = [
                float(np.mean([float(e["loss"]) for e in rows if int(e["epoch"]) == ep]))
                for ep in epochs
            ]
            label = f"{model_name} {split}"
            linestyle = "-" if split == "test" else "--"
            axes[0].plot(epochs, acc_mean, label=label, linestyle=linestyle)
            axes[1].plot(epochs, loss_mean, label=label, linestyle=linestyle)

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(f"Training curves {title_suffix}".strip())
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], output_path: Path, title: str) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    num_classes = arr.shape[0]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_raster(
    raster_hidden: torch.Tensor,
    raster_output: torch.Tensor,
    raster_labels: torch.Tensor,
    raster_inputs: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    """Raster plots of hidden-layer spike activity for a few samples.

    raster_hidden / raster_output: (T, B, neurons). raster_inputs: (B, 1, 28, 28) or (B, 28, 28).
    """
    hidden = raster_hidden.numpy()
    output = raster_output.numpy()
    labels = raster_labels.numpy()
    inputs = raster_inputs.numpy()

    num_samples = min(4, hidden.shape[1])
    fig, axes = plt.subplots(num_samples, 3, figsize=(11, 2.5 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])

    for i in range(num_samples):
        img = inputs[i]
        if img.ndim == 3:
            img = img.squeeze(0)
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"input (label={int(labels[i])})", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(hidden[:, i, :].T, aspect="auto", cmap="binary", interpolation="nearest")
        axes[i, 1].set_xlabel("time step")
        axes[i, 1].set_ylabel("hidden neuron")
        axes[i, 1].set_title("hidden spikes", fontsize=9)

        axes[i, 2].imshow(output[:, i, :].T, aspect="auto", cmap="binary", interpolation="nearest")
        axes[i, 2].set_xlabel("time step")
        axes[i, 2].set_ylabel("output neuron")
        axes[i, 2].set_title("output spikes", fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pareto_time_accuracy(rows: list[dict[str, object]], output_path: Path, title: str) -> None:
    """Scatter of (train_time, accuracy) coloured by model."""
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    models = sorted({str(r["model"]) for r in rows})
    markers = ["o", "s", "^", "D", "v"]
    for idx, model_name in enumerate(models):
        xs = [float(r["train_time_s"]) for r in rows if r["model"] == model_name]
        ys = [float(r["accuracy"]) for r in rows if r["model"] == model_name]
        ax.scatter(xs, ys, label=model_name, marker=markers[idx % len(markers)], s=70)
    ax.set_xlabel("Total training time [s]")
    ax.set_ylabel("Final test accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_latency_vs_batch(rows: list[dict[str, object]], output_path: Path, title: str) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    models = sorted({str(r["model"]) for r in rows})
    for model_name in models:
        sub = sorted([r for r in rows if r["model"] == model_name], key=lambda r: int(r["batch_size"]))
        xs = [int(r["batch_size"]) for r in sub]
        latency_batch = [float(r["latency_per_batch_ms"]) for r in sub]
        latency_sample = [float(r["latency_per_sample_ms"]) for r in sub]
        axes[0].plot(xs, latency_batch, marker="o", label=model_name)
        axes[1].plot(xs, latency_sample, marker="o", label=model_name)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()
    axes[0].set_ylabel("Latency per batch [ms]")
    axes[0].set_title("Inference latency per batch")
    axes[1].set_ylabel("Latency per sample [ms]")
    axes[1].set_title("Inference latency per sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_noise_robustness(rows: list[dict[str, object]], output_path: Path, title: str) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    models = sorted({str(r["model"]) for r in rows})
    for model_name in models:
        sub = sorted([r for r in rows if r["model"] == model_name], key=lambda r: float(r["sigma"]))
        xs = [float(r["sigma"]) for r in sub]
        ys = [float(r["accuracy"]) for r in sub]
        ax.plot(xs, ys, marker="o", label=model_name)
    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@dataclass
class SeedAggregate:
    """Aggregate per-seed final test metrics into mean / std."""

    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=0)) if self.values else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std, "n": len(self.values)}
