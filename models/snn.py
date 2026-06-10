from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader


class SpikingMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_size: int = 128,
        num_classes: int = 10,
        beta: float = 0.95,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, spike_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps, batch_size = spike_train.shape[0], spike_train.shape[1]
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=spike_train.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=spike_train.device)

        output_spikes = []
        hidden_spikes = []

        for time_step in range(time_steps):
            cur1 = self.fc1(spike_train[time_step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            output_spikes.append(spk2)
            hidden_spikes.append(spk1)

        return torch.stack(output_spikes), torch.stack(hidden_spikes)


def encode_rate_coded(images: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Rate coding: pixel intensity (0..1) -> Bernoulli spike probability per timestep."""
    flat_images = images.view(images.size(0), -1)
    random_values = torch.rand((time_steps,) + flat_images.shape, device=flat_images.device)
    return (random_values <= flat_images.unsqueeze(0)).float()


def encode_sequential(images: torch.Tensor) -> torch.Tensor:
    """Sequential MNIST encoding: feed image row-by-row as a 28-step sequence.

    Input shape: (B, 1, 28, 28) or (B, 28, 28). Output: (28, B, 28) where the
    leading dim is time and each timestep is one row of pixel values.
    """
    if images.dim() == 4:
        images = images.squeeze(1)
    batch_size = images.size(0)
    sequence = images.view(batch_size, 28, 28)
    return sequence.permute(1, 0, 2).contiguous()


def encode_inputs(images: torch.Tensor, encoding: str, time_steps: int) -> torch.Tensor:
    if encoding == "rate":
        return encode_rate_coded(images, time_steps)
    if encoding == "sequential":
        return encode_sequential(images)
    raise ValueError(f"Unknown encoding: {encoding}")


def train_snn(
    model: SpikingMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    time_steps: int,
    encoding: str = "rate",
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    hidden_spike_total = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        spike_train = encode_inputs(images, encoding, time_steps)

        optimizer.zero_grad()
        output_spikes, hidden_spikes = model(spike_train)
        logits = output_spikes.sum(dim=0)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
        hidden_spike_total += hidden_spikes.mean().item() * batch_size

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "hidden_spike_rate": hidden_spike_total / total,
    }


@torch.no_grad()
def evaluate_snn(
    model: SpikingMLP,
    loader: DataLoader,
    device: torch.device,
    time_steps: int,
    encoding: str = "rate",
    collect_predictions: bool = False,
    collect_raster: bool = False,
    raster_samples: int = 8,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    hidden_spike_total = 0.0
    predictions_all: list[int] = []
    labels_all: list[int] = []
    raster_hidden: torch.Tensor | None = None
    raster_output: torch.Tensor | None = None
    raster_labels: torch.Tensor | None = None
    raster_inputs: torch.Tensor | None = None

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        spike_train = encode_inputs(images, encoding, time_steps)
        output_spikes, hidden_spikes = model(spike_train)
        logits = output_spikes.sum(dim=0)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
        hidden_spike_total += hidden_spikes.mean().item() * batch_size

        if collect_predictions:
            predictions_all.extend(predictions.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

        if collect_raster and raster_hidden is None:
            take = min(raster_samples, batch_size)
            raster_hidden = hidden_spikes[:, :take, :].detach().cpu()
            raster_output = output_spikes[:, :take, :].detach().cpu()
            raster_labels = labels[:take].detach().cpu()
            raster_inputs = images[:take].detach().cpu()

    result: dict[str, object] = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "hidden_spike_rate": hidden_spike_total / total,
    }
    if collect_predictions:
        result["predictions"] = predictions_all
        result["labels"] = labels_all
    if collect_raster and raster_hidden is not None:
        result["raster_hidden"] = raster_hidden
        result["raster_output"] = raster_output
        result["raster_labels"] = raster_labels
        result["raster_inputs"] = raster_inputs
    return result
