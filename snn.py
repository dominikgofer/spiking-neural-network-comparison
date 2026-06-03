from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader


class SpikingMLP(nn.Module):
    def __init__(self, hidden_size: int = 128, beta: float = 0.95) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_size, 10)
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
    flat_images = images.view(images.size(0), -1)
    random_values = torch.rand((time_steps,) + flat_images.shape, device=flat_images.device)
    return (random_values <= flat_images.unsqueeze(0)).float()


def train_snn(
    model: SpikingMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    time_steps: int,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    hidden_spike_total = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        spike_train = encode_rate_coded(images, time_steps)

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

    return total_loss / total, correct / total, hidden_spike_total / total


@torch.no_grad()
def evaluate_snn(
    model: SpikingMLP,
    loader: DataLoader,
    device: torch.device,
    time_steps: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    hidden_spike_total = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        spike_train = encode_rate_coded(images, time_steps)
        output_spikes, hidden_spikes = model(spike_train)
        logits = output_spikes.sum(dim=0)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
        hidden_spike_total += hidden_spikes.mean().item() * batch_size

    return total_loss / total, correct / total, hidden_spike_total / total