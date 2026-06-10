from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ANNClassifier(nn.Module):
    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.flatten(images)
        hidden = F.relu(self.fc1(x))
        return self.fc2(hidden)

    def forward_with_hidden(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.flatten(images)
        hidden = F.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return logits, hidden


def train_ann(
    model: ANNClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_ann(
    model: ANNClassifier,
    loader: DataLoader,
    device: torch.device,
    collect_predictions: bool = False,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    sparsity_sum = 0.0
    predictions_all: list[int] = []
    labels_all: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits, hidden = model.forward_with_hidden(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
        sparsity_sum += (hidden == 0).float().mean().item() * batch_size

        if collect_predictions:
            predictions_all.extend(predictions.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    result: dict[str, object] = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "sparsity": sparsity_sum / total,
    }
    if collect_predictions:
        result["predictions"] = predictions_all
        result["labels"] = labels_all
    return result
