from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(1)
        elif images.dim() == 2:
            images = images.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._features(images)
        x = x.flatten(1)
        hidden = F.relu(self.fc1(x))
        return self.fc2(hidden)

    def forward_with_hidden(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._features(images)
        x = x.flatten(1)
        hidden = F.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return logits, hidden


def train_cnn(
    model: CNNClassifier,
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
def evaluate_cnn(
    model: CNNClassifier,
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
