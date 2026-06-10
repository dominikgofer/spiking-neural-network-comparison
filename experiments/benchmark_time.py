from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from models.snn import SpikingMLP, encode_inputs


@dataclass
class BenchmarkResult:
    name: str
    mean: float
    std: float
    n: int
    unit: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "mean": self.mean, "std": self.std, "n": self.n, "unit": self.unit}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _make_inputs(spec_image_shape: tuple[int, int, int], batch_size: int, device: torch.device) -> torch.Tensor:
    """Random tensor with the dataset image shape; values in [0,1] for rate coding."""
    shape = (batch_size,) + spec_image_shape
    return torch.rand(shape, device=device)


def _make_labels(batch_size: int, num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, num_classes, (batch_size,), device=device)


def _run_forward(model: torch.nn.Module, inputs: torch.Tensor, time_steps: int, encoding: str) -> torch.Tensor:
    if isinstance(model, SpikingMLP):
        spike_train = encode_inputs(inputs, encoding, time_steps)
        output_spikes, _ = model(spike_train)
        return output_spikes.sum(dim=0)
    return model(inputs)


def benchmark_throughput(
    model: torch.nn.Module,
    image_shape: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    *,
    mode: str,
    batch_size: int,
    time_steps: int,
    encoding: str,
    repeats: int,
    warmup_batches: int,
) -> BenchmarkResult:
    """Measure samples-per-second throughput.

    mode in {"train", "inference"}. For "train" runs forward + backward + step.
    Returns mean / std over `repeats` independent timings.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) if mode == "train" else None

    inputs = _make_inputs(image_shape, batch_size, device)
    labels = _make_labels(batch_size, num_classes, device)

    if mode == "train":
        model.train()
    else:
        model.eval()

    for _ in range(warmup_batches):
        if mode == "train":
            optimizer.zero_grad()
            logits = _run_forward(model, inputs, time_steps, encoding)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _run_forward(model, inputs, time_steps, encoding)
    _synchronize(device)

    timings: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        t0 = time.perf_counter()
        if mode == "train":
            optimizer.zero_grad()
            logits = _run_forward(model, inputs, time_steps, encoding)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _run_forward(model, inputs, time_steps, encoding)
        _synchronize(device)
        elapsed = time.perf_counter() - t0
        timings.append(batch_size / elapsed)

    return BenchmarkResult(
        name=f"throughput_{mode}",
        mean=float(np.mean(timings)),
        std=float(np.std(timings, ddof=0)),
        n=repeats,
        unit="samples/s",
    )


def benchmark_latency_vs_batch(
    model: torch.nn.Module,
    image_shape: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    *,
    batch_sizes: list[int] | tuple[int, ...],
    time_steps: int,
    encoding: str,
    repeats: int,
    warmup_batches: int,
) -> list[dict[str, object]]:
    """For each batch size, measure inference latency per batch (ms) and per sample (ms)."""
    rows: list[dict[str, object]] = []
    model.eval()
    for batch_size in batch_sizes:
        inputs = _make_inputs(image_shape, batch_size, device)
        for _ in range(warmup_batches):
            with torch.no_grad():
                _run_forward(model, inputs, time_steps, encoding)
        _synchronize(device)
        timings: list[float] = []
        for _ in range(repeats):
            _synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                _run_forward(model, inputs, time_steps, encoding)
            _synchronize(device)
            timings.append((time.perf_counter() - t0) * 1000.0)
        rows.append(
            {
                "batch_size": batch_size,
                "latency_per_batch_ms": float(np.mean(timings)),
                "latency_per_batch_std_ms": float(np.std(timings, ddof=0)),
                "latency_per_sample_ms": float(np.mean(timings)) / batch_size,
            }
        )
    return rows


def benchmark_time_per_timestep(
    snn_model: SpikingMLP,
    image_shape: tuple[int, int, int],
    device: torch.device,
    *,
    batch_size: int,
    time_steps_list: list[int] | tuple[int, ...],
    encoding: str,
    repeats: int,
    warmup_batches: int,
) -> list[dict[str, object]]:
    """How latency scales with number of timesteps (slope ~= per-timestep cost)."""
    rows: list[dict[str, object]] = []
    snn_model.eval()
    for ts in time_steps_list:
        inputs = _make_inputs(image_shape, batch_size, device)
        for _ in range(warmup_batches):
            with torch.no_grad():
                _run_forward(snn_model, inputs, ts, encoding)
        _synchronize(device)
        timings: list[float] = []
        for _ in range(repeats):
            _synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                _run_forward(snn_model, inputs, ts, encoding)
            _synchronize(device)
            timings.append((time.perf_counter() - t0) * 1000.0)
        rows.append(
            {
                "time_steps": ts,
                "latency_ms": float(np.mean(timings)),
                "latency_std_ms": float(np.std(timings, ddof=0)),
                "per_timestep_ms": float(np.mean(timings)) / ts,
            }
        )
    return rows


def profile_training_step(
    model: torch.nn.Module,
    image_shape: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    *,
    batch_size: int,
    time_steps: int,
    encoding: str,
    repeats: int,
    warmup_batches: int,
) -> dict[str, BenchmarkResult]:
    """Decompose one training iteration into data-prep / forward / backward / step times."""
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()

    inputs_cpu = torch.rand((batch_size,) + image_shape)
    labels_cpu = torch.randint(0, num_classes, (batch_size,))

    for _ in range(warmup_batches):
        optimizer.zero_grad()
        inputs = inputs_cpu.to(device)
        labels = labels_cpu.to(device)
        logits = _run_forward(model, inputs, time_steps, encoding)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    _synchronize(device)

    data_times: list[float] = []
    forward_times: list[float] = []
    backward_times: list[float] = []
    step_times: list[float] = []

    for _ in range(repeats):
        optimizer.zero_grad()
        _synchronize(device)

        t0 = time.perf_counter()
        inputs = inputs_cpu.to(device)
        labels = labels_cpu.to(device)
        _synchronize(device)
        data_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        logits = _run_forward(model, inputs, time_steps, encoding)
        loss = F.cross_entropy(logits, labels)
        _synchronize(device)
        forward_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        loss.backward()
        _synchronize(device)
        backward_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        optimizer.step()
        _synchronize(device)
        step_times.append((time.perf_counter() - t0) * 1000.0)

    def to_result(name: str, values: list[float]) -> BenchmarkResult:
        return BenchmarkResult(
            name=name,
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=0)),
            n=repeats,
            unit="ms",
        )

    return {
        "data_to_device": to_result("data_to_device", data_times),
        "forward": to_result("forward", forward_times),
        "backward": to_result("backward", backward_times),
        "optimizer_step": to_result("optimizer_step", step_times),
    }
