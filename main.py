"""CLI orchestrator for the ANN vs CNN vs SNN comparison experiment.

Usage examples:
  uv run main.py
  uv run main.py --dataset fashion-mnist --models ann snn cnn --epochs 15
  uv run main.py --dataset seq-mnist --models ann snn --seeds 7 13 42
  uv run main.py --datasets mnist fashion-mnist kmnist seq-mnist --models ann cnn snn
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from datasets import create_dataloaders, get_dataset_spec, list_datasets
from experiments import (
    benchmark_latency_vs_batch,
    benchmark_throughput,
    benchmark_time_per_timestep,
    noise_robustness_sweep,
    profile_training_step,
)
from models import (
    ANNClassifier,
    CNNClassifier,
    SpikingMLP,
    evaluate_ann,
    evaluate_cnn,
    evaluate_snn,
    train_ann,
    train_cnn,
    train_snn,
)
from utils import (
    Config,
    SeedAggregate,
    classification_metrics,
    count_parameters,
    energy_proxy_ann,
    energy_proxy_snn,
    ensure_results_dir,
    epoch_to_target_accuracy,
    get_device,
    plot_confusion_matrix,
    plot_latency_vs_batch,
    plot_noise_robustness,
    plot_pareto_time_accuracy,
    plot_raster,
    plot_training_curves,
    save_history_csv,
    save_json,
    set_seed,
    time_to_target_accuracy,
)


SUPPORTED_MODELS = ("ann", "cnn", "snn")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANN vs CNN vs SNN comparison")
    parser.add_argument("--dataset", default=None, choices=list_datasets(), help="single dataset shortcut")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list_datasets(),
        help="run multiple datasets sequentially (overrides --dataset)",
    )
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS), choices=SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-subset-size", type=int, default=None)
    parser.add_argument("--test-subset-size", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> Config:
    base = Config()
    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.train_subset_size is not None:
        overrides["train_subset_size"] = args.train_subset_size
    if args.test_subset_size is not None:
        overrides["test_subset_size"] = args.test_subset_size
    if args.time_steps is not None:
        overrides["time_steps"] = args.time_steps
    if not overrides:
        return base
    data = {**base.__dict__, **overrides}
    return Config(**data)


def _resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return list(args.datasets)
    if args.dataset:
        return [args.dataset]
    return ["mnist"]


def _build_model(model_name: str, spec, config: Config) -> torch.nn.Module:
    if model_name == "ann":
        return ANNClassifier(input_size=spec.ann_input_size, hidden_size=config.hidden_size, num_classes=spec.num_classes)
    if model_name == "cnn":
        return CNNClassifier(num_classes=spec.num_classes)
    if model_name == "snn":
        return SpikingMLP(
            input_size=spec.snn_input_size,
            hidden_size=config.hidden_size,
            num_classes=spec.num_classes,
            beta=config.beta,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _run_epoch(
    model_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    test_loader,
    device: torch.device,
    time_steps: int,
    encoding: str,
) -> tuple[dict[str, float], dict[str, float], float]:
    t0 = time.perf_counter()
    if model_name == "ann":
        train_metrics = train_ann(model, train_loader, optimizer, device)
        test_metrics = evaluate_ann(model, test_loader, device)
    elif model_name == "cnn":
        train_metrics = train_cnn(model, train_loader, optimizer, device)
        test_metrics = evaluate_cnn(model, test_loader, device)
    elif model_name == "snn":
        train_metrics = train_snn(model, train_loader, optimizer, device, time_steps, encoding=encoding)
        test_metrics = evaluate_snn(model, test_loader, device, time_steps, encoding=encoding)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    epoch_time = time.perf_counter() - t0
    return train_metrics, test_metrics, epoch_time


def _final_evaluation(
    model_name: str,
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    time_steps: int,
    encoding: str,
) -> dict[str, object]:
    if model_name == "ann":
        return evaluate_ann(model, test_loader, device, collect_predictions=True)
    if model_name == "cnn":
        return evaluate_cnn(model, test_loader, device, collect_predictions=True)
    if model_name == "snn":
        return evaluate_snn(
            model,
            test_loader,
            device,
            time_steps,
            encoding=encoding,
            collect_predictions=True,
            collect_raster=True,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _train_single_seed(
    config: Config,
    spec,
    model_names: Iterable[str],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    set_seed(seed)
    train_loader, test_loader, _ = create_dataloaders(
        spec.name,
        batch_size=config.batch_size,
        train_subset_size=config.train_subset_size,
        test_subset_size=config.test_subset_size,
        seed=seed,
    )

    models: dict[str, torch.nn.Module] = {
        name: _build_model(name, spec, config).to(device) for name in model_names
    }
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for name, model in models.items()
    }

    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        epoch_line_parts: list[str] = [f"seed={seed} epoch={epoch:02d}"]
        for name, model in models.items():
            train_metrics, test_metrics, epoch_time = _run_epoch(
                name, model, optimizers[name], train_loader, test_loader,
                device, config.time_steps, spec.encoding,
            )
            history.append({
                "seed": seed,
                "epoch": epoch,
                "model": name.upper(),
                "split": "train",
                "loss": train_metrics["loss"],
                "accuracy": train_metrics["accuracy"],
                "hidden_spike_rate": train_metrics.get("hidden_spike_rate"),
                "train_time_s": epoch_time,
            })
            history.append({
                "seed": seed,
                "epoch": epoch,
                "model": name.upper(),
                "split": "test",
                "loss": test_metrics["loss"],
                "accuracy": test_metrics["accuracy"],
                "hidden_spike_rate": test_metrics.get("hidden_spike_rate"),
                "sparsity": test_metrics.get("sparsity"),
                "train_time_s": 0.0,
            })
            epoch_line_parts.append(
                f"{name.upper()} test={test_metrics['accuracy']:.4f}/{test_metrics['loss']:.3f} t={epoch_time:.2f}s"
            )
        print(" | ".join(epoch_line_parts))

    final_metrics: dict[str, dict[str, object]] = {}
    for name, model in models.items():
        result = _final_evaluation(name, model, test_loader, device, config.time_steps, spec.encoding)
        result["params"] = count_parameters(model)
        if name == "snn":
            spike_rate = float(result.get("hidden_spike_rate", 0.0))
            result["energy_proxy"] = energy_proxy_snn(
                result["params"]["params"], spike_rate, config.time_steps
            )
        else:
            result["energy_proxy"] = energy_proxy_ann(result["params"]["params"])
        result["epoch_to_target"] = epoch_to_target_accuracy(history, name.upper(), config.target_accuracy)
        result["time_to_target_s"] = time_to_target_accuracy(history, name.upper(), config.target_accuracy)
        result["total_train_time_s"] = sum(
            float(row["train_time_s"]) for row in history
            if row["model"] == name.upper() and row["split"] == "train"
        )
        final_metrics[name] = result

    return {
        "seed": seed,
        "history": history,
        "final": final_metrics,
        "models": models,
        "test_loader": test_loader,
    }


def _aggregate_final_metrics(per_seed_finals: list[dict[str, dict[str, object]]]) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate mean / std across seeds for each (model, scalar metric)."""
    model_names = sorted({name for run in per_seed_finals for name in run})
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for name in model_names:
        per_metric: dict[str, SeedAggregate] = {}
        for run in per_seed_finals:
            entry = run.get(name)
            if entry is None:
                continue
            scalar_items: dict[str, float] = {
                "accuracy": float(entry["accuracy"]),
                "loss": float(entry["loss"]),
                "energy_proxy": float(entry["energy_proxy"]),
                "total_train_time_s": float(entry["total_train_time_s"]),
            }
            if entry.get("hidden_spike_rate") is not None:
                scalar_items["hidden_spike_rate"] = float(entry["hidden_spike_rate"])
            if entry.get("sparsity") is not None:
                scalar_items["sparsity"] = float(entry["sparsity"])
            cls_metrics = classification_metrics(entry["predictions"], entry["labels"])
            scalar_items["f1_macro"] = float(cls_metrics["f1_macro"])
            scalar_items["precision_macro"] = float(cls_metrics["precision_macro"])
            scalar_items["recall_macro"] = float(cls_metrics["recall_macro"])
            if entry.get("epoch_to_target") is not None:
                scalar_items["epoch_to_target"] = float(entry["epoch_to_target"])
            if entry.get("time_to_target_s") is not None:
                scalar_items["time_to_target_s"] = float(entry["time_to_target_s"])

            for metric_name, value in scalar_items.items():
                per_metric.setdefault(metric_name, SeedAggregate()).add(value)

        aggregated[name] = {metric: agg.as_dict() for metric, agg in per_metric.items()}
    return aggregated


def _write_comparison_md(
    summary: dict[str, object],
    aggregated: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    lines = [f"# Porownanie modeli na {summary['dataset_display']}", ""]
    lines.append(f"Liczba seedow: {summary['n_seeds']}, epoki: {summary['epochs']}, batch: {summary['batch_size']}.")
    lines.append("")
    lines.append("| Model | Acc (mean +- std) | F1 macro | Loss | Train time [s] | Params | Energy proxy |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in aggregated.items():
        acc = metrics.get("accuracy", {})
        f1 = metrics.get("f1_macro", {})
        loss = metrics.get("loss", {})
        train_time = metrics.get("total_train_time_s", {})
        params = summary["params"].get(name, {}).get("params", "?")
        energy = metrics.get("energy_proxy", {})
        lines.append(
            f"| {name.upper()} "
            f"| {acc.get('mean', float('nan')):.4f} +- {acc.get('std', 0.0):.4f} "
            f"| {f1.get('mean', float('nan')):.4f} "
            f"| {loss.get('mean', float('nan')):.4f} "
            f"| {train_time.get('mean', float('nan')):.2f} "
            f"| {params} "
            f"| {energy.get('mean', float('nan')):.2e} |"
        )
    lines.append("")
    lines.append("Pelne metryki w `summary.json`, historia w `history.csv`.")
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _save_classification_reports(
    per_seed_finals: list[dict[str, dict[str, object]]],
    output_dir: Path,
) -> None:
    """Save sklearn classification_report (text) for the first seed of each model."""
    if not per_seed_finals:
        return
    first = per_seed_finals[0]
    for name, entry in first.items():
        cls = classification_metrics(entry["predictions"], entry["labels"])
        report_path = output_dir / f"classification_report_{name.upper()}.txt"
        report_path.write_text(cls["report"], encoding="utf-8")


def _save_confusion_matrices(
    per_seed_finals: list[dict[str, dict[str, object]]],
    dataset_display: str,
    output_dir: Path,
) -> None:
    if not per_seed_finals:
        return
    first = per_seed_finals[0]
    for name, entry in first.items():
        cls = classification_metrics(entry["predictions"], entry["labels"])
        plot_confusion_matrix(
            cls["confusion_matrix"],
            output_dir / f"confusion_matrix_{name.upper()}.png",
            title=f"{name.upper()} on {dataset_display}",
        )


def _save_raster(
    per_seed_finals: list[dict[str, dict[str, object]]],
    dataset_display: str,
    output_dir: Path,
) -> None:
    if not per_seed_finals:
        return
    first = per_seed_finals[0]
    snn = first.get("snn")
    if snn is None or "raster_hidden" not in snn:
        return
    plot_raster(
        snn["raster_hidden"],
        snn["raster_output"],
        snn["raster_labels"],
        snn["raster_inputs"],
        output_dir / "raster_SNN.png",
        title=f"SNN spike raster on {dataset_display}",
    )


def _run_benchmarks(
    models: dict[str, torch.nn.Module],
    spec,
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> None:
    print("  -> running time benchmarks ...")
    bench_summary: dict[str, object] = {}

    throughput_rows: list[dict[str, object]] = []
    for name, model in models.items():
        for mode in ("inference", "train"):
            result = benchmark_throughput(
                model, spec.image_shape, spec.num_classes, device,
                mode=mode, batch_size=config.batch_size,
                time_steps=config.time_steps, encoding=spec.encoding,
                repeats=config.benchmark_repeats, warmup_batches=config.benchmark_warmup_batches,
            )
            throughput_rows.append({"model": name.upper(), "mode": mode, **result.as_dict()})
    bench_summary["throughput"] = throughput_rows

    latency_rows: list[dict[str, object]] = []
    for name, model in models.items():
        sub = benchmark_latency_vs_batch(
            model, spec.image_shape, spec.num_classes, device,
            batch_sizes=config.benchmark_batch_sizes,
            time_steps=config.time_steps, encoding=spec.encoding,
            repeats=config.benchmark_repeats, warmup_batches=config.benchmark_warmup_batches,
        )
        for row in sub:
            row["model"] = name.upper()
            latency_rows.append(row)
    bench_summary["latency_vs_batch"] = latency_rows
    plot_latency_vs_batch(latency_rows, output_dir / "latency_vs_batch.png", title=f"Latency vs batch on {spec.display_name}")

    snn_model = models.get("snn")
    if snn_model is not None:
        time_step_rows = benchmark_time_per_timestep(
            snn_model, spec.image_shape, device,
            batch_size=config.batch_size,
            time_steps_list=(1, 5, 10, 20, 40),
            encoding=spec.encoding,
            repeats=config.benchmark_repeats, warmup_batches=config.benchmark_warmup_batches,
        )
        bench_summary["snn_time_per_timestep"] = time_step_rows

    profile_rows: dict[str, dict[str, object]] = {}
    for name, model in models.items():
        sub = profile_training_step(
            model, spec.image_shape, spec.num_classes, device,
            batch_size=config.batch_size,
            time_steps=config.time_steps, encoding=spec.encoding,
            repeats=config.benchmark_repeats, warmup_batches=config.benchmark_warmup_batches,
        )
        profile_rows[name.upper()] = {k: v.as_dict() for k, v in sub.items()}
    bench_summary["training_step_profile_ms"] = profile_rows

    save_json(bench_summary, output_dir / "benchmark_time.json")


def _run_noise_sweep(
    models: dict[str, torch.nn.Module],
    test_loader,
    spec,
    config: Config,
    device: torch.device,
    output_dir: Path,
    seed: int,
) -> None:
    print("  -> running noise robustness sweep ...")
    rows = noise_robustness_sweep(
        {name.upper(): model for name, model in models.items()},
        test_loader,
        device,
        sigmas=config.noise_sigmas,
        time_steps=config.time_steps,
        encoding=spec.encoding,
        seed=seed,
    )
    save_json(rows, output_dir / "noise_robustness.json")
    plot_noise_robustness(rows, output_dir / "noise_robustness.png", title=f"Noise robustness on {spec.display_name}")


def run_for_dataset(dataset_name: str, model_names: list[str], config: Config, device: torch.device, args) -> None:
    spec = get_dataset_spec(dataset_name)
    output_dir = ensure_results_dir(spec.name)
    print(f"\n=== Dataset: {spec.display_name} ({spec.name}) ===")

    per_seed_runs: list[dict[str, object]] = []
    combined_history: list[dict[str, object]] = []
    pareto_rows: list[dict[str, object]] = []

    for seed in config.seeds:
        run = _train_single_seed(config, spec, model_names, seed, device)
        per_seed_runs.append(run)
        combined_history.extend(run["history"])

    per_seed_finals = [run["final"] for run in per_seed_runs]

    for run in per_seed_runs:
        for name, entry in run["final"].items():
            pareto_rows.append({
                "model": name.upper(),
                "seed": run["seed"],
                "train_time_s": float(entry["total_train_time_s"]),
                "accuracy": float(entry["accuracy"]),
            })

    aggregated = _aggregate_final_metrics(per_seed_finals)

    params_summary = {
        name: per_seed_finals[0][name]["params"] for name in per_seed_finals[0]
    }

    summary = {
        "dataset": spec.name,
        "dataset_display": spec.display_name,
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "n_seeds": len(config.seeds),
        "seeds": list(config.seeds),
        "time_steps": config.time_steps,
        "encoding": spec.encoding,
        "target_accuracy": config.target_accuracy,
        "params": params_summary,
        "metrics_aggregated": aggregated,
    }

    save_json(summary, output_dir / "summary.json")
    save_history_csv(combined_history, output_dir / "history.csv")
    _write_comparison_md(summary, aggregated, output_dir)
    plot_training_curves(combined_history, output_dir, title_suffix=f"on {spec.display_name}")
    plot_pareto_time_accuracy(pareto_rows, output_dir / "pareto.png", title=f"Pareto on {spec.display_name}")
    _save_classification_reports(per_seed_finals, output_dir)
    _save_confusion_matrices(per_seed_finals, spec.display_name, output_dir)
    _save_raster(per_seed_finals, spec.display_name, output_dir)

    last_run = per_seed_runs[-1]
    models_last = last_run["models"]
    test_loader_last = last_run["test_loader"]
    if not args.skip_benchmarks:
        _run_benchmarks(models_last, spec, config, device, output_dir)
    if not args.skip_noise:
        _run_noise_sweep(models_last, test_loader_last, spec, config, device, output_dir, seed=last_run["seed"])

    print(f"  -> wyniki: {output_dir.resolve()}")


def main() -> None:
    args = _parse_args()
    config = _resolve_config(args)
    device = get_device()
    datasets_to_run = _resolve_datasets(args)
    print(
        f"Config: epochs={config.epochs} seeds={list(config.seeds)} batch={config.batch_size} "
        f"time_steps={config.time_steps} hidden={config.hidden_size} device={device}"
    )
    print(f"Models: {args.models}")
    print(f"Datasets: {datasets_to_run}")

    for dataset_name in datasets_to_run:
        run_for_dataset(dataset_name, args.models, config, device, args)


if __name__ == "__main__":
    main()
