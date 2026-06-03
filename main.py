from __future__ import annotations

import torch


def main() -> None:
    from ann import ANNClassifier, evaluate_ann, train_ann
    from snn import SpikingMLP, evaluate_snn, train_snn
    from utils import Config, create_dataloaders, ensure_results_dir, get_device, save_results, set_seed

    config = Config()
    set_seed(config.seed)
    device = get_device()
    results_dir = ensure_results_dir()

    train_loader, test_loader = create_dataloaders(config)

    ann_model = ANNClassifier(hidden_size=config.hidden_size).to(device)
    ann_optimizer = torch.optim.Adam(ann_model.parameters(), lr=config.learning_rate)

    snn_model = SpikingMLP(hidden_size=config.hidden_size, beta=config.beta).to(device)
    snn_optimizer = torch.optim.Adam(snn_model.parameters(), lr=config.learning_rate)

    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        ann_train_loss, ann_train_acc = train_ann(ann_model, train_loader, ann_optimizer, device)
        ann_test_loss, ann_test_acc = evaluate_ann(ann_model, test_loader, device)

        snn_train_loss, snn_train_acc, snn_train_spike_rate = train_snn(
            snn_model, train_loader, snn_optimizer, device, config.time_steps
        )
        snn_test_loss, snn_test_acc, snn_test_spike_rate = evaluate_snn(
            snn_model, test_loader, device, config.time_steps
        )

        history.extend(
            [
                {"epoch": epoch, "model": "ANN", "split": "train", "loss": ann_train_loss, "accuracy": ann_train_acc},
                {"epoch": epoch, "model": "ANN", "split": "test", "loss": ann_test_loss, "accuracy": ann_test_acc},
                {"epoch": epoch, "model": "SNN", "split": "train", "loss": snn_train_loss, "accuracy": snn_train_acc, "hidden_spike_rate": snn_train_spike_rate},
                {"epoch": epoch, "model": "SNN", "split": "test", "loss": snn_test_loss, "accuracy": snn_test_acc, "hidden_spike_rate": snn_test_spike_rate},
            ]
        )

        print(
            f"Epoch {epoch:02d} | "
            f"ANN test acc: {ann_test_acc:.4f} loss: {ann_test_loss:.4f} | "
            f"SNN test acc: {snn_test_acc:.4f} loss: {snn_test_loss:.4f}"
        )

    final_ann_test = evaluate_ann(ann_model, test_loader, device)
    final_snn_test = evaluate_snn(snn_model, test_loader, device, config.time_steps)

    summary = {
        "dataset": "MNIST",
        "config": config.__dict__,
        "device": str(device),
        "ann_test_loss": final_ann_test[0],
        "ann_test_accuracy": final_ann_test[1],
        "snn_test_loss": final_snn_test[0],
        "snn_test_accuracy": final_snn_test[1],
        "snn_test_hidden_spike_rate": final_snn_test[2],
    }

    save_results(summary, history, results_dir)

    print(f"Wyniki zapisano do: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
