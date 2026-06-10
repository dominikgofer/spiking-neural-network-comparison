# Projekt: ANN vs CNN vs SNN -- porownanie na wielu datasetach

Drugi etap projektu spiking-neural-network-comparison (patrz [plan.md](plan.md) i [results/raport_postepu.md](results/raport_postepu.md)). Pipeline trenuje trzy modele -- klasyczny MLP, mala konwolucyjna siec oraz Spiking Neural Network -- na kilku zbiorach danych (MNIST, Fashion-MNIST, KMNIST, Sequential MNIST), zbiera bogaty zestaw metryk (accuracy / F1 / sparsity / energy proxy / robustnosc na szum), wykonuje benchmarki czasowe i zapisuje wyniki w osobnych podkatalogach per dataset.

## Struktura

```
spiking-neural-network-comparison/
  main.py                    -- CLI orkiestrator
  utils.py                   -- Config, seedy, metryki, ploty, agregacja po seedach
  models/
    ann.py                   -- ANNClassifier (MLP)
    cnn.py                   -- CNNClassifier
    snn.py                   -- SpikingMLP, kodowanie rate / sequential
  datasets/
    loaders.py               -- create_dataloaders + DATASET_REGISTRY
  experiments/
    benchmark_time.py        -- throughput, latency vs batch, profil treningu, time-per-timestep
    noise_robustness.py      -- sweep po sigma szumu Gaussa
  results/
    <dataset>/               -- summary.json, history.csv, comparison.md, *.png, *.json
  data/                      -- pobierane datasety torchvision
```

## Uruchomienie

Najprostszy bieg (MNIST, 3 seedy, 15 epok, wszystkie modele i benchmarki):

```bash
uv run main.py
```

Wiele zbiorow w jednym przebiegu:

```bash
uv run main.py --datasets mnist fashion-mnist kmnist seq-mnist
```

Wybor modeli, epok, seedow:

```bash
uv run main.py --dataset fashion-mnist --models ann snn --epochs 10 --seeds 7 13
```

Szybki tryb bez ciezkich eksperymentow:

```bash
uv run main.py --dataset mnist --epochs 3 --seeds 7 --skip-benchmarks --skip-noise
```

Pelna lista flag: `uv run main.py --help`.

## Co powstaje w `results/<dataset>/`

- `summary.json` -- konfiguracja + metryki zagregowane po seedach (mean / std / n) + liczba parametrow per model.
- `history.csv` -- pelna historia uczenia: seed, epoka, model, split, loss, accuracy, hidden_spike_rate, sparsity, train_time_s.
- `comparison.md` -- czytelna tabela porownawcza (acc, F1, loss, train time, params, energy proxy).
- `comparison_curves.png` -- wykresy accuracy / loss (sredniowane po seedach).
- `pareto.png` -- scatter (train_time, final accuracy) dla obserwacji per seed.
- `confusion_matrix_<MODEL>.png` -- macierz pomylek dla pierwszego seeda.
- `classification_report_<MODEL>.txt` -- raport sklearn (precision / recall / F1 per klasa).
- `raster_SNN.png` -- raster plot aktywnosci spikow (warstwa ukryta + wyjsciowa) dla wybranych probek.
- `benchmark_time.json` -- throughput, latency vs batch, profil training step, time-per-timestep dla SNN.
- `latency_vs_batch.png` -- wykres skalowania latency z batch_size.
- `noise_robustness.json` + `noise_robustness.png` -- drop accuracy przy roznych poziomach szumu.

## Zaleznosci

Lista w `pyproject.toml`: `torch`, `torchvision`, `snntorch`, `matplotlib`, `numpy`, `scikit-learn`.
