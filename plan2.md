# Plan 2: Pogłębione porównanie ANN vs CNN vs SNN

## Cel etapu
Rozszerzyć prototyp z planu 1 o trzeci model (CNN), dodatkowe zbiory danych, bogaty zestaw metryk, benchmarki czasowe i test robustności na szum. Wynikiem ma być powtarzalny pipeline, który pozwala jednym poleceniem CLI uruchomić pełny eksperyment na wielu datasetach i zapisać porównywalne wyniki.

## Kontekst
Plan 1 zrealizował podstawowy prototyp: ANN (MLP) vs SNN na MNIST, 3 epoki, bez pomiaru czasu i bez głębszych metryk. Raport postępu (`results/raport_postepu.md`) identyfikuje braki, które ten etap ma domknąć.

## Zakres projektu
1. Domknięcie braków z planu 1.
2. Rozszerzenie zestawu metryk.
3. Dodanie nowych zbiorów danych.
4. Dodanie trzeciego modelu (CNN), benchmarków czasowych i testu robustności.
5. Refaktor struktury projektu i pipeline CLI.
6. Finalny raport z wnioskami (po uruchomieniu eksperymentów).

## Kroki implementacji

### 1. Domknięcie braków z planu 1
- Zwiększyć liczbę epok z 3 do **15**.
- Dodać **pomiar czasu trenowania** per epoka i łącznie.
- Raportować **liczbę parametrów** i rozmiar modeli w MB.
- Zaimplementować **raster plot impulsów** (hidden + output spikes dla wybranych próbek).
- Wykorzystać `scikit-learn`: `confusion_matrix`, `classification_report`, F1 / precision / recall (macro + per-class).
- Utworzyć plik wniosków `results/conclusions.md`.

### 2. Więcej metryk
Wzbogacić `summary.json` i `comparison.md` o:
- F1 / precision / recall (macro + per-class).
- **Inference latency** -- czas predykcji na batch i na pojedynczą próbkę.
- **Energy proxy**:
  - ANN/CNN: szacunkowe FLOPs (~2 * params per forward pass).
  - SNN: średnia liczba spikeów * parametry * time_steps.
- **Sparsity** -- procent zerowych aktywacji warstwy ukrytej (ANN/CNN po ReLU vs SNN spike rate).
- **Convergence rate** -- numer epoki i wall-clock do osiągnięcia 90% accuracy testowej.
- **Stabilność** -- powtórzenie eksperymentu na **3 seedach** (7, 13, 42), średnia +/- odchylenie standardowe.

### 3. Więcej zbiorów danych
- **Fashion-MNIST** -- 28x28, 10 klas, trudniejszy semantycznie.
- **KMNIST** -- znaki japońskie, test generalizacji architektury.
- **Sequential MNIST** -- obraz podawany wiersz po wierszu (28 kroków czasowych), naturalne środowisko dla SNN.
- (opcjonalnie) **N-MNIST** przez bibliotekę `tonic` -- prawdziwy dataset event-based.

Wybór datasetu jako parametr CLI: `--dataset {mnist,fashion-mnist,kmnist,seq-mnist}`.

### 4. Dodatkowe rozszerzenia

#### 4.1 CNN jako trzeci baseline
- Mały CNN: Conv2d(1,16,3) -> ReLU -> MaxPool -> Conv2d(16,32,3) -> ReLU -> MaxPool -> FC(128) -> FC(10).
- Pozwoli pokazać, czy przewaga SNN utrzymuje się wobec mocniejszego modelu klasycznego.

#### 4.2 Test robustności na szum
- Gaussian noise sigma ∈ {0.0, 0.1, 0.2, 0.3} na zbiorze testowym.
- Porównanie spadku accuracy ANN vs CNN vs SNN.
- Hipoteza: SNN dzięki kodowaniu impulsowemu jest bardziej odporny.

#### 4.3 Benchmarki czasowe
Dedykowany moduł `experiments/benchmark_time.py`:
- **Throughput training** -- samples/sec (forward + backward), batch_size=128, N=5 powtórzeń.
- **Throughput inference** -- samples/sec (forward-only, no_grad).
- **Latency vs batch_size** -- batch_size ∈ {1, 8, 32, 128, 512}, wykres latency per batch i per sample.
- **Wall-clock do target accuracy** -- sekundy do osiągnięcia 90% test acc.
- **Profil per komponent** -- data loading vs forward vs backward vs optimizer step.
- **Time-per-timestep SNN** -- ile czasu zajmuje jeden krok pętli czasowej.
- **Pareto: czas vs accuracy** -- wykres scatter (czas treningu, accuracy końcowa).
- Warmup: 2 batche pomijane. Wszystkie pomiary na tym samym sprzęcie/seedzie.

### 5. Struktura projektu i pipeline
Refaktor z płaskiej struktury na pakietową:
```text
projekt/
  models/         -- ann.py, cnn.py, snn.py
  datasets/       -- loaders.py (rejestr datasetów)
  experiments/    -- benchmark_time.py, noise_robustness.py
  results/
    <dataset>/    -- summary.json, history.csv, wykresy, raporty per dataset
  main.py         -- CLI orkiestrator (argparse)
  utils.py        -- Config, seedy, metryki, ploty, agregacja
```

CLI z `argparse`:
```bash
uv run main.py --datasets mnist fashion-mnist kmnist seq-mnist --models ann cnn snn --epochs 15 --seeds 7 13 42
```

### 6. Finalny raport (po uruchomieniu eksperymentów)
- Tabele zbiorcze w `results/conclusions.md`: jedna tabela na dataset, kolumny ANN/CNN/SNN, metryki: accuracy, F1, czas treningu, czas inferencji, rozmiar modelu, sparsity, robustność na szum.
- Wnioski:
  - Kiedy SNN wygrywa (rzadkie aktywacje, dane czasowe, niskoenergetyczny hardware).
  - Kiedy klasyczne sieci wygrywają (frame-based dane, krótka latencja inferencji, prosty trening).
  - Trudności treningu SNN (surrogate gradient, wybór beta, czułość na kodowanie wejścia).
  - Czy zyski energetyczne rekompensują złożoność implementacji.

## Co powstaje w `results/<dataset>/`
- `summary.json` -- konfiguracja + metryki zagregowane po seedach (mean / std / n).
- `history.csv` -- pełna historia uczenia per seed, epoka, model, split.
- `comparison.md` -- czytelna tabela porównawcza.
- `comparison_curves.png` -- wykresy accuracy / loss.
- `pareto.png` -- scatter (czas treningu, accuracy).
- `confusion_matrix_<MODEL>.png` -- macierz pomyłek.
- `classification_report_<MODEL>.txt` -- raport sklearn (precision / recall / F1 per klasa).
- `raster_SNN.png` -- raster plot aktywności spikeów.
- `benchmark_time.json` -- throughput, latency, profil, time-per-timestep.
- `latency_vs_batch.png` -- wykres skalowania latency.
- `noise_robustness.json` + `noise_robustness.png` -- robustność na szum.
