# Wnioski (do uzupelnienia po peelnym uruchomieniu eksperymentow)

Sekcja "Wnioski, ktore warto opisac" z [plan.md](../plan.md). Plik powstal jako szablon -- liczby wstaw po odpaleniu `uv run main.py --datasets mnist fashion-mnist kmnist seq-mnist`.

## 1. Kiedy SNN ma przewage nad klasycznymi sieciami?

- Porownaj `accuracy` i `f1_macro` ANN vs SNN per dataset (`results/<dataset>/summary.json`).
- Sprawdz roznice na **Sequential MNIST** (encoding="sequential") -- naturalne srodowisko dla SNN.
- Porownaj `energy_proxy` (SNN korzysta z `spike_rate * params * time_steps`, ANN z `2 * params`). Im nizszy `spike_rate`, tym mniej operacji.

## 2. Jakie sa trudnosci w trenowaniu SNN?

- Surrogate gradient (`fast_sigmoid`) -- inny niz zwykla pochodna; wybor `beta` istotny.
- Dluzszy czas treningu -- zobacz `total_train_time_s` w `summary.json` i benchmark `throughput_train`.
- Wieksza wariancja miedzy seedami -- `std` w `metrics_aggregated.accuracy`.

## 3. Czy SNN sa sensowne dla tego typu danych?

- Frame-based (MNIST / Fashion-MNIST / KMNIST): porownaj czy SNN nadal wygrywa, czy roznica zanika.
- Czasowe (Sequential MNIST): zobacz, czy przewaga SNN rosnie.
- Robustnosc na szum: porownaj krzywe accuracy(sigma) w `noise_robustness.json` -- czy SNN spada wolniej?

## 4. Czy zyski energetyczne rekompensuja zlozonosc implementacji?

- Energy proxy ANN: ~2 * params operacji.
- Energy proxy SNN: `spike_rate * params * time_steps`. Przy `spike_rate ~= 0.15` i `time_steps = 20` to ~3 * params. Czyli **przewagi energetycznej nie ma na konwencjonalnym GPU/CPU**; ma sens tylko na sprzecie zdarzeniowym (Loihi, SpiNNaker), gdzie liczy sie tylko aktywny spike.
- Inference latency: `latency_vs_batch.png` -- SNN bedzie wolniejszy ze wzgledu na petle czasowa.

## 5. Tabela podsumowujaca (uzupelnic recznie)

| Dataset | Najlepszy model | Acc / F1 | Train time | Wnioski |
| --- | --- | --- | --- | --- |
| MNIST | ? | ? | ? | ? |
| Fashion-MNIST | ? | ? | ? | ? |
| KMNIST | ? | ? | ? | ? |
| Sequential MNIST | ? | ? | ? | ? |
