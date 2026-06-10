# Raport z postepu: Spiking Neural Network vs ANN

**Repozytorium:** [github.com/dominikgofer/spiking-neural-network-comparison](https://github.com/dominikgofer/spiking-neural-network-comparison)
**Autor:** Dominik Godek
**Data raportu:** 10 czerwca 2026

---

## 1. Cel projektu

Porownanie klasycznej sieci neuronowej (ANN) i Spiking Neural Network (SNN) na tym samym zadaniu klasyfikacji, aby sprawdzic w jakich warunkach SNN stanowia sensowna alternatywe dla klasycznych modeli. Projekt koncentruje sie na dokladnosci, czasie trenowania, stabilnosci uczenia, zlozonosci obliczeniowej i potencjale zastosowan niskoenergetycznych.

---

## 2. Obecny stan projektu

### 2.1 Srodowisko

- Python 3.12, menedzer srodowiska `uv`
- Zaleznosci: `torch>=2.4`, `torchvision>=0.19`, `snntorch>=0.9`, `matplotlib>=3.9`, `numpy>=2.0`, `scikit-learn>=1.5`
- Lockfile (`uv.lock`) wygenerowany, srodowisko odtwarzalne przez `uv run main.py`

### 2.2 Dane

- Zbior: **MNIST** (28x28 obrazy cyfr 0-9)
- Podzbiory: 12 000 probek treningowych, 2 000 testowych (losowe, deterministyczne, seed=7)

### 2.3 Zaimplementowane modele

**ANN (baseline)** -- prosty MLP:

```
Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)
```

Trening: Adam (lr=1e-3), CrossEntropy loss.

**SNN** -- SpikingMLP z neuronami LIF:

```
Linear(784, 128) -> Leaky IF (beta=0.95) -> Linear(128, 10) -> Leaky IF (beta=0.95)
```

Wejscie: rate coding (20 krokow czasowych). Surrogate gradient: fast_sigmoid (slope=25). Logity = suma spikow wyjsciowych po czasie.

### 2.4 Pipeline eksperymentu

Plik `main.py` trenuje oba modele w petli epok na tych samych danych, zbiera historie (loss, accuracy, spike rate), a po zakonczeniu zapisuje:
- `results/summary.json` -- metryki koncowe
- `results/history.csv` -- historia uczenia
- `results/comparison.md` -- tabela porownawcza
- `results/comparison_curves.png` -- wykresy accuracy i loss

---

## 3. Wyniki eksperymentu

### 3.1 Przebieg treningu (3 epoki)

| Epoka | ANN test acc | ANN test loss | SNN test acc | SNN test loss | SNN spike rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 90.20% | 0.3687 | 91.85% | 0.2652 | 0.1475 |
| 2 | 91.65% | 0.2919 | 94.15% | 0.2013 | 0.1512 |
| 3 | 91.65% | 0.2675 | 95.00% | 0.1736 | 0.1579 |

### 3.2 Wynik koncowy

| Model | Test accuracy | Test loss |
| --- | ---: | ---: |
| ANN | 91.65% | 0.2675 |
| **SNN** | **94.80%** | **0.1820** |

- SNN przewyzsza ANN o **+3.15 pp** dokladnosci
- SNN osiaga nizszy loss testowy (0.1820 vs 0.2675)
- Srednia aktywnosc spikow w warstwie ukrytej: ~15.8%

### 3.3 Obserwacje

- SNN juz po 1. epoce (91.85%) przewyzsza koncowy wynik ANN (91.65%)
- ANN plateau'uje na 91.65% w epokach 2-3, co moze sugerowac potrzebe wiecej epok lub bardziej rozbudowanej architektury
- SNN uczy sie stabilnie i nie wykazuje oznak overfittingu (test acc rosnie monotoniczne)

---

## 4. Realizacja planu -- checklist

Plan z `plan.md` definiuje 7 krokow implementacji:

| # | Krok | Status |
| --- | --- | --- |
| 1 | Przygotowanie srodowiska | ZROBIONE |
| 2 | Wybor danych | ZROBIONE (MNIST) |
| 3 | Model ANN jako baseline | ZROBIONE |
| 4 | Przygotowanie danych dla SNN (rate coding) | ZROBIONE |
| 5 | Implementacja SNN (LIF, surrogate gradients) | ZROBIONE |
| 6 | Trening i ewaluacja | ZROBIONE (minimalnie -- 3 epoki) |
| 7 | Wizualizacja wynikow | CZESCIOWO (wykresy tak, raster plot -- nie) |

---

## 5. Czego brakuje

### 5.1 Braki krytyczne

1. **Za malo epok** -- 3 epoki to za malo dla sprawiedliwego porownania. ANN moglby nadal sie poprawiac przy 10-15 epokach. Obecne wyniki moga faworyzowac SNN.
2. **Raster plot impulsow** -- plan wyraznie wymienia wizualizacje aktywnosci spikow. Brak implementacji.
3. **Pomiar czasu trenowania** -- plan wymienia czas jako metryke porownawcza. Nie jest mierzony.
4. **Analiza liczby parametrow** -- plan wymienia zlozonosc obliczeniowa. Nie jest raportowana (oba modele maja porownywalna liczbe parametrow, ale to powinno byc explicite policzone i wykazane).
5. **Wnioski koncowe** -- plan konczy sie sekcja "wnioski, ktore warto opisac" (kiedy SNN maja przewage, trudnosci treningu, sensownosc dla danego typu danych, bilans energetyczny vs zlozonosc). Brak takiego dokumentu.

