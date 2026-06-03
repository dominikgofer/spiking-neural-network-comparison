# Plan projektu: Spiking Neural Network jako alternatywa dla klasycznych sieci

## Cel projektu
Zaprojektować i porównać klasyczną sieć neuronową (ANN) oraz Spiking Neural Network (SNN) na tym samym zadaniu, aby sprawdzić, w jakich warunkach SNN stanowią sensowną alternatywę dla klasycznych modeli.

## Proponowany temat badawczy
Porównanie ANN i SNN w klasyfikacji danych sekwencyjnych lub event-based, z naciskiem na:
- dokładność klasyfikacji,
- czas trenowania,
- stabilność uczenia,
- liczbę parametrów i złożoność obliczeniową,
- możliwość zastosowania w systemach niskoenergetycznych.

## Zakres projektu
1. Wprowadzenie teoretyczne do SNN i ANN.
2. Wybór zbioru danych, najlepiej:
   - MNIST jako prosty start,
   - dane sekwencyjne,
   - dane event-based, jeśli projekt ma być bardziej zaawansowany.
3. Implementacja modelu bazowego ANN.
4. Implementacja modelu SNN.
5. Porównanie wyników obu podejść.
6. Wizualizacja aktywności impulsów i analiza rezultatów.

## Biblioteki w Pythonie
Najbardziej praktyczny zestaw na start:
- `torch` - baza do budowy i trenowania modeli,
- `snntorch` - prosta biblioteka do SNN,
- `numpy` - operacje numeryczne,
- `matplotlib` - wykresy i wizualizacje,
- `scikit-learn` - metryki i podział danych,
- `tqdm` - pasek postępu,
- opcjonalnie `tonic` - jeśli używane będą dane event-based.

Jeśli projekt ma być prosty i edukacyjny, wystarczy:
- `torch`
- `snntorch`
- `numpy`
- `matplotlib`
- nd -v python3; command -v pip
q`scikit-learn`

## Kroki implementacji
### 1. Przygotowanie środowiska
- Utworzyć środowisko Python 3.12.
- Zainstalować wymagane biblioteki.
- Przygotować strukturę katalogów projektu.

### 2. Wybór danych
- Zacząć od MNIST, jeśli celem jest prosty prototyp.
- Jeśli projekt ma większy zakres, przejść do danych czasowych lub event-based.

### 3. Model ANN jako punkt odniesienia
- Zaimplementować prosty MLP lub mały CNN.
- Wytrenować model na tych samych danych, na których będzie testowany SNN.
- Zapisać metryki jako baseline.

### 4. Przygotowanie danych dla SNN
- Zastosować kodowanie wejścia, np. rate coding.
- Ustalić liczbę kroków czasowych symulacji.
- Przekształcić dane do formy zgodnej z pętlą czasową SNN.

### 5. Implementacja SNN
- Użyć neuronów typu LIF lub podobnych.
- Zbudować warstwy liniowe lub konwolucyjne.
- Dodać obsługę propagacji w czasie.
- Zastosować surrogate gradients do uczenia.

### 6. Trening i ewaluacja
- Wytrenować ANN i SNN w porównywalnych warunkach.
- Porównać accuracy, loss, czas trenowania i stabilność.
- Jeśli to możliwe, przeanalizować także liczbę impulsów i aktywność neuronów.

### 7. Wizualizacja wyników
- Narysować wykresy accuracy i loss.
- Pokazać raster plot impulsów.
- Porównać wyniki ANN i SNN w formie tabeli lub wykresu.

## Proponowana struktura plików
```text
projekt/
  data/
  models/
  experiments/
  plots/
  main.py
  train_ann.py
  train_snn.py
  utils.py
  plan.md
  README.md
```

## Wnioski, które warto opisać
- Kiedy SNN mają przewagę nad klasycznymi sieciami.
- Jakie są trudności w trenowaniu SNN.
- Czy SNN są sensowne dla danego typu danych.
- Czy zyski energetyczne rekompensują większą złożoność implementacji.
