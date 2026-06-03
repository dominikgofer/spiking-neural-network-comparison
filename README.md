# Projekt SNN vs ANN na MNIST

Ten projekt porownuje klasyczna siec neuronowa i Spiking Neural Network na zbiorze MNIST. Prototyp trenuje oba modele na tym samym podzbiorze danych, zapisuje metryki i wykresy porownawcze do folderu `results/`.

## Uruchomienie

Najprostszy sposob to uruchomienie przez `uv`:

```bash
uv run main.py
```

Przy pierwszym uruchomieniu `uv` utworzy lokalne srodowisko `.venv` i pobierze zaleznosci z `pyproject.toml`.

## Co powstaje w `results/`

Po zakonczeniu treningu zapisuje sie:

- `results/summary.json` - podsumowanie eksperymentu MNIST z metrykami ANN i SNN,
- `results/history.csv` - historia uczenia dla obu modeli,
- `results/comparison.md` - tekstowe porownanie wynikow na MNIST,
- `results/comparison_curves.png` - wykresy accuracy i loss dla przypadku MNIST.

## Zaleznosci

Zaleznosci sa zdefiniowane w `pyproject.toml` i obejmuja m.in.:

- `torch`
- `torchvision`
- `snntorch`
- `matplotlib`
