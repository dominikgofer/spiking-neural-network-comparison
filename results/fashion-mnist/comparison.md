# Porownanie modeli na Fashion-MNIST

Liczba seedow: 3, epoki: 15, batch: 128.

| Model | Acc (mean +- std) | F1 macro | Loss | Train time [s] | Params | Energy proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANN | 0.8505 +- 0.0067 | 0.8513 | 0.4272 | 15.08 | 101770 | 2.04e+05 |
| CNN | 0.8778 +- 0.0077 | 0.8795 | 0.3416 | 24.49 | 206922 | 4.14e+05 |
| SNN | 0.8343 +- 0.0031 | 0.8318 | 0.4827 | 64.02 | 101770 | 3.83e+05 |

Pelne metryki w `summary.json`, historia w `history.csv`.