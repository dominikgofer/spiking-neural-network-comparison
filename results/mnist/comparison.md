# Porownanie modeli na MNIST

Liczba seedow: 3, epoki: 15, batch: 128.

| Model | Acc (mean +- std) | F1 macro | Loss | Train time [s] | Params | Energy proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANN | 0.9555 +- 0.0057 | 0.9545 | 0.1537 | 16.61 | 101770 | 2.04e+05 |
| CNN | 0.9815 +- 0.0011 | 0.9814 | 0.0567 | 32.11 | 206922 | 4.14e+05 |
| SNN | 0.9635 +- 0.0041 | 0.9627 | 0.1402 | 77.68 | 101770 | 3.52e+05 |

Pelne metryki w `summary.json`, historia w `history.csv`.