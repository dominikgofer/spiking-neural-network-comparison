# Porownanie modeli na KMNIST

Liczba seedow: 3, epoki: 15, batch: 128.

| Model | Acc (mean +- std) | F1 macro | Loss | Train time [s] | Params | Energy proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANN | 0.8080 +- 0.0144 | 0.8082 | 0.6678 | 19.26 | 101770 | 2.04e+05 |
| CNN | 0.8968 +- 0.0114 | 0.8968 | 0.4899 | 19.52 | 206922 | 4.14e+05 |
| SNN | 0.8243 +- 0.0090 | 0.8245 | 0.7235 | 58.70 | 101770 | 3.46e+05 |

Pelne metryki w `summary.json`, historia w `history.csv`.