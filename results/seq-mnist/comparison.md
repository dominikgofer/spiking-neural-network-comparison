# Porownanie modeli na Sequential MNIST

Liczba seedow: 3, epoki: 15, batch: 128.

| Model | Acc (mean +- std) | F1 macro | Loss | Train time [s] | Params | Energy proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANN | 0.9538 +- 0.0029 | 0.9529 | 0.1544 | 13.18 | 101770 | 2.04e+05 |
| CNN | 0.9775 +- 0.0014 | 0.9771 | 0.0758 | 20.02 | 206922 | 4.14e+05 |
| SNN | 0.7692 +- 0.0095 | 0.7603 | 0.6859 | 48.14 | 5002 | 7.82e+03 |

Pelne metryki w `summary.json`, historia w `history.csv`.