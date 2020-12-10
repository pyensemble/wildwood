

# Optimisation


## 2020 / 12 / 10

                  timings
classifier        sk_tree       tree
dataset task
circles fit      3.265248  11.521672
        predict  0.074003   0.074817
moons   fit      2.617778  10.505634
        predict  0.064629   0.072816

- Profiler le code dans un 

```bash

python -m cProfile -o main_tree.prof main_tree.py
snakeviz main_tree.prof
```

- Reprendre le code de scikit-learn en pure Python / Numba
- Traduire la foret en numba
- J'ai commence le _splitter mais faut finir avant le _criterion
- Dense splitted

1. Faire marcher la foret avec aggregation avec dense splitter
2. Implementer la histogramme strategy
