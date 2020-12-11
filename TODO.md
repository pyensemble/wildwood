

# Optimisation


## 2020 / 12 / 10

                  timings
classifier        sk_tree       tree
dataset task
circles fit      3.265248  11.521672
        predict  0.074003   0.074817
moons   fit      2.617778  10.505634
        predict  0.064629   0.072816

```bash

python -m cProfile -o main_tree.prof main_tree.py
snakeviz main_tree.prof
```
le gros du temps est dans best_splitter_node_split

- virer les jitclass autant que possible -> faire un test a part, accès à des
 tableaux via des attributes de classes en permanence -> lent en numba, pert la
  vectorisation ?

# Unit tests

- Tester que l'arbre marche bien avec des jeux de données simples

# TODO generaux

- Simplifier / nettoyer le code
    - j'ai commence. Virer petit a petit les jitclass... 
- random state doit marcher et on doit parfaitement matcher scikit
    
- RandomSplitter ou un truc du genre pour la foret ?
- calculer loss dans chaque noeud (regarder onelearn) sur un validation set : faut
 avoir les indices quelque part pour la validation, regarder le code de scikit 

- redecouper splitter en sous fonctions car trop de cas
    - strategy
    - dense/sparse
    - histogram ou pas (feature categorielle ou pas)
    - approximation

- splitter sparse
- spliter mom

- Traduire la foret en numba

1. Faire marcher la foret avec aggregation avec dense splitter
