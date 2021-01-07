
# Avancement

- 2020 / 12 / 18 : presque exactement les meme resultats de scikit, sauf avec
 beaucoup de donnees 

- pretri ok mais super lent en fait a cause du masque alors que c'est cense etre plus
 rapide ?
 
 

# TODO generaux

- se contenter du sort lent et implementer l'aggregation

- comprendre la construction de l'arbre a partir de node_id = tree_add_node et printer

- ecrire code en reprenant onelearn qui affiche l'arbre pour debbuger

- comprendre le rc=0 ou -1 qui resize aussi le tree ?!?

- coder la forest et kes echantillons bootstrap


- Verifier que c'est OK aussi avec un sample_weight et en multi-classe, multi-label

- Ecrire directement des unittests

- Gerer le random state : a priori OK

- **C'est l'option fastmath=True dans @njit qui fait que les resultats avec scikit diff
èrent !**

- Avec beaucoup beaucoup de points on a pas exactement le même resultat mais ce n'est
 pas loin... ca ira je pense 

- faut que je comprenne mieux ce qui se passe dans la splitting strategy
 
- faire marcher le tri avec la pre-sort strategy

- On a pas exactement le meme nombre de noeuds compare a scikit learn, on en a
 toujours un peu plus, c'est bizarre...

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

