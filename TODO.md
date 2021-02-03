
# Bilan 2021 / 01 / 25

- Dans les splits (le long d'une feature) on fait attention a ce que les childs aient au
 moins 1 train et 1 valid
 
- Normalement impurity est OK dans les noeuds. On ne splitte pas un noeud pur

- Numba remarche apparemment

# TODO

- Parametre dirichlet et predictions avec prior dirichlet par un noeud

- Ou est-ce qu'on met le calcul des poids d'aggregation ? Au moment de l'ajout du
 noeud ? Ou alors au moment du init_node_context (y'a une boucle deja. On a besoin de
  connaitre deux choses pour ca : y_pred du noeud et valid_indices du noeud, c'est tout)

- Calcul des loss de validation dans les noeuds valid_loss et du poids d'aggregtion

- On peut changer le step dans re-entrainer un arbre en fait...

- Le code numba est compile pour les n_jobs threads, faudrait forcer la compilation
 avant... 

- Il doit y avoir des divisions par zero lors de la prediction: 
/Users/stephanegaiffas/Code/wildwood/wildwood/tree.py:1114: RuntimeWarning: invalid value encountered in true_divide

- Faire en sorte que la foret marche en parallele (verifier que ca tourne bien en
 parallele) : on dirait que n_estimators marche pas ou alors que tous les arbres sont
  identiques ? : Maintenant c'est OK, j'ai juste utilise random_state + tree_idx pour
   avoir des samples differents, je ne comprends pas comment ils font dans scikit

- faire de progression TQDM sur les arbres entraines

- Coder l'aggregation

- Quand on trouve plusieurs splits avec meme gain_proxy on prefere celui qui met
 autant de valid samples des deux cotes 
 
- Ajouter des tests sur les childs avant d'accepter un split

- Checker les histoires de bitsets ? Est ce que ca peut etre interessant ici ?

- RandomSplitter et ExtremeRandomSplitter ? Faut tirer les features au hasard 

- Que de passe-t-il si on a une feature categorielle avec 

- Gestion des features categorielles : tri 

- Gestion des features categorielles : #modalites <= max_bins

- Gestion des features categorielles : #modalites >= 255

- Gestion des NAN: split vers la gauche et la droite

- Histogram binning : pour l'instant on utilise le truc de sklearn.ensemble.experimental

- ForestBinaryClassifier : properties et options 

- ERM et MOM strategy

- Gerer le random state

- Sparse features ?

# Vieux TODOs

- **C'est l'option fastmath=True dans @njit qui fait que les resultats avec scikit diff
èrent !**


# Vieux Optimisation


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

