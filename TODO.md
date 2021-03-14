

- Faire marcher predict avec threshold sans bin_threshold

- Coder le calcul de la loss de validation dans init_node_context, mais pas besoin d
'utiliser tout de suite le step ? Ou est-ce qu'on met le calcul des poids d'aggregation ? Au moment de l'ajout du
 noeud ? Ou alors au moment du init_node_context (y'a une boucle deja. On a besoin de
  connaitre deux choses pour ca : y_pred du noeud et valid_indices du noeud, c'est
   tout) -> OK aussi a priori (check streamlit)

- IDEE: pour le calcul des poids d'aggregation, on est en mode "depth first", donc on
 traite toujours les enfants avant les parents. Donc quand on ajoute un noeud dans l
 'arbre, on est sur qu'on a deja ajoute ses enfants. Donc c'est a mettre dans la
  fonction add_node_tree

- Calcul des loss de validation dans les noeuds valid_loss et du poids d'aggregtion

- On peut changer le step dans re-entrainer un arbre en fait...

- Le code numba est compile pour les n_jobs threads, faudrait forcer la compilation
 avant... 

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

