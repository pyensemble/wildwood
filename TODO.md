
# Bilan 2021 / 01 / 25

- Dans les splits (le long d'une feature) on fait attention a ce que les childs aient au
 moins 1 train et 1 valid
 
- Normalement impurity est OK dans les noeuds. On ne splitte pas un noeud pur


# TODO

- Ensuite nettoyer _tree_context.py

- La loss de validation n'est pas divisee par le nombre de samples...

- Finish to clean the _grow.py module

- Faut prendre un parametre Dirichlet beaucoup plus petit que dans le cas en ligne ?!?

- dans check_forest reprendre la fonction de plot de playground_forest

- Y'a un bug quand step > 1.0 ? Bizarre ? overflow ou autre chose ?

- Il se passe quoi si on change un de ces parametres apres le fit ?

- Verifier tout ca dans streamlit

- On dirait que Le calcul des poids d'aggregation marche avec numba mais pas python
 ?!? Et ca a l'air plutôt OK ?!?

- Y'a vraiment un bug bizarre avec ssize_t et la facon dont je fais les tests
 -> mettre un flag pour dire si un noeud est une feuille ou pas... OK

- Use jit everywhere with spec and correct options

- Y'a aussi le feature bootstrap a mettre !

- Faire marcher le parametre dirichlet -> OK a priori

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

- J'ai juste utilise random_state + tree_idx pour
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

