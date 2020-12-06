
- Pour le tree : faut recoder les data structures 

```python
from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport PriorityHeap
from ._utils cimport PriorityHeapRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray
```

Et puis faut verifier qu'on peut mettre n'importe quel dtype dans un numpy array

- ils mettent les nodes dans un numpy array !!
- On peut faire un dtype et s'en servir dans numba
- Comment ils allouent / decalouent ?
- Faut bien comprendre la gestion de noeuds

-> C'est ok il suffit d'utiliser numba.from_dtype

- Reprendre le code de scikit-learn en pure Python / Numba
- Traduire la foret en numba
- J'ai commence le _splitter mais faut finir avant le _criterion
- Dense splitted

1. Faire marcher la foret avec aggregation avec dense splitter
2. Implementer la histogramme strategy
