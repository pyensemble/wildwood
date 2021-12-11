
# API

(preprocessing_api)=
## Preprocessing

The ``wildwood.preprocessing`` module contains the ``Encoder`` class. The Encoder 
performs the transformation of an input ``pandas.DataFrame`` or ``numpy.ndarray`` 
into a ``wildwood.FeaturesBitArray`` class.

(encoder_api)=
### Encoder

```{eval-rst}

.. currentmodule:: wildwood.preprocessing

.. autoclass:: Encoder
   :members: fit, transform
```

(random_forest_api)=
## Random Forest algorithms

``WildWood`` exposes the two classes ``ForestClassifier`` for multi-class 
classification and ``ForestRegressor`` for regression.

(ForestClassifier)=
### Multi-class classification with the `ForestClassifier` class

```{eval-rst}

.. currentmodule:: wildwood 

.. autoclass:: ForestClassifier
   :members: fit, predict_proba, predict, apply, score, get_params, set_params
```


(ForestRegressor)=
### Regression with the `ForestRegressor` class

```{eval-rst}

.. currentmodule:: wildwood 

.. autoclass:: ForestRegressor
```
