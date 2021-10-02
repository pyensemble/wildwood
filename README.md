
[![Build Status](https://app.travis-ci.com/pyensemble/wildwood.svg?branch=master)](https://app.travis-ci.com/pyensemble/wildwood)
[![Documentation Status](https://readthedocs.org/projects/wildwood/badge/?version=latest)](https://wildwood.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wildwood)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/wildwood)
[![GitHub stars](https://img.shields.io/github/stars/pyensemble/wildwood)](https://github.com/pyensemble/wildwood/stargazers)
[![GitHub license](https://img.shields.io/github/license/pyensemble/wildwood)](https://github.com/pyensemble/wildwood/blob/master/LICENSE)


`WildWood` is a python package providing improved random forest algorithms for 
multiclass classification and regression introduced in the paper *Wildwood: a new 
random forest algorithm* by S. Gaïffas, I. Merad and Y. Yu (2021).
It follows `scikit-learn`'s API and can be used as an inplace replacement for its 
Random Forest algorithms (although multilabel/multiclass training is *not* supported yet).
`WildWood` mainly provides, compared to standard Random Forest algorithms, the 
following things: 

- Improved predictions with less trees
- Faster training times (using a histogram strategy similar to LightGBM)
- Native support for categorical features
- Parallel training of the trees in the forest 

Multi-class classification can be performed with `WildWood` using `ForestClassifier` 
while regression can be performed with `ForestRegressor`.

## Documentation

Documentation is available here: 

>   [http://wildwood.readthedocs.io](http://wildwood.readthedocs.io)

## Installation

The easiest way to install wildwood is using pip
```{code-block} bash
pip install wildwood
```
But you can also use the latest development from github directly with
```{code-block} bash
pip install git+https://github.com/pyensemble/wildwood.git
```

## Basic usage

Basic usage follows the standard scikit-learn API. You can simply use
```{code-block} python
from wildwood import ForestClassifier

clf = ForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
```
to train a classifier with all default hyper-parameters.
However, let us pinpoint below some of the most interesting ones.

### Categorical features

You should avoid one-hot encoding of categorical features and specify instead to 
`WildWood` which features should be considered as categorical. 
This is done using the `categorical_features` argument, which is either a boolean mask 
or an array of indices corresponding to the categorical features.

```{code-block} python
from wildwood import ForestClassifier

# Assuming columns 0 and 2 are categorical in X
clf = ForestClassifier(categorical_features=[0, 2])
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
```

```{warning}
For now, `WildWood` will actually use a maximum of 256 modalities for categorical 
features, since internally features are encoded using a memory efficient ``uint8`` data 
type. This will change in a near future.
```

### Improved predictions through aggregation with exponential weights

By default (`aggregation=True`) the predictions produced by `WildWood` are an 
aggregation with exponential weights (computed on out-of-bag samples) of the predictions
given by all the possible prunings of each tree. This is computed exactly and very 
efficiently, at a cost nearly similar to that of a standard Random Forest (which 
averages the prediction of leaves).
See {ref}`description-wildwood` for a deeper description of `WildWood`.
