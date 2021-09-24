
# Experiments

Here, we describe how some experiments from {cite}`c-wildwood` can be reproduced.

## Experiments with hyperparameters optimization

To run experiments with hyperparameters optimization, under directory `experiments/`,
use for instance
```bash
python run_hyperopt_classfiers.py --clf_name WildWood --dataset_name adult
```
with the `WildWood` classifier and `adult` dataset. Some options are

``--n_estimators`` or ``-t``
: Number of estimators (maximal number of boosting iterations for gradient boosting), default=100.

``--hyperopt_evals`` or ``-n``
: Number of hyperopt (hyperoptimization) steps, default=50.

## Experiments on default parameters

To run experiments with default parameters, under directory `experiments/`, use
```bash
python run_benchmark_default_params_classifiers.py --clf_name WildWood --dataset_name adult
```
with the `WildWood` classifier and `adult` dataset. 

## Datasets and classifiers

Here are the options available for the scripts ``run_hyperopt_classfiers.py`` and 
``run_benchmark_default_params_classifiers.py``.

``dataset_name``
: can be set as ``adult``, ``bank``, ``breastcancer``, ``car``, ``cardio``, ``churn``, 
``default-cb``, ``letter``, ``satimage``, ``sensorless``, ``spambase``, ``amazon``, 
``covtype``, ``internet``, ``kick``, ``kddcup``, ``higgs``

``clf_name``
: can be set as ``LGBMClassifier``, ``XGBClassifier``, ``CatBoostClassifier``, 
``RandomForestClassifier``, ``HistGradientBoostingClassifier``, ``WildWood``

## Experiments presented in {cite}`c-wildwood`


1. Figure 1 is produced using ``fig_aggregation_effect.py``.
1. Figure 2 is produced using ``n_tree_experiment.py``.
1. Tables 1 and 3 from the paper are produced using ``run_hyperopt_classfiers.py``
   with ``n_estimators=5000`` for gradient boosting algorithms and with
   ``n_estimators=n`` for ``RFn`` and ``WWn``. Use
   ```bash
   python run_hyperopt_classfiers.py --clf_name <classifier> --dataset_name <dataset> --n_estimators <n_estimators>
   ```
   for each pair ``(<classifier>, <dataset>)`` to run hyperparameters optimization 
   experiments and use for example
   ```python
   import pickle as pkl
   filename = 'exp_hyperopt_xxx.pickle'
   with open(filename, "rb") as f:
       results = pkl.load(f)
   df = results["results"]
   ```
   to retrieve experiments information, such as AUC, logloss and their standard 
   deviations.

1. Tables 2 and 4 are produced with ``benchmark_default_params.py``, using
   ```bash
   python run_benchmark_default_params_classifiers.py --clf_name <classifier> --dataset_name <dataset>
   ```
   for each pair ``(<classifier>, <dataset>)`` to run experiments with default 
   parameters and use similar commands to retrieve experiments information.

1. Using experiments results (AUC and fit time) done by ``run_hyperopt_classfiers.py``,
   then concatenating dataframes and using ``fig_auc_fit_time.py`` to produce Figure 3.


## References


```{bibliography} biblio.bib
---
labelprefix: C
keyprefix: c-
style: plain
filter: docname in docnames
---
