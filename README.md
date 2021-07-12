
[![Build Status](https://travis-ci.com/pyensemble/wildwood.svg?branch=master)](https://travis-ci.com/pyensemble/wildwood)
[![Documentation Status](https://readthedocs.org/projects/wildwood/badge/?version=latest)](https://wildwood.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wildwood)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/wildwood)
[![GitHub stars](https://img.shields.io/github/stars/pyensemble/wildwood)](https://github.com/pyensemble/wildwood/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/pyensemble/wildwood)](https://github.com/pyensemble/wildwood/issues)
[![GitHub license](https://img.shields.io/github/license/pyensemble/wildwood)](https://github.com/pyensemble/wildwood/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/pyensemble/wildwood/badge.svg?branch=master)](https://coveralls.io/github/pyensemble/wildwood?branch=master)

# WildWood

Scikit-Learn compatible Random Forest algorithms

[Documentation](https://wildwood.readthedocs.io) | [Reproduce experiments](https://wildwood.readthedocs.io/en/latest/experiments.html) |

# Installation

The easiest way to install wildwood is using pip

    pip install wildwood

But you can also use the latest development from github directly with

    pip install git+https://github.com/pyensemble/wildwood.git

# Experiments

## Experiments with hyperparameters optimization

To run experiments with hyperparameters optimization, under directory `experiments/`, use

    python run_hyperopt_classfiers.py --clf_name WildWood --dataset_name adult

(with `WildWood` and on `adult` dataset in this example).

Some options are

- Setting `--n_estimators` or `-t` for number of estimators 
  (for maximal number of boosting iterations in case of gradient boosting algorithms), default 100.
- Setting `--hyperopt_evals` or `-n` for number of hyperopt steps, default 50.

## Experiments on default parameters

To run experiments with default parameters, under directory `experiments/`, use

    python run_benchmark_default_params_classifiers.py --clf_name WildWood --dataset_name adult

(with `WildWood` and on `adult` dataset in this example).

## Datasets and classifiers

For both `run_hyperopt_classfiers.py` and `run_benchmark_default_params_classifiers.
py`, the available options for `dataset_name` are:

- `adult`
- `bank`
- `breastcancer`
- `car`
- `cardio`
- `churn`
- `default-cb`
- `letter`
- `satimage`
- `sensorless`
- `spambase`
- `amazon`
- `covtype`
- `internet`
- `kick`
- `kddcup`
- `higgs`

while the available options for `clf_name` are

- `LGBMClassifier`
- `XGBClassifier`
- `CatBoostClassifier`
- `RandomForestClassifier`
- `HistGradientBoostingClassifier`
- `WildWood`

## Experiments presented in the paper

All the scripts allowing to reproduce the experiments from the paper are available 
in the `experiments/` folder

1. Figure 1 is produced using `fig_aggregation_effect.py`.
1. Figure 2 is produced using `n_tree_experiment.py`. 
1. Tables 1 and 3 from the paper are produced using `run_hyperopt_classfiers.py` 
   with `n_estimators=5000` for gradient boosting algorithms and with 
   `n_estimators=n` for `RFn` and `WWn`
   - call
   ```shell
   python run_hyperopt_classfiers.py --clf_name <classifier> --dataset_name <dataset> --n_estimators <n_estimators>
   ```   
   for each pair `(<classifier>, <dataset>)` to run hyperparameters optimization experiments;
   - use for example
   ```python
   import pickle as pkl
   filename = 'exp_hyperopt_xxx.pickle'
   with open(filename, "rb") as f:
       results = pkl.load(f)
   df = results["results"]
   ```
   to retrieve experiments information, such as AUC, logloss and their standard deviation.

1. Tables 2 and 4 are produced using `benchmark_default_params.py`
    - call
   ```shell
   python run_benchmark_default_params_classifiers.py --clf_name <classifier> --dataset_name <dataset>
   ```   
   for each pair `(<classifier>, <dataset>)` to run experiments with default parameters;
   -  use similar commands to retrieve experiments information.
    
1. Using experiments results (AUC and fit time) done by `run_hyperopt_classfiers.py`, 
   then concatenating dataframes and using `fig_auc_fit_time.py` to produce Figure 3.

