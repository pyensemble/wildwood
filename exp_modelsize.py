import pandas as pd
import numpy as np
import pickle, sys

import lightgbm as lgb
from wildwood import ForestClassifier

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.ensemble import RandomForestClassifier as RFC

from time import time
from datetime import datetime

import xgboost as xgb
import catboost as cab

print("importing data loaders")

from wildwood.datasets import (
    load_adult,
    load_default_cb,
    load_bank,
    load_car,
    load_letter,
    load_breastcancer,
    load_satimage,
    load_spambase,
    load_internet,
    load_covtype,
    load_kick,
    load_sensorless,
    load_churn,
    load_electrical,
    load_eeg,
    load_amazon,
    load_kddcup99,
    load_higgs,
)

print("defining sizeof function")
def sizeof(x):
    return sys.getsizeof(pickle.dumps(x)) / 1000000


from collections import namedtuple

Classifier = namedtuple("Classifier", ["name", "objet"])

print("defining classifiers")
classifiers = [
    Classifier(name=nm, objet=obj)
    for nm, obj in list(
        zip(
            ["WW", "LGB", "CAB", "XGB", "HGB", "RF"],
            [
                ForestClassifier,
                lgb.LGBMClassifier,
                cab.CatBoostClassifier,
                xgb.XGBClassifier,
                HGB,
                RFC,
            ],
        )
    )
]

datasets = [
    load_sensorless,
    load_covtype,
    load_internet,
    load_kick,
    load_kddcup99,
    load_amazon,
    load_higgs,
]


import os
from pathlib import Path

print("defining get_params function")
def get_params(dataset_name, model_name, path):
    if os.path.exists(path + dataset_name):
        unfiltered = sorted(
            Path(path + dataset_name).iterdir(), key=os.path.getmtime, reverse=True
        )
        # print("unfiltered")
        # for x in unfiltered:
        #    print(x)
        def predicate(path):
            str_path = str(path)
            if str_path.split("/")[-1][0] != "b":
                return False
            with open(path, "rb") as f:
                content = pickle.load(f)
                if model_name.lower() not in str_path:
                    return False
                if model_name.lower() == "ww":
                    return content["result"]["params"]["n_estimators"] == 10
            return True

        paths = list(filter(predicate, unfiltered))
        # print("filtered")
        # for x in paths:
        #    print(x)
        if len(paths) == 0:
            print("couldn't find parameters for %s , %s" % (dataset_name, model_name))
            return {}
        with open(str(paths[0]), "rb") as f:
            content = pickle.load(f)
            params = content["result"]["params"]
            return params

print("launching main loop")

clf_col, data_col, size_col = [], [], []
for data in datasets[5:]:
    print(data.__name__[5:])
    X_train, _, y_train, _ = data().extract()
    for clf_tup in classifiers:
        print(clf_tup.name)

        params = get_params(data.__name__[5:], clf_tup.name, "results-all/results/")

        clf = clf_tup.objet(**params)

        if clf_tup.name == "CAB":
            clf.fit(X_train, y_train, verbose=False)
        else:
            clf.fit(X_train, y_train)

        clf_col.append(clf_tup.name)
        data_col.append(data.__name__[5:])
        size_col.append(sizeof(clf))

        if clf_tup.name == "WW":
            clf.lighten()

            clf_col.append("WW_light")
            data_col.append(data.__name__[5:])
            size_col.append(sizeof(clf))


    df = pd.DataFrame({"classifier": clf_col, "dataset": data_col, "size": size_col})

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    filename = "exp_modelsize_" + "_" + now + ".pickle"
    print("saving results into %s" % filename)
    with open(filename, "wb") as f:
        pickle.dump(
            df,
            f,
        )
    print("done")

print("finished all datasets")