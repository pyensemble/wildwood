from time import time
import pandas as pd
import numpy as np
import datasets
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

import sys

sys.path.extend([".", ".."])

from wildwood.forest import ForestBinaryClassifier
import pickle
import datetime


def logistic_regression_classifier(dataset, penalty='l2', dual=False, random_state=0, n_jobs=-1, solver='lbfgs', verbose=0, max_iter=100):
    clf = LogisticRegression(penalty=penalty, dual=dual, n_jobs=n_jobs, solver=solver,
                             verbose=verbose, max_iter=max_iter, class_weight="balanced")
    tic = time()
    clf.fit(dataset.data_train, dataset.target_train)  # , sample_weight=sample_weights)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)
    eval.update({"fit_time": fit_time})
    return eval


def xgboost_classifier(dataset, random_state=0, use_label_encoder=False, n_estimators=100, n_jobs=-1, max_depth=None, verbose=False):

    print("Training XGBoost classifier ...")
    tic = time()

    clf = xgb.XGBClassifier(random_state= random_state, use_label_encoder=False, n_estimators=n_estimators,
                            n_jobs= n_jobs, max_depth= max_depth)
    print("Running XGBClassifier with use_label_encoder=False")
    clf.fit(dataset.data_train, dataset.target_train, verbose=verbose, sample_weight=dataset.get_train_sample_weights())
    toc = time()

    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)
    eval.update({"fit_time": fit_time})
    return eval


def catboost_classifier(dataset, n_estimators=100, n_jobs=-1, random_state=0, verbose=0):
    cat_features = list(range(dataset.nb_continuous_features, dataset.n_features))

    clf = CatBoostClassifier(n_estimators= n_estimators, random_seed= random_state, cat_features=cat_features,
                             thread_count= n_jobs)

    tic = time()
    clf.fit(dataset.data_train, dataset.target_train, verbose=bool(verbose), sample_weight=dataset.get_train_sample_weights(),
            cat_features=cat_features)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")
    eval = datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)
    eval.update({"fit_time": fit_time})
    return eval


def lightgbm_classifier(dataset, n_estimators=100, n_jobs=-1, random_state=0, verbose=0):
    cat_features = list(range(dataset.nb_continuous_features, dataset.n_features))

    clf = lgb.LGBMClassifier(n_estimators= n_estimators, random_state= random_state, n_jobs= n_jobs)

    tic = time()
    clf.fit(dataset.data_train, dataset.target_train, verbose=bool(verbose),
            sample_weight=dataset.get_train_sample_weights(), categorical_feature=cat_features)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)
    eval.update({"fit_time": fit_time})
    return eval


def sklearn_random_forest_classifier(dataset, n_estimators=100, n_jobs=-1, criterion='gini', random_state=0):

    clf = RandomForestClassifier(n_estimators= n_estimators, criterion= criterion, random_state= random_state, n_jobs= n_jobs)

    tic = time()

    clf.fit(dataset.data_train, dataset.target_train, sample_weight=dataset.get_train_sample_weights())
    toc = time()

    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)
    eval.update({"fit_time": fit_time})
    return eval


def wildwood_classifier(dataset, n_estimators=100, n_jobs=-1, criterion='gini', random_state=0):

    clf = ForestBinaryClassifier(n_estimators= n_estimators, random_state= random_state,
                                 n_jobs= n_jobs , criterion=criterion)
    clf.fit(dataset.data_train[:100], dataset.target_train[:100])# , sample_weight=train_sample_weights)

    tic = time()
    clf.fit(dataset.data_train, dataset.target_train)#, sample_weight=dataset.get_train_sample_weights())
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    clf.predict((dataset.data_test[:100]))
    eval = datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)
    eval.update({"fit_time": fit_time})
    return eval

class args_class():
    def __init__(self):
        self.dataset = "Moons"
        self.normalize_intervals=True
        self.one_hot_categoricals=False
        self.dataset_path="data"
        self.random_state = 0
        self.dataset_subsample=None#50000
args = args_class()

dataset_names = ["Adult", "Bank", "Car", "Cardio", "Churn", "Default_cb", "Letter", "Satimage", "Sensorless", "Spambase", "BreastCancer"]#, "Higgs", "Covtype", "KDDCup", "NewsGroups"]

classifiers_one_hot = [logistic_regression_classifier, xgboost_classifier, sklearn_random_forest_classifier, wildwood_classifier]
classifiers_categorical = [catboost_classifier, lightgbm_classifier]
classifiers = classifiers_categorical + classifiers_one_hot

stats = pd.DataFrame(index=dataset_names, columns=[c.__name__ for c in classifiers])

for dataset_name in dataset_names:

    args.dataset = dataset_name
    args.one_hot_categoricals = False
    print("")
    print("Loading dataset {} with preserved categories".format(dataset_name))
    print("")
    dataset = datasets.load_dataset(args, as_pandas=True)
    dataset.info()

    for clf in classifiers_categorical:

        print("")
        print("Benchmarking {} on dataset {}".format(clf.__name__, dataset_name))
        print("")
        stats.at[dataset_name, clf.__name__] = clf(dataset)

    args.one_hot_categoricals = True
    print("")
    print("Loading dataset {} with one-hot categories".format(dataset_name))
    print("")
    dataset = datasets.load_dataset(args, as_pandas=True)
    dataset.info()

    for clf in classifiers_one_hot:
        clf_name = clf.__name__
        print("")
        print("Benchmarking {} on dataset {}".format(clf_name, dataset_name))
        print("")
        stats.at[dataset_name, clf_name] = clf(dataset)
        #stats.loc[dataset_name, clf_name,] = clf(dataset)


def extract_field(df, field):
    indexes = list(df.index)
    cols = list(df.columns)
    new_df = pd.DataFrame(index=indexes, columns=cols)
    for i in indexes:
        for c in cols:
            new_df.at[i, c] = df.loc[i, c][field]
    return new_df


with open("benchmark"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".pickle", 'wb') as f:
    pickle.dump(stats, f)
