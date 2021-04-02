from time import time
import pandas as pd
import datasets
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--all-datasets', action="store_true", default=False)

all_datasets = parser.parse_args().all_datasets
del parser

import sys

sys.path.extend([".", ".."])

from wildwood.forest import ForestClassifier
import pickle
import datetime
import subprocess

def logistic_regression(dataset, penalty='l2', dual=False, random_state=0, n_jobs=-1, solver='lbfgs', verbose=0, max_iter=100):
    clf = LogisticRegression(penalty=penalty, dual=dual, n_jobs=n_jobs, solver=solver,
                             verbose=verbose, max_iter=max_iter, class_weight="balanced")
    tic = time()
    clf.fit(dataset.data_train, dataset.target_train)  # , sample_weight=sample_weights)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)


def xgboost(dataset, random_state=0, use_label_encoder=False, n_estimators=100, n_jobs=-1, max_depth=None, verbose=False):

    print("Training XGBoost classifier ...")
    tic = time()

    clf = xgb.XGBClassifier(random_state= random_state, use_label_encoder=False, n_estimators=n_estimators,
                            n_jobs= n_jobs, max_depth= max_depth)
    print("Running XGBClassifier with use_label_encoder=False")
    clf.fit(dataset.data_train, dataset.target_train, verbose=verbose, sample_weight=dataset.get_train_sample_weights())
    toc = time()

    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)


def catboost(dataset, n_estimators=100, n_jobs=-1, random_state=0, verbose=0):
    cat_features = list(range(dataset.nb_continuous_features, dataset.n_features))

    clf = CatBoostClassifier(n_estimators= n_estimators, random_seed= random_state, cat_features=cat_features,
                             thread_count= n_jobs)

    tic = time()
    clf.fit(dataset.data_train, dataset.target_train, verbose=bool(verbose), sample_weight=dataset.get_train_sample_weights(),
            cat_features=cat_features)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)


def lightgbm(dataset, n_estimators=100, n_jobs=-1, random_state=0, verbose=0):
    cat_features = list(range(dataset.nb_continuous_features, dataset.n_features))

    clf = lgb.LGBMClassifier(n_estimators= n_estimators, random_state= random_state, n_jobs= n_jobs)

    tic = time()
    clf.fit(dataset.data_train, dataset.target_train, verbose=bool(verbose),
            sample_weight=dataset.get_train_sample_weights(), categorical_feature=cat_features)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)


def sklearn_random_forest(dataset, n_estimators=100, n_jobs=-1, criterion='gini', random_state=0):

    clf = RandomForestClassifier(n_estimators= n_estimators, criterion= criterion, random_state= random_state, n_jobs= n_jobs)

    tic = time()

    clf.fit(dataset.data_train, dataset.target_train, sample_weight=dataset.get_train_sample_weights())
    toc = time()

    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)


def wildwood(dataset, n_estimators=100, n_jobs=-1, criterion='gini', random_state=0):

    clf = ForestClassifier(n_estimators= n_estimators, random_state= random_state,
                                 n_jobs= n_jobs , criterion=criterion)
    clf.fit(dataset.data_train[:100], dataset.target_train[:100])# , sample_weight=train_sample_weights)

    tic = time()
    clf.fit(dataset.data_train, dataset.target_train)#, sample_weight=dataset.get_train_sample_weights())
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    clf.predict((dataset.data_test[:100]))

    return [fit_time] + datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)

class args_class():
    def __init__(self):
        self.dataset = "Moons"
        self.normalize_intervals=True
        self.one_hot_categoricals=False
        self.dataset_path="data"
        self.random_state = 0
        self.dataset_subsample=None#50000
args = args_class()

dataset_names = ["Adult", "Bank", "Car", "Cardio", "Churn", "Default_cb", "Letter", "Satimage", "Sensorless", "Spambase", "BreastCancer"]
if all_datasets:
    dataset_names = ["Higgs", "Covtype", "KDDCup", "NewsGroups"] + dataset_names #take the biggest ones first in case there are data loading issues ...
    print("Using ALL datasets for benchmarking including : Higgs, Covtype, KDDCup, NewsGroups")

classifiers_one_hot = [wildwood, logistic_regression, xgboost, sklearn_random_forest]
classifiers_categorical = [catboost, lightgbm, wildwood]

order = ["fit time", "predict time", "AUC", "accuracy", "log loss", "avg precision score"]


stats = list()

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
        print("Benchmarking {} classifier on dataset {}".format(clf.__name__, dataset_name))
        print("")
        record = clf(dataset)
        for i in range(len(record)):
            stats.append([clf.__name__, dataset_name, order[i], record[i]])

    args.one_hot_categoricals = True
    print("")
    print("Loading dataset {} with one-hot categories".format(dataset_name))
    print("")
    dataset = datasets.load_dataset(args, as_pandas=True)
    dataset.info()

    for clf in classifiers_one_hot:
        clf_name = clf.__name__
        print("")
        print("Benchmarking {} classifier on dataset {}".format(clf_name, dataset_name))
        print("")
        record = clf(dataset)
        for i in range(len(record)):
            stats.append([clf.__name__ + "_oh", dataset_name, order[i], record[i]])

df = pd.DataFrame(stats, columns=["algorithm", "dataset", "metric", "value"])

bench = df.pivot(index=["dataset", "metric"], columns="algorithm")

#def extract_field(df, field):
#    indexes = list(df.index)
#    cols = list(df.columns)
#    new_df = pd.DataFrame(index=indexes, columns=cols)
#    for i in indexes:
#        for c in cols:
#            new_df.at[i, c] = df.loc[i, c][field]
#    return new_df

dtime1 = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
dtime2 = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])

with open("benchmark_clf_"+dtime1+".pickle", 'wb') as f:
    pickle.dump({"date_time" : dtime2, "commit": commit, "dataframe" : bench}, f)
