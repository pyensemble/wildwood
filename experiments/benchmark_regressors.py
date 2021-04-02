from time import time
import pandas as pd
import numpy as np
import datasets
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

import sys

sys.path.extend([".", ".."])

from wildwood.forest import ForestRegressor
import pickle
import datetime
import subprocess

def linear_regression(dataset, random_state=0, n_jobs=-1):


    #train_sample_weights = dataset.get_train_sample_weights()

    reg = LinearRegression(n_jobs=n_jobs)
    tic = time()
    reg.fit(dataset.data_train, dataset.target_train)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)
    eval.update({"fit_time" : fit_time})
    return eval


def xgboost_regressor(dataset, random_state=0, n_estimators=100, n_jobs=-1, max_depth=None, verbose=False):

    reg = xgb.XGBRegressor(n_estimators= n_estimators, random_state= random_state, n_jobs=n_jobs)
    tic = time()
    reg.fit(dataset.data_train,
            dataset.target_train)  # , sample_weight=sample_weight)#, categorical_feature=list(cat_features))# if args.specify_cat_features else None)
    toc = time()
    fit_time = toc - tic

    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)
    eval.update({"fit_time" : fit_time})
    return eval


def catboost_regressor(dataset, n_estimators=100, n_jobs=-1, random_state=0, verbose=0):


    reg = CatBoostRegressor(n_estimators=n_estimators, random_state=random_state, thread_count=n_jobs)
    tic = time()
    reg.fit(dataset.data_train,
            dataset.target_train)  # , sample_weight=sample_weight)#, categorical_feature=list(cat_features))# if args.specify_cat_features else None)
    toc = time()
    fit_time = toc - tic

    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)
    eval.update({"fit_time" : fit_time})
    return eval


def lightgbm_regressor(dataset, n_estimators=100, n_jobs=-1, random_state=0, verbose=0):
    reg = lgb.LGBMRegressor(n_estimators= n_estimators, random_state= random_state, n_jobs= n_jobs)
    tic = time()
    data_train = dataset.data_train.astype({c: 'category' for c in range(dataset.nb_continuous_features, dataset.n_features)})
    reg.fit(data_train,
            dataset.target_train)  # , sample_weight=sample_weight)#, categorical_feature=list(cat_features))# if args.specify_cat_features else None)
    toc = time()
    fit_time = toc - tic

    print(f"fitted in {toc - tic:.3f}s")
    data_test = dataset.data_test.astype({c: 'category' for c in range(dataset.nb_continuous_features, dataset.n_features)})

    eval = datasets.evaluate_regressor(reg, data_test, dataset.target_test)
    eval.update({"fit_time" : fit_time})
    return eval


def sklearn_random_forest_regressor(dataset, n_estimators=100, n_jobs=-1, criterion='mse', random_state=0):
    reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion,
                                random_state=random_state, n_jobs=n_jobs)
    tic = time()
    reg.fit(dataset.data_train, dataset.target_train)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)
    eval.update({"fit_time" : fit_time})
    return eval


def wildwood_regressor(dataset, n_estimators=100, n_jobs=-1, random_state=0):
    #train_sample_weights = dataset.get_train_sample_weights()

    reg = ForestRegressor(n_jobs=n_jobs)
    tic = time()
    reg.fit(dataset.data_train, dataset.target_train)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    eval = datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)
    eval.update({"fit_time" : fit_time})
    return eval

class args_class():
    def __init__(self):
        self.dataset = "BreastCancer"
        self.normalize_intervals=True
        self.dataset_path="data"
        self.one_hot_categoricals=False
        self.random_state = 0
        self.dataset_subsample=None#50000
args = args_class()

dataset_names = ["BreastCancer", "Diabetes", "Boston", "CaliforniaHousing"]

regressors = [linear_regression, xgboost_regressor, sklearn_random_forest_regressor, catboost_regressor, lightgbm_regressor, wildwood_regressor]


stats = pd.DataFrame(index=dataset_names, columns=[c.__name__ for c in regressors])

for dataset_name in dataset_names:

    args.dataset = dataset_name
    print("")
    print("Loading dataset {}".format(dataset_name))
    print("")
    dataset = datasets.load_dataset(args, as_pandas=True)
    dataset.info()

    for reg in regressors:

        print("")
        print("Benchmarking {} on dataset {}".format(reg.__name__, dataset_name))
        print("")
        stats.at[dataset_name, reg.__name__] = reg(dataset)



def extract_field(df, field):
    indexes = list(df.index)
    cols = list(df.columns)
    new_df = pd.DataFrame(index=indexes, columns=cols)
    for i in indexes:
        for c in cols:
            new_df.at[i, c] = df.loc[i, c][field]
    return new_df


with open("benchmark_regression"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".pickle", 'wb') as f:
    dtime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    gitind = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    pickle.dump(dtime, f)
    pickle.dump(gitind, f)
    pickle.dump(stats, f)
