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

    return [fit_time] + datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)


def xgboost(dataset, random_state=0, n_estimators=100, n_jobs=-1, max_depth=None, verbose=False):

    reg = xgb.XGBRegressor(n_estimators= n_estimators, random_state= random_state, n_jobs=n_jobs)
    tic = time()
    reg.fit(dataset.data_train,
            dataset.target_train)  # , sample_weight=sample_weight)#, categorical_feature=list(cat_features))# if args.specify_cat_features else None)
    toc = time()
    fit_time = toc - tic

    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)


def catboost(dataset, n_estimators=100, n_jobs=-1, random_state=0, verbose=0):


    reg = CatBoostRegressor(n_estimators=n_estimators, random_state=random_state, thread_count=n_jobs)
    tic = time()
    reg.fit(dataset.data_train,
            dataset.target_train)  # , sample_weight=sample_weight)#, categorical_feature=list(cat_features))# if args.specify_cat_features else None)
    toc = time()
    fit_time = toc - tic

    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)


def lightgbm(dataset, n_estimators=100, n_jobs=-1, random_state=0, verbose=0):
    reg = lgb.LGBMRegressor(n_estimators= n_estimators, random_state= random_state, n_jobs= n_jobs)
    tic = time()
    data_train = dataset.data_train.astype({c: 'category' for c in range(dataset.nb_continuous_features, dataset.n_features)})
    reg.fit(data_train,
            dataset.target_train)  # , sample_weight=sample_weight)#, categorical_feature=list(cat_features))# if args.specify_cat_features else None)
    toc = time()
    fit_time = toc - tic

    print(f"fitted in {toc - tic:.3f}s")
    data_test = dataset.data_test.astype({c: 'category' for c in range(dataset.nb_continuous_features, dataset.n_features)})

    return [fit_time] + datasets.evaluate_regressor(reg, data_test, dataset.target_test)


def sklearn_random_forest(dataset, n_estimators=100, n_jobs=-1, criterion='mse', random_state=0):
    reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion,
                                random_state=random_state, n_jobs=n_jobs)
    tic = time()
    reg.fit(dataset.data_train, dataset.target_train)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)


def wildwood(dataset, n_estimators=100, n_jobs=-1, random_state=0):
    #train_sample_weights = dataset.get_train_sample_weights()

    reg = ForestRegressor(n_jobs=n_jobs)
    tic = time()
    reg.fit(dataset.data_train, dataset.target_train)
    toc = time()
    fit_time = toc - tic
    print(f"fitted in {toc - tic:.3f}s")

    return [fit_time] + datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)

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

regressors_categorical = [catboost, lightgbm, wildwood]
regressors_one_hot = [linear_regression, xgboost, sklearn_random_forest, wildwood]
order = ["fit time", "predict time", "mse"]
stats = list()#pd.DataFrame(index=dataset_names, columns=[c.__name__ for c in regressors])

for dataset_name in dataset_names:

    args.dataset = dataset_name
    args.one_hot_categoricals = False
    print("")
    print("Loading dataset {} with categorical features".format(dataset_name))
    print("")
    dataset = datasets.load_dataset(args, as_pandas=True)
    dataset.info()

    for reg in regressors_categorical:

        print("")
        print("Benchmarking {} regressor on dataset {}".format(reg.__name__, dataset_name))
        print("")
        record = reg(dataset)

        for i in range(len(record)):
            stats.append([reg.__name__, dataset_name, order[i], record[i]])


    args.one_hot_categoricals = True
    print("")
    print("Loading dataset {} with one hot categorical features".format(dataset_name))
    print("")
    dataset = datasets.load_dataset(args, as_pandas=True)
    dataset.info()

    for reg in regressors_one_hot:

        print("")
        print("Benchmarking {} regressor on dataset {}".format(reg.__name__, dataset_name))
        print("")
        record = reg(dataset)

        for i in range(len(record)):
            stats.append([reg.__name__ + "_oh", dataset_name, order[i], record[i]])

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

with open("benchmark_reg_"+dtime1+".pickle", 'wb') as f:
    pickle.dump({"date_time" : dtime2, "commit": commit, "dataframe" : bench}, f)
