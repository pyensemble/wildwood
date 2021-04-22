from time import time
import argparse

import numpy as np
import datasets
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, average_precision_score
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default="BreastCancer")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_false", default=True)
parser.add_argument('--datasets-path', type=str, default="data")
parser.add_argument('--datasets-subsample', type=int, default=100000)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--n-jobs', type=int, default=-1)
parser.add_argument('--random-state', type=int, default=0)



args = parser.parse_args()



print("Running XGBoost regressor with training set {}".format(args.dataset))
print ("")
print("with arguments")
print(args)
print ("")

dataset = datasets.load_dataset(args)

if dataset.task != "regression":
    print("The loaded datasets is not for regression ... exiting")
    exit()

print("Training XGBoost regressor ...")


#sample_weight = datasets.get_train_sample_weights()

reg = xgb.XGBRegressor(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=args.n_jobs)
tic = time()
reg.fit(dataset.data_train, dataset.target_train)#, sample_weight=sample_weight)#, categorical_feature=list(cat_features))# if args.specify_cat_features else None)
toc = time()

print(f"fitted in {toc - tic:.3f}s")

datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)
