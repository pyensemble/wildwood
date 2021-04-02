import os

from time import time
import argparse

import numpy as np
import datasets

from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="BreastCancer")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_false", default=True)
parser.add_argument('--dataset-path', type=str, default="data")
parser.add_argument('--dataset-subsample', type=int, default=100000)
parser.add_argument('--random-state', type=int, default=0)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--n-jobs', type=int, default=-1)



args = parser.parse_args()


print("Running sklearn Linear Regression with training set {}".format(args.dataset))

dataset = datasets.load_dataset(args)
#train_sample_weights = dataset.get_train_sample_weights()

if dataset.task != "regression":
    print("The loaded dataset is not for regression ... exiting")
    exit()

print("Training sklearn Linear Regression ...")

reg = LinearRegression(n_jobs=args.n_jobs)
tic = time()
reg.fit(dataset.data_train, dataset.target_train)
toc = time()

print(f"fitted in {toc - tic:.3f}s")

datasets.evaluate_regressor(reg, dataset.data_test, dataset.target_test)

