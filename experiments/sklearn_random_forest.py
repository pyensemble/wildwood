import os

from time import time
import argparse

import numpy as np
import datasets

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, average_precision_score

from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Moons")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_true", default=False)
parser.add_argument('--dataset-path', type=str, default="data")
parser.add_argument('--dataset-subsample', type=int, default=100000)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--n-jobs', type=int, default=-1)
parser.add_argument('--criterion', type=str, default='gini')
parser.add_argument('--random-state', type=int, default=0)


args = parser.parse_args()


print("Running sklearn Random Forest classifier with training set {}".format(args.dataset))

dataset = datasets.load_dataset(args)
train_sample_weights = dataset.get_train_sample_weights()

print("Training Scikit Learn Random forest classifier ...")
tic = time()

clf = RandomForestClassifier(n_estimators=args.n_estimators, criterion=args.criterion, random_state=args.random_state, n_jobs=args.n_jobs)
clf.fit(dataset.data_train, dataset.target_train, sample_weight=train_sample_weights)
toc = time()

print(f"fitted in {toc - tic:.3f}s")

datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)

