import os

from time import time
import argparse

import numpy as np
import datasets

import sys

sys.path.extend([".", ".."])

from wildwood.forest import ForestClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default="Moons")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_true", default=False)
parser.add_argument('--datasets-path', type=str, default="data")
parser.add_argument('--datasets-subsample', type=int, default=100000)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--n-jobs', type=int, default=-1)
parser.add_argument('--criterion', type=str, default='gini')
parser.add_argument('--random-state', type=int, default=0)


args = parser.parse_args()


print("Running Wildwood classifier with training set {}".format(args.dataset))

dataset = datasets.load_dataset(args)

if dataset.task != "classification":
    print("The loaded datasets is not for classification ... exiting")
    exit()
train_sample_weights = dataset.get_train_sample_weights()



print("Training Wildwood classifier ...")
clf = ForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=args.n_jobs, criterion=args.criterion, multiclass="ovr", verbose=True)
clf.fit(dataset.data_train[:100], dataset.target_train[:100])#, sample_weight=train_sample_weights)

tic = time()
clf.fit(dataset.data_train, dataset.target_train, sample_weight=train_sample_weights)
toc = time()
print(f"fitted in {toc - tic:.3f}s")

clf.predict((dataset.data_test[:100]))

datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)

"""
predicted_proba_test = clf.predict_proba(datasets.data_test)
predicted_test = np.argmax(predicted_proba_test, axis=1)


if datasets.binary:
    roc_auc = roc_auc_score(datasets.target_test ,predicted_proba_test[:,1] , multi_class="ovo")
    avg_precision_score = average_precision_score(datasets.target_test, predicted_proba_test[:, 1])
else:
    onehot_target_test = datasets.onehotencode(datasets.target_test)

    roc_auc = roc_auc_score(onehot_target_test , predicted_proba_test, multi_class="ovo")
    avg_precision_score = average_precision_score(onehot_target_test, predicted_proba_test)

acc = accuracy_score(datasets.target_test, predicted_test)
log_loss_value = log_loss(datasets.target_test, predicted_proba_test)
print(f"ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")
print("ROC AUC computed with multi_class='ovo' (see sklearn docs)")

print(f"Log loss: {log_loss_value :.4f}")

print(f"Average precision score: {avg_precision_score :.4f}")
"""