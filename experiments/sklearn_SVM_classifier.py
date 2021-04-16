import os

from time import time
import argparse

import numpy as np
import datasets

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default="Moons")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_true", default=False)
parser.add_argument('--datasets-path', type=str, default="data")
parser.add_argument('--datasets-subsample', type=int, default=100000)
parser.add_argument('--penalty', type=str, default='l2')
parser.add_argument('--random-state', type=int, default=0)
parser.add_argument('--n-jobs', type=int, default=None)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--max-iter', type=int, default=1000)


args = parser.parse_args()


print("Running sklearn linear SVM with training set {}".format(args.dataset))
print ("")
print("with arguments")
print(args)
print ("")

if not args.one_hot_categoricals:
    print("WARNING : categorical variables will not be one hot encoded")

dataset = datasets.load_dataset(args)
dataset.info()
sample_weights = dataset.get_train_sample_weights()

print("Training Scikit Learn Linear SVM classifier ...")
tic = time()

clf = SGDClassifier(penalty=args.penalty, random_state=args.random_state, n_jobs=args.n_jobs, verbose=args.verbose, max_iter=args.max_iter)

clf.fit(dataset.data_train, dataset.target_train, sample_weight=sample_weights)
toc = time()

print(f"done in {toc - tic:.3f}s")

predicted_test = clf.predict(dataset.data_test)
acc = accuracy_score(dataset.target_test, predicted_test)
print(f"Accuracy : {acc :.4f}")
