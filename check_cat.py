from time import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score

from wildwood.datasets import load_churn, load_bank
from wildwood.forest import ForestClassifier

np.set_printoptions(precision=2)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


data_random_state = 42

dataset = load_bank()
dataset.one_hot_encode = False
dataset.standardize = False
X_train, X_test, y_train, y_test = dataset.extract(random_state=data_random_state)


n_estimators = 100

clf = ForestClassifier(
    n_estimators=n_estimators,
    random_state=42,
    aggregation=False,
    max_features=None,
    categorical_features=dataset.categorical_features_,
    n_jobs=1,
    class_weight="balanced",
    criterion="entropy",
)
clf.fit(X_train, y_train)
y_scores_train = clf.predict_proba(X_train)
y_scores_test = clf.predict_proba(X_test)
avg_prec_train = average_precision_score(y_train, y_scores_train[:, 1])
avg_prec_test = average_precision_score(y_test, y_scores_test[:, 1])
print("Categorical")
print("AP(train):", avg_prec_train, "AP(test):", avg_prec_test)


clf = ForestClassifier(
    n_estimators=n_estimators,
    random_state=42,
    aggregation=False,
    max_features=None,
    # categorical_features=dataset.categorical_features_,
    criterion="entropy",
    n_jobs=1,
    class_weight="balanced",
)
clf.fit(X_train, y_train)
y_scores_train = clf.predict_proba(X_train)
y_scores_test = clf.predict_proba(X_test)
avg_prec_train = average_precision_score(y_train, y_scores_train[:, 1])
avg_prec_test = average_precision_score(y_test, y_scores_test[:, 1])
print("Ordinal")
print("AP(train):", avg_prec_train, "AP(test):", avg_prec_test)

dataset.one_hot_encode = True
dataset.standardize = False
X_train, X_test, y_train, y_test = dataset.extract(random_state=data_random_state)


clf = ForestClassifier(
    n_estimators=n_estimators,
    random_state=42,
    aggregation=False,
    max_features=None,
    n_jobs=1,
    class_weight="balanced",
    criterion="entropy",
)
clf.fit(X_train, y_train)
y_scores_train = clf.predict_proba(X_train)
y_scores_test = clf.predict_proba(X_test)
avg_prec_train = average_precision_score(y_train, y_scores_train[:, 1])
avg_prec_test = average_precision_score(y_test, y_scores_test[:, 1])
print("One-hot")
print("AP(train):", avg_prec_train, "AP(test):", avg_prec_test)
