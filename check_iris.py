"""
This module trains wildwood on the iris dataset, just to check that things work for
more than 2 features, to check random sampling of columns and performance on 3-class
classification

Some notes
----------

With aggregation :

time to fit:  22.22562289237976
time to predict_proba:  39.113215923309326 python check_iris.py
time to fit:  22.22562289237976
time to predict_proba:  39.113215923309326

"""

from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
# from sklearn.tree import ExtraTreeClassifier as SkExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier

# from wildwood._classes import DecisionTreeClassifier

from wildwood.forest import ForestClassifier


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

np.set_printoptions(precision=2)


random_state = 42

iris = datasets.load_iris()


covtype = datasets.fetch_covtype(download_if_missing=True)

# rcv1_train = datasets.fetch_rcv1(subset='train', download_if_missing=True)
# rcv1_test = datasets.fetch_rcv1(subset='test', download_if_missing=True)

X = covtype.data
y = covtype.target

# X_train = rcv1_train.data
# y_train = rcv1_train.target
# X_test = rcv1_test.data
# y_test = rcv1_test.target

clf_kwargs = {
    "n_estimators": 5,
    "min_samples_split": 2,
    "random_state": random_state,
    "n_jobs": -1,
    "dirichlet": 1e-5,
    "step": 2.0,
    "aggregation": True
}

clf = ForestClassifier(**clf_kwargs)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

tic = time()
clf.fit(X_train, y_train)
toc = time()
print("time to fit: ", toc - tic)

tic = time()
y_scores = clf.predict_proba(X_test)
toc = time()
print("time to predict_proba: ", toc - tic)

tic = time()
y_pred = clf.predict(X_test)
toc = time()
print("time to predict: ", toc - tic)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


print(cm)
print(acc)
