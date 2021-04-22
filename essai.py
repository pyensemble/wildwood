import sys
import numpy as np
import pandas as pd

sys.path.extend([".."])

from wildwood.forest import ForestClassifier
from wildwood.datasets import load_bank

dataset = load_bank()
dataset.one_hot_encode = True
dataset.standardize = False
dataset.drop = None
X_train, X_test, y_train, y_test = dataset.extract(random_state=42)


clf = ForestClassifier(
    n_estimators=1,
    n_jobs=1,
    class_weight="balanced",
    random_state=42,
    aggregation=False,
    max_features=None,
    dirichlet=0.0,
)

clf.fit(X_train, y_train)


print(clf.path_leaf(X_train[:1]))

