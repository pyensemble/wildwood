from time import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from wildwood.dataset import load_churn, load_bank
from wildwood.forest import ForestClassifier

np.set_printoptions(precision=2)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


data_random_state = 42


dataset = load_bank()
dataset.one_hot_encode = False
X_train, X_test, y_train, y_test = dataset.extract(random_state=data_random_state)


clf = ForestClassifier(
    n_estimators=1,
    random_state=42,
    aggregation=False,
    categorical_features=dataset.categorical_features_,
    n_jobs=1,
    class_weight="balanced",
    # verbose=True,
)


clf.fit(X_train, y_train)
y_scores = clf.predict_proba(X_test)



# exit(0)
#
#
# random_state = 42
# np.random.seed(0)
#
# dataset = load_dataset(args=args)
# categorical_features = np.arange(dataset.nb_continuous_features, dataset.n_features)
# print("categorical features are ", categorical_features)
#
# print('wildwood clf with `categorical_features=None`')
# clf_without_cat = ForestClassifier(n_estimators=100,
#                        random_state=42,
#                        categorical_features=None,
#                        n_jobs=-1, class_weight='balanced')
#
# clf_without_cat.fit(dataset.data_train, dataset.target_train)
# evaluate_classifier(clf_without_cat, dataset.data_test, dataset.target_test, binary=dataset.binary)
#
# print('wildwood clf with `categorical_features`')
# clf_with_cat = ForestClassifier(n_estimators=100,
#                        random_state=42,
#                        categorical_features=categorical_features,
#                        n_jobs=-1, class_weight='balanced')
#
# clf_with_cat.fit(dataset.data_train, dataset.target_train)
# evaluate_classifier(clf_with_cat, dataset.data_test, dataset.target_test, binary=dataset.binary)
