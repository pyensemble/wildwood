"""
This module trains wildwood on the wine dataset, just to check that things work for
categorical features,

Some notes
----------



"""

from time import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
# from sklearn.tree import ExtraTreeClassifier as SkExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from wildwood.forest import ForestBinaryClassifier


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

np.set_printoptions(precision=2)


random_state = 42
np.random.seed(0)

# boston = datasets.load_boston(return_X_y=False)
# data = pd.DataFrame(boston.data)
# categorical_features = [3]
# X = data.drop('PRICE', axis=1)
# y = data['PRICE']

X = np.random.randint(0, 3, (20, 5))
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
categorical_features = [i for i in range(5)]
clf1 = ForestBinaryClassifier(n_estimators=3, random_state=42, categorical_features=categorical_features)
clf1.fit(X, y)

