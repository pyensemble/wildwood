from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier as SkExtraTreeClassifier

from wildwood._classes import DecisionTreeClassifier


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

np.set_printoptions(precision=2)

#
# logging.info("JIT compiling...")
# tic = time()
# X, y = make_circles(n_samples=5, noise=0.2, factor=0.5, random_state=1)
# clf = DecisionTreeClassifier(min_samples_split=3)
# clf.fit(X, y)
# clf.predict_proba(X)
# toc = time()
# logging.info("Spent {time} compiling.".format(time=toc - tic))

# n_samples = 150

n_samples = 10_000

random_state = 42

X, y = make_circles(
    n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_state
)

from wildwood._binning import Binner


binner = Binner()


X = np.array([
    [1, 0, 1],
    [1, 2, 4],
    [1, np.nan, -6],
    [0, 0, 125],
])

binner.fit(X)


X_binned = binner.transform(X)

print(binner.bin_thresholds_)

print(binner.n_bins_non_missing_)

print(X_binned)


print(X_binned.dtype)

#
# datasets = [
#     (
#         "circles",
#         make_circles(
#             n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_state
#         ),
#     ),
#     ("moons", make_moons(n_samples=n_samples, noise=0.3, random_state=random_state),),
# ]
#
# clf_kwargs = {"min_samples_split": 2, "random_state": random_state}
