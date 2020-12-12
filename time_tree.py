
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier as SkExtraTreeClassifier

from wildwood._classes import DecisionTreeClassifier


from time import time

import cProfile


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

np.set_printoptions(precision=2)


print("JIT compilation...")
tic = time()
X, y = make_circles(n_samples=5, noise=0.2, factor=0.5, random_state=1)
clf = DecisionTreeClassifier(min_samples_split=3)
clf.fit(X, y)
clf.predict_proba(X)
toc = time()
print("Spent {time} compiling.".format(time=toc - tic))


n_samples = 1_000_000
random_state = 42


datasets = [
    (
        "circles",
        make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
    ),
    ("moons", make_moons(n_samples=n_samples, noise=0.3, random_state=0)),
]

classifiers = [
    ("tree", DecisionTreeClassifier(min_samples_split=3)),
    ("sk_tree", SkDecisionTreeClassifier(min_samples_split=3, random_state=42)),
    ("sk_extra", SkExtraTreeClassifier(min_samples_split=3, random_state=42))
]


dataset = []
classifier = []
timings = []
task = []


# cprofile
# n_samples = 1_000_000
# X, y = make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1)
#
# cProfile.run("clf.fit(X, y)", "main_tree2")
#
# exit(0)


for data_name, (X, y) in datasets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    for clf_name, clf in classifiers:
        tic = time()
        clf.fit(X_train, y_train)
        toc = time()
        logging.info("%s had %d nodes" % (clf_name, clf.tree_.node_count))
        dataset.append(data_name)
        classifier.append(clf_name)
        timings.append(toc - tic)
        task.append("fit")

        tic = time()
        clf.predict_proba(X_test)
        toc = time()
        dataset.append(data_name)
        classifier.append(clf_name)
        timings.append(toc - tic)
        task.append("predict")


results = pd.DataFrame(
    {"dataset": dataset, "task": task, "classifier": classifier, "timings": timings}
)

print(results.pivot(index=["dataset", "task"], columns="classifier"))

