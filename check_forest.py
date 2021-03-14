"""
In this module we use wildwood on a binary classification problem with 2 features but
with a very large sample size (to check that parallization works, and to track the
evolution of computing times
"""

from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split

# from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
# from sklearn.tree import ExtraTreeClassifier as SkExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier

# from wildwood._classes import DecisionTreeClassifier

from wildwood.forest import ForestClassifier


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


# n_samples = 1000
n_samples = 2_000_000

# n_samples = 10
random_state = 42

data_random_state = 123

datasets = [
    # (
    #     "circles",
    #     make_circles(
    #         n_samples=n_samples, noise=0.2, factor=0.5, random_state=data_random_state
    #     ),
    # ),
    (
        "moons",
        make_moons(n_samples=n_samples, noise=0.2, random_state=data_random_state),
    ),
]

clf_kwargs = {
    "n_estimators": 100,
    "min_samples_split": 2,
    "random_state": random_state,
    "n_jobs": -1,
    "dirichlet": 1e-8,
    "step": 1.0,
    "aggregation": True,
    "verbose": True
}


# classifiers = [
#     ("tree", DecisionTreeClassifier),
#     ("sk_tree", SkDecisionTreeClassifier)
# ]


classifiers = [
    # ("forest", ForestBinaryClassifier(n_estimators=1, **clf_kwargs)),
    ("forest", ForestClassifier(**clf_kwargs)),
    # ("sk_forest", RandomForestClassifier(**clf_kwargs))
    # ("tree", DecisionTreeClassifier(**clf_kwargs)),
    # ("sk_tree", SkDecisionTreeClassifier(**clf_kwargs)),
]


n_classifiers = len(classifiers)
n_datasets = len(datasets)
h = 0.2
i = 1

# iterate over datasets

# for ds_cnt, ds in enumerate(datasets):
#     # preprocess dataset, split into training and test part
#     ds_name, (X, y) = ds
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.4, random_state=42
#     )
#     # x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
#     # y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
#     # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # # just plot the dataset first
#     # cm = plt.cm.RdBu
#     # cm_bright = ListedColormap(["#FF0000", "#0000FF"])
#     # ax = plt.subplot(n_datasets, n_classifiers + 1, i)
#     # if ds_cnt == 0:
#     #     ax.set_title("Input data")
#     # Plot the training points
#     # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=cm)
#     # # and testing points
#     # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=10, alpha=0.6)
#     # ax.set_xlim(xx.min(), xx.max())
#     # ax.set_ylim(yy.min(), yy.max())
#     # ax.set_xticks(())
#     # ax.set_yticks(())
#     # i += 1
#     # iterate over classifiers
#     for name, clf in classifiers:
#         # ax = plt.subplot(n_datasets, n_classifiers + 1, i)
#         clf.fit(X_train, y_train)
#         # # logging.info("%s had %d nodes" % (name, clf.tree_.node_count))
#         # truc = np.empty((xx.ravel().shape[0], 2))
#         # truc[:, 0] = xx.ravel()
#         # truc[:, 1] = yy.ravel()
#
#         clf.predict_proba(X_test)
#
#         # Z = clf.predict_proba(truc)[:, 1]
#         # # score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
#         # # Put the result into a color plot
#         # Z = Z.reshape(xx.shape)
#         # ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
#         # ax.set_xlim(xx.min(), xx.max())
#         # ax.set_ylim(yy.min(), yy.max())
#         # ax.set_xticks(())
#         # ax.set_yticks(())
#         # if ds_cnt == 0:
#         #     ax.set_title(name)
#         # i += 1
#
# exit(0)


def plot_decision_classification(classifiers, datasets):
    n_classifiers = len(classifiers)
    n_datasets = len(datasets)
    h = 0.2
    fig = plt.figure(figsize=(2 * (n_classifiers + 1), 2 * n_datasets))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        ds_name, (X, y) = ds
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(n_datasets, n_classifiers + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=cm)
        # and testing points
        # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=10, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        # iterate over classifiers
        for name, clf in classifiers:
            ax = plt.subplot(n_datasets, n_classifiers + 1, i)
            clf.fit(X_train, y_train)
            # logging.info("%s had %d nodes" % (name, clf.tree_.node_count))
            truc = np.empty((xx.ravel().shape[0], 2))
            truc[:, 0] = xx.ravel()
            truc[:, 1] = yy.ravel()

            Z = clf.predict_proba(truc)[:, 1]
            # Z = clf.predict_proba_trees(truc)[0][:, 1]

            # score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            i += 1

    plt.tight_layout()


#
tic = time()
plot_decision_classification(classifiers, datasets)
toc = time()

print("time: ", toc-tic)

plt.show()
