
import numpy as np
import logging
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from wildwood._utils import NP_DOUBLE_t

from wildwood._classes import DecisionTreeClassifier

n_samples = 150
random_state = 42


#
# logging.basicConfig(
#     level=logging.DEBUG, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
# )


X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5,
                    random_state=random_state)
y = y.astype(NP_DOUBLE_t)

# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# clf = DecisionTreeClassifier()
#
# print(clf)
#
# clf.fit(X, y)

# clf.predict_proba()



# print_tree(clf.tree_)
#
#
# nodes = get_nodes(clf.tree_)


def plot_decision_classification(classifiers, datasets):
    n_classifiers = len(classifiers)
    n_datasets = len(datasets)
    h = 0.2
    fig = plt.figure(figsize=(2 * (n_classifiers + 1), 2 * n_datasets))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # print('=' * 64)
        # preprocess dataset, split into training and test part
        X, y = ds
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
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=cm)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=10, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        # iterate over classifiers
        for name, clf in classifiers:
            ax = plt.subplot(n_datasets, n_classifiers + 1, i)
            clf.fit(X_train, y_train)
            truc = np.empty((xx.ravel().shape[0], 2))
            truc[:, 0] = xx.ravel()
            truc[:, 1] = yy.ravel()
            # truc = np.array([xx.ravel(), yy.ravel()]).T
            # print(truc.flags)
            Z = clf.predict_proba(truc)[:, 1]

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
            # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            #         size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()



datasets = [
    (X, y)
]

classifiers = [
    ("tree", DecisionTreeClassifier())
]

plot_decision_classification(classifiers, datasets)

plt.show()

# print(nodes)

# # print(y[:10])
#
# from wildwood._tree import Tree, print_tree, get_nodes, tree_add_node
#
# from wildwood._classes import DecisionTreeClassifier
#
#
# from wildwood._utils import Stack
#
# from wildwood._utils import SIZE_MAX


#
# @njit
# def main():
#     n_features = 12
#     n_classes = np.array([3], dtype=NP_SIZE_t)
#     n_outputs = 1
#
#     tree = Tree(n_features, n_classes, n_outputs)
#
#     parent = 0
#     is_left = 1
#     is_leaf = 1
#     feature = 11
#     threshold = 3.14
#     impurity = 2.78
#     n_node_samples = 42
#     weighted_n_node_samples = 2.32
#
#     tree_add_node(tree, parent, is_left, is_leaf, feature, threshold, impurity,
#                   n_node_samples, weighted_n_node_samples)
#     tree_add_node(tree, parent, is_left, is_leaf, feature, threshold, impurity,
#                   n_node_samples, weighted_n_node_samples)
#     tree_add_node(tree, parent, is_left, is_leaf, feature, threshold, impurity,
#                   n_node_samples, weighted_n_node_samples)
#
#     return tree
#
# tree = main()
#
# pd.set_option("display.max_columns", 20)
# pd.set_option("display.width", 150)
#
# print_tree(tree)
#
# # print(tree.nodes)
#
# print(get_nodes(tree))

# main()

# print(SIZE_MAX.max)
