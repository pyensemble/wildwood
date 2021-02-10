# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# streamlit run playground_forest.py

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import colorcet as cc
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    make_classification,
)
from sklearn.metrics import roc_auc_score, log_loss

sys.path.extend([".", ".."])

from wildwood.forest import ForestBinaryClassifier


@st.cache
def get_mesh(X, h=0.02, padding=0.5):
    """Build a regular meshgrid using the range of the features in X
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_mesh = np.array([xx.ravel(), yy.ravel()]).T
    return xx, yy, X_mesh


def plot_scatter_binary_classif(
    ax,
    xx,
    yy,
    X,
    y,
    s=10,
    alpha=None,
    cm=None,
    title=None,
    fontsize=None,
    lw=None,
    norm=None,
    noaxes=False,
):
    if cm is None:
        cm = plt.get_cmap("RdBu")

    ax.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap=cm, alpha=alpha, lw=lw, norm=norm)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if noaxes:
        ax.axis("off")


def plot_contour_binary_classif(
    ax, xx, yy, Z, cm=None, alpha=0.8, levels=200, title=None, score=None, norm=None
):
    if cm is None:
        cm = plt.get_cmap("RdBu")
    ax.contourf(xx, yy, Z, cmap=cm, alpha=alpha, levels=levels, norm=norm)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title)
    if score is not None:
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )


@st.cache
def simulate_data(dataset, random_state=42):
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)
    elif dataset == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state
        )
    elif dataset == "linear":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=random_state,
            n_clusters_per_class=1,
            flip_y=0.001,
            class_sep=2.0,
        )
        rng = np.random.RandomState(random_state)
        X += 2 * rng.uniform(size=X.shape)
    elif dataset == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=5, random_state=random_state)
        y[y == 2] = 0
        y[y == 3] = 1
        y[y == 4] = 0
    else:
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    return X, y, X_train, y_train, X_test, y_test


@st.cache
def compute_features_range(X_train, eps=0.5):
    x_min = X_train[:, 0].min() - eps
    x_max = X_train[:, 0].max() + eps
    y_min = X_train[:, 1].min() - eps
    y_max = X_train[:, 1].max() + eps
    return x_min, x_max, y_min, y_max


def fit_forest(
    X_train,
    y_train,
    aggregation=True,
    n_estimators=10,
    dirichlet=0.5,
    step=1.0,
    min_samples_split=2,
    n_jobs=1,
):

    clf_kwargs = {
        "n_estimators": n_estimators,
        "aggregation": aggregation,
        "min_samples_split": min_samples_split,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "step": step,
        "dirichlet": dirichlet,
    }
    clf = ForestBinaryClassifier(**clf_kwargs)
    clf.fit(X_train, y_train)
    return clf


# x_min, x_max, y_min, y_max = compute_features_range()


# @st.cache
# def get_data_df(X, y):
#     y_color = {0: "blue", 1: "red"}
#     df = pd.DataFrame(
#         {"x1": X[:, 0], "x2": X[:, 1], "y": y}, columns=["x1", "x2", "y"],
#     )
#     df["y"] = df["y"].map(lambda y: y_color[y])
#     return df


# df_data = get_data_df(X_train, y_train)


# @st.cache
# def get_mesh(x_min, x_max, y_min, y_max, grid_size):
#     xx, yy = np.meshgrid(
#         np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
#     )
#     xy = np.array([xx.ravel(), yy.ravel()]).T
#     xy = np.ascontiguousarray(xy, dtype="float32")
#     return xy


# def get_tree(clf, tree_idx):
#     df_tree = clf.get_nodes(tree_idx)
#     # df_tree["count_0"] = df_tree["y_sum"].apply(lambda t: t[0])
#     # df_tree["count_1"] = df_tree["y_sum"].apply(lambda t: t[1])
#     df_tree.sort_values(by=["depth", "parent", "node_id"], inplace=True)
#     max_depth = df_tree.depth.max()
#     max_depth = 10
#     n_nodes = df_tree.shape[0]
#     x = np.zeros(n_nodes)
#     x[0] = 0.5
#     indexes = df_tree["node_id"].values
#     df_tree["x"] = x
#     df_tree["y"] = max_depth - df_tree["depth"]
#     df_tree["x0"] = df_tree["x"]
#     df_tree["y0"] = df_tree["y"]
#     for node in range(1, n_nodes):
#         index = indexes[node]
#         parent = df_tree.at[index, "parent"]
#         depth = df_tree.at[index, "depth"]
#         left_parent = df_tree.at[parent, "left_child"]
#         x_parent = df_tree.at[parent, "x"]
#         if left_parent == index:
#             # It's a left node
#             df_tree.at[index, "x"] = x_parent - 0.5 ** (depth + 1)
#         else:
#             df_tree.at[index, "x"] = x_parent + 0.5 ** (depth + 1)
#         df_tree.at[index, "x0"] = x_parent
#         df_tree.at[index, "y0"] = df_tree.at[parent, "y"]
#
#     df_tree["color"] = df_tree["is_leaf"].astype("str")
#     df_tree.replace({"color": {"False": "blue", "True": "green"}}, inplace=True)
#
#     # Compute the decision function of all the trees and of the whole forest
#     zz = clf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
#     zz_trees = clf.predict_proba_trees(xy)
#
#     # zz = clf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
#     # zz = None
#
#     return zz, df_tree, clf, zz_trees


# def plot_decision_classification(clf, datasets):
#     n_classifiers = len(classifiers)
#     n_datasets = len(datasets)
#     h = 0.2
#     fig = plt.figure(figsize=(2 * (n_classifiers + 1), 2 * n_datasets))
#     i = 1
#     # iterate over datasets
#     for ds_cnt, ds in enumerate(datasets):
#         # preprocess dataset, split into training and test part
#         ds_name, (X, y) = ds
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.4, random_state=42
#         )
#
#         x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
#         y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#         # just plot the dataset first
#         cm = plt.cm.RdBu
#         cm_bright = ListedColormap(["#FF0000", "#0000FF"])
#         ax = plt.subplot(n_datasets, n_classifiers + 1, i)
#         if ds_cnt == 0:
#             ax.set_title("Input data")
#         # Plot the training points
#         # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=cm)
#         # and testing points
#         # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=10, alpha=0.6)
#         ax.set_xlim(xx.min(), xx.max())
#         ax.set_ylim(yy.min(), yy.max())
#         ax.set_xticks(())
#         ax.set_yticks(())
#         i += 1
#         # iterate over classifiers
#         for name, clf in classifiers:
#             ax = plt.subplot(n_datasets, n_classifiers + 1, i)
#             clf.fit(X_train, y_train)
#             # logging.info("%s had %d nodes" % (name, clf.tree_.node_count))
#             truc = np.empty((xx.ravel().shape[0], 2))
#             truc[:, 0] = xx.ravel()
#             truc[:, 1] = yy.ravel()
#
#             Z = clf.predict_proba(truc)[:, 1]
#             # Z = clf.predict_proba_trees(truc)[0][:, 1]
#
#             # score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
#             # Put the result into a color plot
#             Z = Z.reshape(xx.shape)
#             ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
#             ax.set_xlim(xx.min(), xx.max())
#             ax.set_ylim(yy.min(), yy.max())
#             ax.set_xticks(())
#             ax.set_yticks(())
#             if ds_cnt == 0:
#                 ax.set_title(name)
#             i += 1
#
#     plt.tight_layout()
#
#     return fig

# @st.cache
def get_decision(clf, X_mesh):
    zz = clf.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
    return zz


@st.cache
def get_normalizer(normalize):
    if normalize:
        return plt.Normalize(vmin=0.0, vmax=1.0)
    else:
        return None


st.title("`WildWood` playground")
st.sidebar.title("Dataset")
st.sidebar.markdown("Choose the dataset below")
dataset = st.sidebar.selectbox(
    "dataset", ["moons", "circles", "linear", "blobs"], index=0
)
st.sidebar.title("Parameters")
st.sidebar.markdown(
    """You can tune below some 
hyperparameters"""
)
n_estimators = st.sidebar.selectbox("n_estimators", [1, 5, 10, 50, 100], index=2)
aggregation = st.sidebar.checkbox("aggregation", value=True)
dirichlet = st.sidebar.selectbox(
    "dirichlet", [1e-8, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 100], index=3
)
step = st.sidebar.selectbox(
    "step", [1e-2, 1e-1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0], index=3
)
show_data = st.sidebar.checkbox("Show data", value=True)
normalize = st.sidebar.checkbox("Normalize colors", value=True)


n_samples = 200
random_state = 42
grid_size = 200
levels = 20


X, y, X_train, y_train, X_test, y_test = simulate_data(dataset, random_state=42)
clf = fit_forest(
    X_train,
    y_train,
    aggregation,
    n_estimators,
    dirichlet,
    step,
    n_jobs=-1
)
xx, yy, X_mesh = get_mesh(X)
zz = get_decision(clf, X_mesh)
norm = get_normalizer(normalize)

y_pred = clf.predict_proba(X_test)[:, 1]


score_test = log_loss(y_test, y_pred)

# score_test = roc_auc_score(y_test, y_pred)

_ = plt.figure(figsize=(3, 3))
ax = plt.subplot(1, 1, 1)
plot_contour_binary_classif(ax, xx, yy, zz, levels=levels, norm=norm, score=score_test)
if show_data:
    plot_scatter_binary_classif(ax, xx, yy, X_train, y_train, s=5, lw=1, norm=norm)

plt.tight_layout()
st.pyplot()


"""This small demo illustrates the effet of some hyper-parameters involved in 
`WildWood` for binary classification on toy examples.
"""
