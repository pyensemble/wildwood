# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# streamlit run playground_tree.py

import sys
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import colorcet as cc

sys.path.extend([".", ".."])


from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split

from wildwood._binning import Binner

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


n_samples = 200
# n_samples = 200
data_random_state = 123
random_state = 42


clf_kwargs = {"n_estimators": 10, "min_samples_split": 2, "random_state": random_state}

grid_size = 200
random_state = 42
n_estimators = 20


st.title("`WildWood` playground")

# The sidebar
# st.sidebar.title("Dataset")
# st.sidebar.markdown("Choose the dataset below")
# dataset = st.sidebar.selectbox("dataset", ["moons"], index=0)
st.sidebar.title("Parameters")
st.sidebar.markdown(
    """You can tune below some 
hyperparameters"""
)
aggregation = st.sidebar.checkbox("aggregation", value=True)
dirichlet = st.sidebar.selectbox(
    "dirichlet", [1e-8, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 100], index=3
)
step = st.sidebar.selectbox("step", [1e-2, 1e-1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.], index=3)

# split_pure = st.sidebar.checkbox("split_pure", value=True)

tree_idx = st.sidebar.selectbox("Number of the tree", range(n_estimators), index=0)



# @st.cache
# def simulate_data():
#     X, y = make_circles(
#         n_samples=n_samples, noise=0.2, factor=0.4, random_state=data_random_state
#     )
#     return X, y


@st.cache
def simulate_data():
    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=data_random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    return X_train, y_train


X_train, y_train = simulate_data()


# n_classes = int(y_train.max() + 1)
# n_samples_train = X_train.shape[0]


@st.cache
def compute_features_range(eps=0.5):
    x_min = X_train[:, 0].min() - eps
    x_max = X_train[:, 0].max() + eps
    y_min = X_train[:, 1].min() - eps
    y_max = X_train[:, 1].max() + eps
    return x_min, x_max, y_min, y_max


x_min, x_max, y_min, y_max = compute_features_range()

# eps = 0.0
# x_min = X_train[:, 0].min() - eps
# x_max = X_train[:, 0].max() + eps
# y_min = X_train[:, 1].min() - eps
# y_max = X_train[:, 1].max() + eps
#


@st.cache
def get_data_df(X, y):
    y_color = {0: "blue", 1: "red"}
    df = pd.DataFrame(
        {"x1": X[:, 0], "x2": X[:, 1], "y": y}, columns=["x1", "x2", "y"],
    )
    df["y"] = df["y"].map(lambda y: y_color[y])
    return df


df_data = get_data_df(X_train, y_train)


@st.cache
def get_mesh(x_min, x_max, y_min, y_max, grid_size):
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )
    xy = np.array([xx.ravel(), yy.ravel()]).T
    xy = np.ascontiguousarray(xy, dtype="float32")
    return xy


xy = get_mesh(x_min, x_max, y_min, y_max, grid_size)


def fit_forest(X_train, y_train, n_estimators=10, dirichlet=0.5, step=1.0):
    clf_kwargs = {
        "n_estimators": n_estimators,
        "min_samples_split": 2,
        "random_state": random_state,
        "n_jobs": 1,
        "step": step,
        "dirichlet": dirichlet,
    }

    clf = ForestClassifier(**clf_kwargs)
    clf.fit(X_train, y_train)
    return clf


clf = fit_forest(X_train, y_train, n_estimators, dirichlet, step)


def get_tree(clf, tree_idx):
    df_tree = clf.get_nodes(tree_idx)
    # df_tree["count_0"] = df_tree["y_sum"].apply(lambda t: t[0])
    # df_tree["count_1"] = df_tree["y_sum"].apply(lambda t: t[1])
    df_tree.sort_values(by=["depth", "parent", "node_id"], inplace=True)
    max_depth = df_tree.depth.max()
    max_depth = 10
    n_nodes = df_tree.shape[0]
    x = np.zeros(n_nodes)
    x[0] = 0.5
    indexes = df_tree["node_id"].values
    df_tree["x"] = x
    df_tree["y"] = max_depth - df_tree["depth"]
    df_tree["x0"] = df_tree["x"]
    df_tree["y0"] = df_tree["y"]
    for node in range(1, n_nodes):
        index = indexes[node]
        parent = df_tree.at[index, "parent"]
        depth = df_tree.at[index, "depth"]
        left_parent = df_tree.at[parent, "left_child"]
        x_parent = df_tree.at[parent, "x"]
        if left_parent == index:
            # It's a left node
            df_tree.at[index, "x"] = x_parent - 0.5 ** (depth + 1)
        else:
            df_tree.at[index, "x"] = x_parent + 0.5 ** (depth + 1)
        df_tree.at[index, "x0"] = x_parent
        df_tree.at[index, "y0"] = df_tree.at[parent, "y"]

    # df_tree["color"] = df_tree["is_leaf"].astype("str")
    # df_tree.replace({"color": {"False": "blue", "True": "green"}}, inplace=True)

    # Compute the decision function of all the trees and of the whole forest
    zz = clf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
    zz_trees = clf.predict_proba_trees(xy)

    # zz = clf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
    # zz = None

    return zz, df_tree, clf, zz_trees


# @st.cache
# def fit_and_get_tree(use_aggregation, n_estimators, dirichlet, step):
#     # TODO: add a progress bar
#     clf = ForestBinaryClassifier(
#         n_estimators=n_estimators,
#         random_state=random_state,
#         # use_aggregation=use_aggregation,
#         # split_pure=split_pure,
#         dirichlet=dirichlet,
#         step=step,
#     )
#     clf.fit(X_train, y_train)
#
#     # print("clf.is_categorical_):", clf.is_categorical_)
#
#     df_tree = clf.get_nodes(0)
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
#     # df_tree["color"] = df_tree["is_leaf"].astype("str")
#     # df_tree.replace({"color": {"False": "blue", "True": "green"}}, inplace=True)
#
#     # Compute the decision function
#     zz = clf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
#     # zz = None
#
#     return zz, df_tree, clf


zz, df_tree, clf, zz_trees = get_tree(clf, tree_idx)


# df_tree

# zz = zzs[iteration - 1]
# df_data_current = df_datas[iteration - 1]
# df_tree = df_trees[iteration - 1]

source_tree = ColumnDataSource(ColumnDataSource.from_df(df_tree))

source_data = ColumnDataSource(ColumnDataSource.from_df(df_data))

# print(zz)

source_decision = ColumnDataSource(data={"image": [zz]})


source_decision_tree = ColumnDataSource(
    data={"image": [zz_trees[tree_idx][:, 1].reshape(grid_size, grid_size)]}
)


# TODO: Use max_depth
plot_tree = figure(
    plot_width=800, plot_height=400, x_range=[-0.1, 1.1], y_range=[0, 11],
)

plot_tree.outline_line_color = None
plot_tree.axis.visible = False
plot_tree.grid.visible = False


circles = plot_tree.circle(
    x="x",
    y="y",
    size=10,
    # fill_color="color",
    name="circles",
    fill_alpha=0.4,
    source=source_tree,
)

plot_tree.segment(
    x0="x",
    y0="y",
    x1="x0",
    y1="y0",
    line_color="#151515",
    line_alpha=0.4,
    source=source_tree,
)

tree_hover = HoverTool(
    renderers=[circles],
    tooltips=[
        ("index", "@node_id"),
        ("depth", "@depth"),
        ("parent", "@parent"),
        ("left_child", "@left_child"),
        ("right_child", "@right_child"),
        ("is_leaf", "@is_leaf"),
        ("feature", "@feature"),
        ("impurity", "@impurity"),
        ("bin_threshold", "@bin_threshold"),
        ("loss_valid", "@loss_valid"),
        ("log_weight_tree", "@log_weight_tree"),
        ("n_samples_train", "@n_samples_train"),
        ("n_samples_valid", "@n_samples_valid"),
        ("start_train", "@start_train"),
        ("end_train", "@end_train"),
        ("start_valid", "@start_valid"),
        ("end_valid", "@end_valid"),

        # ("weight", "@weight"),
        # ("log_weight_tree", "@log_weight_tree"),
        # ("count_0", "@count_0"),
        # ("count_1", "@count_1"),
        # ("memorized", "@memorized"),
    ],
)


plot_tree.add_tools(tree_hover)
plot_tree.text(x="x", y="y", text="node_id", source=source_tree)

# plot_data = figure(
#     plot_width=500, plot_height=500,
#     x_range=[x_min_binned, x_max_binned], y_range=[y_min_binned, y_max_binned]
# )

plot_data = figure(
    plot_width=500, plot_height=500, x_range=[x_min, x_max], y_range=[y_min, y_max]
)

plot_tree_decision = figure(
    plot_width=500, plot_height=500, x_range=[x_min, x_max], y_range=[y_min, y_max]
)


plot_data.image(
    "image",
    source=source_decision,
    # x=0, y=0, dw=1, dh=1,
    x=x_min,
    y=y_min,
    dw=x_max - x_min,
    dh=y_max - y_min,
    palette=cc.CET_D1A,
)


plot_tree_decision.image(
    "image",
    source=source_decision_tree,
    # x=0, y=0, dw=1, dh=1,
    x=x_min,
    y=y_min,
    dw=x_max - x_min,
    dh=y_max - y_min,
    palette=cc.CET_D1A,
)


circles_data = plot_data.circle(
    x="x1",
    y="x2",
    size=10,
    color="y",
    line_width=2,
    line_color="black",
    name="circles",
    alpha=0.7,
    source=source_data,
)


def show_samples_in_node(node_id):
    tree_id = 0
    partition_train = clf.trees[tree_id]._tree_context.partition_train
    partition_valid = clf.trees[tree_id]._tree_context.partition_valid

    start_train = df_tree.loc[node_id, "start_train"]
    end_train = df_tree.loc[node_id, "end_train"]
    start_valid = df_tree.loc[node_id, "start_valid"]
    end_valid = df_tree.loc[node_id, "end_valid"]

    train_indices = partition_train[start_train:end_train]
    valid_indices = partition_valid[start_valid:end_valid]

    df_data_train = df_data.loc[train_indices].copy()
    df_data_train["type"] = "green"
    df_data_valid = df_data.loc[valid_indices].copy()
    df_data_valid["type"] = "pink"

    out = pd.concat([df_data_train, df_data_valid], axis="index")
    print(out)
    return out

    # return df_data.loc[train_indices], df_data.loc[valid_indices]

    # print(start_train, end_train)
    # print(clf.trees[0]._tree_context.partition_train)
    # print(clf.trees[0]._tree_context.partition_valid)
    #
    # pass


st.bokeh_chart(plot_tree)

node_id = st.text_input(label="node_id", value=0)
node_id = int(node_id)
st.markdown("Showing stuff about node number {node_id}".format(node_id=node_id))


df_node_data = show_samples_in_node(node_id)


source_node_data = ColumnDataSource(ColumnDataSource.from_df(df_node_data))


circles_node_data = plot_data.circle(
    x="x1",
    y="x2",
    size=10,
    color="y",
    line_width=3,
    line_color="type",
    name="circles",
    alpha=0.7,
    source=source_node_data,
)


# show_samples_in_node(0)


plot_data.outline_line_color = None
plot_data.grid.visible = False
plot_data.axis.visible = False

st.bokeh_chart(plot_data)

st.bokeh_chart(plot_tree_decision)


"""This small demo illustrates the internal tree construction performed by  
`WildWood` for binary classification.
"""
