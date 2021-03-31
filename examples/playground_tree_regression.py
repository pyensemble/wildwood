# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# streamlit run playground_tree.py

import sys
import streamlit as st


from matplotlib.cm import get_cmap
import logging
import numpy as np
import matplotlib.pyplot as plt

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

sys.path.extend([".", ".."])

from wildwood._binning import Binner
from wildwood.signals import get_signal, make_regression
from wildwood.forest import ForestRegressor


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

n_samples_test = 1000
random_state = 42
grid_size = 200


st.title("`WildWood` playground")
st.subheader("`ForestRegressor` on noisy signals")

# The sidebar
st.sidebar.title("Data")
st.sidebar.markdown("Simulation parameters")
signal = st.sidebar.selectbox(
    "signal", ["heavisine", "bumps", "blocks", "doppler"], index=0
)
n_samples = st.sidebar.selectbox("n_samples", [100, 1000, 5000, 50000], index=1,)
noise = st.sidebar.slider("noise", min_value=0.01, max_value=0.1, value=0.03, step=0.01)

st.sidebar.title("Parameters")
st.sidebar.markdown("""You can tune below some hyperparameters""")
aggregation = st.sidebar.checkbox("aggregation", value=True)

step = st.sidebar.text_input(label="step", value=1.0)
step = float(step)
# step = st.sidebar.selectbox(
#     "step", [1e-2, 1e-1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0], index=3,
# )
n_estimators = st.sidebar.selectbox("n_estimators", [1, 2, 5, 10, 100], index=3,)
tree_idx = st.sidebar.selectbox("Number of the tree", range(n_estimators), index=0)


@st.cache
def simulate_data(n_samples, signal, noise, random_state=42):
    X_train, y_train = make_regression(
        n_samples=n_samples, signal=signal, noise=noise, random_state=random_state
    )
    X_test = np.linspace(0, 1, num=n_samples_test).reshape(-1, 1)
    binner = Binner().fit(X_train)
    X_test_binned = binner.transform(X_test)

    return X_train, y_train, X_test, X_test_binned


# @st.cache
def train_forest(
    n_samples, signal, noise, n_estimators, aggregation, step, random_state=42
):
    X_train, y_train, X_test, X_test_binned = simulate_data(
        n_samples, signal, noise, random_state
    )

    reg = ForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        aggregation=aggregation,
        step=step,
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    weighted_depths = reg.weighted_depth(X_test.reshape(n_samples_test, 1))
    return X_train, y_train, X_test, X_test_binned, reg, y_pred, weighted_depths


# @st.cache
# def plot_weighted_depth(n_samples, signal, noise, random_state, aggregation, step):

# X_train, y_train, X_test, X_test_binned = simulate_data(
#     n_samples, signal, noise, random_state
# )


#
#
# weighted_depths = reg.weighted_depth(X_test.reshape(n_samples_test, 1))


# def plot(n_samples, signal, noise, random_state):


X_train, y_train, X_test, X_test_binned, reg, y_pred, weighted_depths = train_forest(
    n_samples, signal, noise, n_estimators, aggregation, step, random_state
)
# X_train, y_train, X_test, X_test_binned = simulate_data(
#     n_samples, signal, noise, random_state
# )
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 5))

colormap = get_cmap("tab20")
plot_samples = ax1.plot(
    X_train, y_train, color=colormap.colors[1], lw=2, label="Samples"
)[0]
plot_signal = ax1.plot(
    X_test_binned / 255,
    get_signal(X_test_binned / 255, signal),
    lw=2,
    color=colormap.colors[0],
    label="Signal",
)[0]
plot_prediction = ax2.plot(
    X_test.ravel(), y_pred, lw=2, color=colormap.colors[2], label="Prediction"
)[0]
plot_weighted_depths = ax3.plot(
    X_test, weighted_depths.T, lw=1, color=colormap.colors[5], alpha=0.2
)[0]
plot_mean_weighted_depths = ax3.plot(
    X_test,
    weighted_depths.mean(axis=0),
    lw=2,
    color=colormap.colors[4],
    label="Mean weighted depth",
)[0]
fig.subplots_adjust(hspace=0.1)

fig.legend(
    (
        plot_signal,
        plot_samples,
        plot_mean_weighted_depths,
        plot_weighted_depths,
        plot_prediction,
    ),
    (
        "Signal",
        "Samples",
        "Average weighted depths",
        "Weighted depths",
        "Prediction",
    ),
    fontsize=12,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
)

st.pyplot(fig)


#
# @st.cache
# def get_mesh(x_min, x_max, y_min, y_max, grid_size):
#     xx, yy = np.meshgrid(
#         np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
#     )
#     xy = np.array([xx.ravel(), yy.ravel()]).T
#     xy = np.ascontiguousarray(xy, dtype="float32")
#     return xy
#
#
# xy = get_mesh(x_min, x_max, y_min, y_max, grid_size)
#
#
# def fit_forest(X_train, y_train, n_estimators=10, dirichlet=0.5, step=1.0):
#     clf_kwargs = {
#         "n_estimators": n_estimators,
#         "min_samples_split": 2,
#         "random_state": random_state,
#         "n_jobs": 1,
#         "step": step,
#         "dirichlet": dirichlet,
#     }
#
#     clf = ForestClassifier(**clf_kwargs)
#     clf.fit(X_train, y_train)
#     return clf
#
#
# # clf = fit_forest(X_train, y_train, n_estimators, dirichlet, step)
#
#
def get_tree(reg, tree_idx):
    df_tree = reg.get_nodes(tree_idx)
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
    # zz = clf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
    # zz_trees = clf.predict_proba_trees(xy)

    # zz = clf.predict_proba(xy)[:, 1].reshape(grid_size, grid_size)
    # zz = None
    # , clf, zz_trees
    return df_tree

df_tree = get_tree(reg, tree_idx)

source_tree = ColumnDataSource(ColumnDataSource.from_df(df_tree))

# # TODO: Use max_depth
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
        # ("feature", "@feature"),
        ("impurity", "@impurity"),
        ("bin_threshold", "@bin_threshold"),
        ("loss_valid", "@loss_valid"),
        ("log_weight_tree", "@log_weight_tree"),
        # ("n_samples_train", "@n_samples_train"),
        # ("n_samples_valid", "@n_samples_valid"),
        # ("start_train", "@start_train"),
        # ("end_train", "@end_train"),
        # ("start_valid", "@start_valid"),
        # ("end_valid", "@end_valid"),
        ("y_pred", "@y_pred")
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

# plot_data = figure(
#     plot_width=500, plot_height=500, x_range=[x_min, x_max], y_range=[y_min, y_max]
# )

# plot_tree_decision = figure(
#     plot_width=500, plot_height=500, x_range=[x_min, x_max], y_range=[y_min, y_max]
# )

#
# plot_data.image(
#     "image",
#     source=source_decision,
#     # x=0, y=0, dw=1, dh=1,
#     x=x_min,
#     y=y_min,
#     dw=x_max - x_min,
#     dh=y_max - y_min,
#     palette=cc.CET_D1A,
# )
#
#
# plot_tree_decision.image(
#     "image",
#     source=source_decision_tree,
#     # x=0, y=0, dw=1, dh=1,
#     x=x_min,
#     y=y_min,
#     dw=x_max - x_min,
#     dh=y_max - y_min,
#     palette=cc.CET_D1A,
# )
#
#
# circles_data = plot_data.circle(
#     x="x1",
#     y="x2",
#     size=10,
#     color="y",
#     line_width=2,
#     line_color="black",
#     name="circles",
#     alpha=0.7,
#     source=source_data,
# )
#
#
# def show_samples_in_node(node_id):
#     tree_id = 0
#     partition_train = clf.trees[tree_id]._tree_context.partition_train
#     partition_valid = clf.trees[tree_id]._tree_context.partition_valid
#
#     start_train = df_tree.loc[node_id, "start_train"]
#     end_train = df_tree.loc[node_id, "end_train"]
#     start_valid = df_tree.loc[node_id, "start_valid"]
#     end_valid = df_tree.loc[node_id, "end_valid"]
#
#     train_indices = partition_train[start_train:end_train]
#     valid_indices = partition_valid[start_valid:end_valid]
#
#     df_data_train = df_data.loc[train_indices].copy()
#     df_data_train["type"] = "green"
#     df_data_valid = df_data.loc[valid_indices].copy()
#     df_data_valid["type"] = "pink"
#
#     out = pd.concat([df_data_train, df_data_valid], axis="index")
#     print(out)
#     return out
#
#     # return df_data.loc[train_indices], df_data.loc[valid_indices]
#
#     # print(start_train, end_train)
#     # print(clf.trees[0]._tree_context.partition_train)
#     # print(clf.trees[0]._tree_context.partition_valid)
#     #
#     # pass
#
#

st.bokeh_chart(plot_tree)

#
# node_id = st.text_input(label="node_id", value=0)
# node_id = int(node_id)
# st.markdown("Showing stuff about node number {node_id}".format(node_id=node_id))
#
#
# df_node_data = show_samples_in_node(node_id)
#
#
# source_node_data = ColumnDataSource(ColumnDataSource.from_df(df_node_data))
#
#
# circles_node_data = plot_data.circle(
#     x="x1",
#     y="x2",
#     size=10,
#     color="y",
#     line_width=3,
#     line_color="type",
#     name="circles",
#     alpha=0.7,
#     source=source_node_data,
# )
#
#
# # show_samples_in_node(0)
#
#
# plot_data.outline_line_color = None
# plot_data.grid.visible = False
# plot_data.axis.visible = False
#
# st.bokeh_chart(plot_data)
#
# st.bokeh_chart(plot_tree_decision)
#
#
# ####
#
