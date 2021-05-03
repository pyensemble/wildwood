import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show

from sklearn.preprocessing import OneHotEncoder

from wildwood import ForestClassifier


def _compute_display_tree(clf, tree_idx, max_depth=None):
    df_tree = clf.get_nodes(tree_idx)
    df_tree.sort_values(by=["depth", "parent", "node_id"], inplace=True)
    if max_depth is None:
        max_depth = df_tree.depth.max()

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

    return df_tree


def plot_tree(clf, tree_idx=0, max_depth=None, width=800, height=500, attributes=None):

    if attributes is None:
        attributes = [
            "index",
            "depth",
            "parent",
            "left_child",
            "right_child",
            "is_leaf",
            "feature",
            "impurity",
            "bin_threshold",
            "bin_partition",
            "loss_valid",
            "log_weight_tree",
            "n_samples_train",
            "n_samples_valid",
            "start_train",
            "end_train",
            "start_valid",
            "end_valid",
            "y_pred",
            "weight",
            "log_weight_tree",
        ]

    df_tree = _compute_display_tree(clf, tree_idx, max_depth=None)
    source_tree = ColumnDataSource(ColumnDataSource.from_df(df_tree))

    # TODO: Use max_depth in y_range ?
    fig = figure(
        plot_width=width, plot_height=height, x_range=[-0.1, 1.1], y_range=[0, 11],
    )

    fig.outline_line_color = None
    fig.axis.visible = False
    fig.grid.visible = False

    circles = fig.circle(
        x="x",
        y="y",
        size=10,
        # fill_color="color",
        name="circles",
        fill_alpha=0.4,
        source=source_tree,
    )

    fig.segment(
        x0="x",
        y0="y",
        x1="x0",
        y1="y0",
        line_color="#151515",
        line_alpha=0.4,
        source=source_tree,
    )

    tooltips = [(attribute, "@" + attribute) for attribute in attributes]

    tree_hover = HoverTool(renderers=[circles], tooltips=tooltips)
    fig.add_tools(tree_hover)
    fig.text(x="x", y="y", text="node_id", source=source_tree)
    return fig


if __name__ == "__main__":
    X = np.repeat(np.arange(5), 20).reshape((-1, 1))
    y = np.repeat([1, 0, 0, 1, 0], 20)
    clf = ForestClassifier(
        n_estimators=1, random_state=42, categorical_features=[True], dirichlet=0.0
    )
    # X_onehot = OneHotEncoder(sparse=False).fit_transform(X)
    clf.fit(X, y)
    df = clf.get_nodes(0)
    print(df)
    fig = plot_tree(clf, 0)

    show(fig)
