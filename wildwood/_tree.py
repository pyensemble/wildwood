# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This contains all the data structures for holding tree data.
"""

from math import exp
import numpy as np

from numba import (
    from_dtype,
    njit,
    jit,
    boolean,
    uint8,
    intp,
    uintp,
    float32,
    void,
    optional,
    generated_jit,
)
from numba.experimental import jitclass

from ._utils import max_size_t, get_type, resize, resize2d, log_sum_2_exp

from ._node import node_type, node_dtype


# TODO: on a vraiment besoin de tout ca dans un stack_record ?


IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0

TREE_LEAF = intp(-1)
TREE_UNDEFINED = intp(-2)

# TODO: replace n_classes by pred_size ?

tree_type = [
    # Number of features
    ("n_features", uintp),
    # Number of classes
    ("n_classes", uintp),
    # Maximum depth allowed in the tree
    ("max_depth", uintp),
    # Number of nodes in the tree
    ("node_count", uintp),
    # ???
    ("capacity", uintp),
    # A numpy array containing the nodes data
    ("nodes", node_type[::1]),
    # This array contains values allowing to compute the prediction of each node
    # Its shape is (n_nodes, n_outputs, max_n_classes)
    # TODO: IMPORTANT a priori ca serait mieux ::1 sur le premier axe mais l'init
    #  avec shape (0, ., .) foire dans ce cas avec numba
    ("y_pred", float32[:, ::1]),
]


# TODO: pas sur que ca soit utile en fait values avec cette strategie histogramme ?


@jitclass(tree_type)
class Tree(object):
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        # Both values and nodes arrays have zero on the first axis and are resized
        # later when we know the capacity of the tree
        # The array of nodes contained in the tree
        self.nodes = np.empty(0, dtype=node_dtype)
        # The array of y sums or counts for each node
        self.y_pred = np.empty((0, self.n_classes), dtype=float32)


TreeType = get_type(Tree)


# @njit
# def print_tree(tree):
#     s = "-" * 64 + "\n"
#     s += "Tree("
#     s += "n_features={n_features}".format(n_features=tree.n_features)
#     s += ", n_classes={n_classes}".format(n_classes=tree.n_classes)
#     s += ", capacity={capacity}".format(capacity=tree.capacity)
#     s += ", node_count={node_count}".format(node_count=tree.node_count)
#     s += ")"
#     print(s)
#     if tree.node_count > 0:
#         print_nodes(tree)


def get_nodes(tree):
    import pandas as pd

    nodes = tree.nodes

    columns = [
        "node_id",
        "parent",
        "left_child",
        "right_child",
        "is_leaf",
        "depth",
        "feature",
        "threshold",
        "bin_threshold",
        "impurity",
        "n_samples_train",
        "n_samples_valid",
        "weighted_n_samples_train",
        "weighted_n_samples_valid",
        "start_train",
        "end_train",
        "start_valid",
        "end_valid",
        "is_left",
        "loss_valid",
        "log_weight_tree",
    ]

    # columns = [col_name for col_name, _ in np_node_tree]
    # columns = ["left_child"]

    return pd.DataFrame.from_records(
        (
            tuple(node[col] for col in columns)
            for i, node in enumerate(nodes)
            if i < tree.node_count
        ),
        columns=columns,
    )


@jit(void(TreeType, uintp), nopython=True, nogil=True)
def resize_tree_(tree, capacity):
    """Resizes and updates the tree to have the required capacity. This functions
    resizes the tree no matter what (no test is performed here).

    Parameters
    ----------
    tree : TreeType
        The tree

    capacity : int
        The new desired capacity (maximum number of nodes it can contain) of the tree.
    """
    tree.nodes = resize(tree.nodes, capacity)
    tree.y_pred = resize2d(tree.y_pred, capacity, zeros=True)
    tree.capacity = capacity


@jit(void(TreeType, optional(uintp)), nopython=True, nogil=True)
def resize_tree(tree, capacity=None):
    """Resizes and updates the tree to have the required capacity. By default,
    it doubles the current capacity of the tree if no capacity is specified.

    Parameters
    ----------
    tree : TreeType
        The tree

    capacity : int or None
        The new desired capacity (maximum number of nodes it can contain) of the tree.
        If None, then it doubles the capacity of the tree.
    """
    if capacity is None:
        if tree.capacity == 0:
            # If no capacity is specified and there is no node in the tree yet,
            # we set it to 3
            resize_tree_(tree, 3)
        else:
            # If no capacity is specified we double the current capacity
            resize_tree_(tree, 2 * tree.capacity)
    else:
        if capacity <= tree.capacity and tree.nodes.size > 0:
            # If the capacity of the tree is already large enough, we no nothing
            return
        else:
            # Otherwise, we resize using the specified capacity
            resize_tree_(tree, capacity)


@jit(
    uintp(
        TreeType,
        intp,
        uintp,
        boolean,
        boolean,
        uintp,
        float32,
        uint8,
        float32,
        uintp,
        uintp,
        float32,
        float32,
        uintp,
        uintp,
        uintp,
        uintp,
        float32,
    ),
    nopython=True,
    nogil=True,
)
def add_node_tree(
    tree,
    parent,
    depth,
    is_left,
    is_leaf,
    feature,
    threshold,
    bin_threshold,
    impurity,
    n_samples_train,
    n_samples_valid,
    w_samples_train,
    w_samples_valid,
    start_train,
    end_train,
    start_valid,
    end_valid,
    loss_valid,
):
    """Adds a node in the tree

    Parameters
    ----------
    tree : TreeType
        The tree

    parent : int
        Index of the parent node

    depth : int
       Depth of the node in the tree

    is_left : bool
        True if the node is a left child, False otherwise

    is_leaf : bool
        True if the node is a leaf node, False otherwise

    feature : int
        Feature used to split the node

    threshold : float
        Continuous threshold used to split the node (not used for now)

    bin_threshold : int
        Index of the bin threshold used to split the node

    n_samples_train : int
        Number of training samples in the node

    n_samples_valid : int
        Number of validation (out-of-the-bag) samples in the node

    w_samples_train : float
        Weighted number of training samples in the node

    w_samples_valid : float
        Weighted number of validation (out-of-the-bag) samples in the node

    start_train : int
        Index of the first training sample in the node. We have that
        partition_train[start_train:end_train] contains the indexes of the node's
        training samples

    end_train : int
        End-index of the slice containing the node's training samples indexes

    start_valid : int
        Index of the first validation (out-of-the-bag) sample in the node. We have
        that partition_valid[start_valid:end_valid] contains the indexes of the
        node's validation samples

    end_valid : int
        End-index of the slice containing the node's validation samples indexes

    loss_valid : float
        Validation loss of the node, computed on validation (out-of-the-bag) samples

    """
    # New node index is given by the current number of nodes in the tree
    node_idx = tree.node_count
    if node_idx >= tree.capacity:
        resize_tree(tree, None)

    nodes = tree.nodes
    node = nodes[node_idx]
    node["node_id"] = node_idx
    node["parent"] = parent
    node["depth"] = depth
    node["is_leaf"] = is_leaf
    node["impurity"] = impurity
    node["n_samples_train"] = n_samples_train
    node["n_samples_valid"] = n_samples_valid
    node["w_samples_train"] = w_samples_train
    node["w_samples_valid"] = w_samples_valid
    node["start_train"] = start_train
    node["end_train"] = end_train
    node["start_valid"] = start_valid
    node["end_valid"] = end_valid
    node["loss_valid"] = loss_valid
    node["log_weight_tree"] = np.nan

    if parent != TREE_UNDEFINED:
        if is_left:
            nodes[parent]["left_child"] = node_idx
        else:
            nodes[parent]["right_child"] = node_idx

    if is_leaf:
        node["left_child"] = TREE_LEAF
        node["right_child"] = TREE_LEAF
        node["feature"] = TREE_UNDEFINED
        node["threshold"] = TREE_UNDEFINED
        node["bin_threshold"] = TREE_UNDEFINED
    else:
        node["feature"] = feature
        node["threshold"] = threshold
        node["bin_threshold"] = bin_threshold

    tree.node_count += uintp(1)

    return node_idx


# TODO: tous les trucs de prediction faut les faire a part comme dans pygbm,
#  on y mettra dedans l'aggregation ? Dans un module _prediction separe


# TODO: pas de jit car numba n'accepte pas l'option axis dans .take (a verifier) mais
#  peut etre qu'on s'en fout en fait ? Et puis ca sera assez rapide


@njit
def tree_predict(tree, X, aggregation, step):
    # Index of the leaves containing the samples in X (note that X has been binned by
    # the forest)

    if aggregation:
        return tree_predict_aggregate(tree, X, step)
    else:
        idx_leaves = tree_apply(tree, X)
        n_samples = X.shape[0]
        # TODO: only binary classification right ?
        out = np.empty((n_samples, 2), dtype=float32)
        # Predictions given by each leaf node of the tree
        y_pred = tree.y_pred
        i = 0
        # TODO: idx_leaves.shape[0] == n_samples
        for i in range(n_samples):
            idx_leaf = idx_leaves[i]
            out[i] = y_pred[idx_leaf]

        return out


@njit
def tree_apply(tree, X):
    # TODO: on va supposer que X est deja binnee hein ?
    return tree_apply_dense(tree, X)


@njit
def tree_apply_dense(tree, X):
    # TODO: X is assumed to be binned here
    n_samples = X.shape[0]
    out = np.zeros((n_samples,), dtype=uintp)
    nodes = tree.nodes

    for i in range(n_samples):
        # Index of the leaf containing the sample
        idx_leaf = 0
        node = nodes[idx_leaf]
        # While node not a leaf
        while not node["is_leaf"]:
            # ... and node.right_child != TREE_LEAF:
            if X[i, node["feature"]] <= node["bin_threshold"]:
                idx_leaf = node["left_child"]
            else:
                idx_leaf = node["right_child"]
            node = nodes[idx_leaf]

        # out_ptr[i] = <SIZE_t>(node - tree.nodes)  # node offset
        out[i] = uintp(idx_leaf)

    return out


import numba


@numba.jit(nopython=True, nogil=True, locals={"i": uintp, "idx_current": uintp})
def tree_predict_aggregate(tree, X, step):
    n_samples = X.shape[0]
    n_classes = tree.n_classes
    nodes = tree.nodes
    y_pred = tree.y_pred
    out = np.zeros((n_samples, n_classes), dtype=float32)

    for i in range(n_samples):
        # print(i)
        # Index of the leaf containing the sample
        # idx_current = nb_size_t(0)
        idx_current = 0
        node = nodes[idx_current]
        # While node not a leaf
        while not node["is_leaf"]:
            if X[i, node["feature"]] <= node["bin_threshold"]:
                idx_current = node["left_child"]
            else:
                idx_current = node["right_child"]
            node = nodes[idx_current]
        # Now idx_current is the index of the leaf node containing X[i]

        # The aggregated prediction of the tree is saved in out[i]
        # y_pred_tree = out[i].view()
        # We first put the predictions of the leaf
        # print(y_pred_tree.shape, y_pred[idx_current].shape)
        # print(idx_current, type(idx_current))
        # print(i, type(i))
        out[i, :] = y_pred[idx_current]
        # Go up in the tree
        idx_current = node["parent"]

        while idx_current != 0:
            # Get the current node
            node = nodes[idx_current]
            # The prediction given by the current node
            node_pred = y_pred[idx_current]
            # The aggregation weights of the current node
            log_weight = step * node["loss_valid"]
            log_weight_tree = node["log_weight_tree"]
            w = exp(log_weight - log_weight_tree)
            # Apply the dark magic recursive formula from CTW
            out[i, :] = 0.5 * w * node_pred + (1 - 0.5 * w) * out[i, :]
            # Go up in the tree
            idx_current = node["parent"]

    return out


# #
# # @njit(void(get_type(TreeClassifier), float32[::1], float32[::1], boolean))
# def tree_classifier_predict(tree, x_t, scores, use_aggregation):
#     nodes = tree.nodes
#     leaf = tree_get_leaf(tree, x_t)
#     if not use_aggregation:
#         node_classifier_predict(tree, leaf, scores)
#         return
#     current = leaf
#     # Allocate once and for all
#     pred_new = np.empty(tree.n_classes, float32)
#     while True:
#         # This test is useless ?
#         if nodes.is_leaf[current]:
#             node_classifier_predict(tree, current, scores)
#         else:
#             weight = nodes.weight[current]
#             log_weight_tree = nodes.log_weight_tree[current]
#             w = exp(weight - log_weight_tree)
#             # Get the predictions of the current node
#             node_classifier_predict(tree, current, pred_new)
#             for c in range(tree.n_classes):
#                 scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c]
#         # Root must be update as well
#         if current == 0:
#             break
#         # And now we go up
#         current = nodes.parent[current]
