# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This contains all the data structures for a tree.
"""

from math import exp
import numpy as np

from numba import (
    njit,
    jit,
    boolean,
    uint8,
    intp,
    uintp,
    float32,
    void,
    optional,
)
from numba.experimental import jitclass

from ._utils import get_type, resize, resize2d
from ._node import node_type, node_dtype


IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0
TREE_LEAF = intp(-1)
TREE_UNDEFINED = intp(-2)


tree_type = [
    # Number of features
    ("n_features", uintp),
    # Number of classes
    ("n_classes", uintp),
    # Maximum depth allowed in the tree
    ("max_depth", uintp),
    # Number of nodes in the tree
    ("node_count", uintp),
    # Maximum number of nodes storable in the tree
    ("capacity", uintp),
    # A numpy array containing the nodes data
    ("nodes", node_type[::1]),
    # The predictions of each node in the tree with shape (n_nodes, n_classes)
    ("y_pred", float32[:, ::1]),
]


@jitclass(tree_type)
class Tree(object):
    """A tree containing an array of nodes and an array for its predictions

    Parameters
    ----------
    n_features : int
        Number of input features

    n_classes : int
        Number of label classes

    Attributes
    ----------
    n_features : int
        Number of input features

    n_classes :
        Number of label classes

    max_depth :
        Maximum depth allowed in the tree (not used for now)

    node_count :
        Number of nodes in the tree

    capacity :
        Maximum number of nodes storable in the tree

    nodes : ndarray
        A numpy array containing the nodes data

    y_pred : ndarray
        The predictions of each node in the tree with shape (n_nodes, n_classes)
    """

    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        # TODO: is this useful ?
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        # Both node and prediction arrays have zero on the first axis and are resized
        # later when we know the initial capacity required for the tree
        self.nodes = np.empty(0, dtype=node_dtype)
        self.y_pred = np.empty((0, self.n_classes), dtype=float32)


# Numba type for a Tree
TreeType = get_type(Tree)


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
    locals={"node_idx": uintp, "nodes": node_type[::1], "node": node_type},
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
    """Adds a node in the tree.

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

    impurity : float
        Impurity of the node. Used to avoid to split a "pure" node (with impurity=0).

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

    tree.node_count += 1
    return node_idx


# TODO: tous les trucs de prediction faut les faire a part comme dans pygbm,
#  on y mettra dedans l'aggregation ? Dans un module _prediction separe

# TODO: pas de jit car numba n'accepte pas l'option axis dans .take (a verifier) mais
#  peut etre qu'on s'en fout en fait ? Et puis ca sera assez rapide


@jit(
    uintp[::1](TreeType, uint8[::1, :]),
    nopython=True,
    nogil=True,
    locals={"n_samples": uintp, "out": uintp[:, ::1], "idx_leaf": uintp},
)
def tree_apply_dense(tree, X):
    n_samples = X.shape[0]
    out = np.zeros((n_samples,), dtype=uintp)

    for i in range(n_samples):
        # Index of the leaf containing the sample
        idx_leaf = 0
        node = tree.nodes[idx_leaf]
        # While node not a leaf
        while not node["is_leaf"]:
            if X[i, node["feature"]] <= node["bin_threshold"]:
                idx_leaf = node["left_child"]
            else:
                idx_leaf = node["right_child"]
            node = tree.nodes[idx_leaf]

        out[i] = idx_leaf

    return out


@jit(float32[:, ::1](TreeType, uint8[::1, :]), nopython=True, nogil=True)
def tree_apply(tree, X):
    return tree_apply_dense(tree, X)


@jit(nopython=True, nogil=True, locals={"i": uintp, "idx_current": uintp})
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


@jit(
    float32[:, ::1](TreeType, uint8[::1, :], boolean, float32),
    nopython=True,
    nogil=True,
    locals={"n_samples": uintp, "out": float32[:, ::1]},
)
def tree_predict_proba(tree, X, aggregation, step):

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
