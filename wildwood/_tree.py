# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This contains all the data structures for a tree together with prediction functions
of a tree.
"""

from math import exp
import numpy as np

from numba import (
    jit,
    boolean,
    uint8,
    int32,
    intp,
    uintp,
    float32,
    void,
    optional,
)
from numba.experimental import jitclass

from ._utils import get_type, resize
from ._node import node_type, node_dtype
from ._split import is_bin_in_partition

IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0
TREE_LEAF = intp(-1)
TREE_UNDEFINED = intp(-2)

NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False

tree_type = [
    # Number of features
    ("n_features", uintp),
    #
    # random_state used for seeding numpy from numba. Note that numba uses its own
    # random_state, so that we need to seed numpy.random.seed() within jit-compiled code
    ("random_state", int32),
    #
    # Maximum depth allowed in the tree
    ("max_depth", uintp),
    #
    # Number of nodes in the tree
    ("node_count", uintp),
    #
    # Maximum number of nodes storable in the tree
    ("capacity", uintp),
    #
    # A numpy array containing the nodes data
    ("nodes", node_type[::1]),
    #
    # A numpy array containing the bin partitions of nodes using a split over a
    #  categorical feature. A node has `bin_partition_start` and
    #  `bin_partition_end` attributes, its bin partition is given by
    #  `bin_partitions[bin_partition_start:bin_partition_end]`
    ("bin_partitions", uint8[::1]),
    #
    # Size of bin_partitions
    ("bin_partitions_capacity", uintp),
    #
    # Actual size of bin_partitions
    ("bin_partitions_end", uintp),
]

tree_classifier_type = [
    *tree_type,
    #
    # Number of classes
    ("n_classes", uintp),
    #
    # categorical split strategy
    ("cat_split_strategy", uint8),
    #
    # The predictions of each node in the tree with shape (n_nodes, n_classes)
    ("y_pred", float32[:, ::1]),
]


tree_regressor_type = [
    *tree_type,
    #
    # The predictions of each node in the tree with shape (n_nodes,)
    ("y_pred", float32[::1]),
]


@jitclass(tree_classifier_type)
class _TreeClassifier(object):
    """A tree for classification containing an array of nodes and an array for its
    predictions

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

    n_classes : int
        Number of label classes

    max_depth : int
        Maximum depth allowed in the tree (not used for now)

    node_count : int
        Number of nodes in the tree

    capacity : int
        Maximum number of nodes storable in the tree

    nodes : ndarray
        A numpy array containing the nodes data

    y_pred : ndarray
        The predictions of each node in the tree with shape (n_nodes, n_classes)

    bin_partitions : ndarray
        A numpy array containing the bin partitions of nodes using a split over a
        categorical feature. A node has `bin_partition_start` and
        `bin_partition_end` attributes, its bin partition is given by
        `bin_partitions[bin_partition_start:bin_partition_end]`

    bin_partitions_capacity : int
         Allocated size of `bin_partitions`

    bin_partitions_end : int
         Actual size of `bin_partitions`
    """

    def __init__(self, n_features, n_classes, random_state):
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.random_state = random_state
        # Seed numba's random generator
        np.random.seed(random_state)
        # Both node and prediction arrays have zero on the first axis and are resized
        # later when we know the initial capacity required for the tree
        self.nodes = np.empty(0, dtype=node_dtype)
        self.y_pred = np.empty((0, self.n_classes), dtype=np.float32)
        # for categorical features
        self.bin_partitions = np.empty(0, dtype=np.uint8)
        self.bin_partitions_capacity = 0
        self.bin_partitions_end = 0
        self.cat_split_strategy = 0


@jitclass(tree_regressor_type)
class _TreeRegressor(object):
    """A tree for regression containing an array of nodes and an array for its
    predictions

    Parameters
    ----------
    n_features : int
        Number of input features

    Attributes
    ----------
    n_features : int
        Number of input features

    max_depth :
        Maximum depth allowed in the tree (not used for now)

    node_count :
        Number of nodes in the tree

    capacity :
        Maximum number of nodes storable in the tree

    nodes : ndarray
        A numpy array containing the nodes data

    y_pred : ndarray
        The predictions of each node in the tree with shape (n_nodes,)

    bin_partitions : ndarray
        A numpy array containing the bin partitions of nodes using a split over a
        categorical feature. A node has `bin_partition_start` and
        `bin_partition_end` attributes, its bin partition is given by
        `bin_partitions[bin_partition_start:bin_partition_end]`

    bin_partitions_capacity : int
         Allocated size of `bin_partitions`

    bin_partitions_end : int
         Actual size of `bin_partitions`
    """

    def __init__(self, n_features, random_state):
        self.n_features = n_features
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.random_state = random_state
        # Seed numba's random generator...
        np.random.seed(random_state)
        # Both node and prediction arrays have zero on the first axis and are resized
        # later when we know the initial capacity required for the tree
        self.nodes = np.empty(0, dtype=node_dtype)
        self.y_pred = np.empty(0, dtype=np.float32)
        # bin partitions for categorical features
        self.bin_partitions = np.empty(0, dtype=np.uint8)
        self.bin_partitions_capacity = 0
        self.bin_partitions_end = 0


# Numba types for Trees
TreeClassifierType = get_type(_TreeClassifier)
TreeRegressorType = get_type(_TreeRegressor)


def get_nodes(tree):
    import pandas as pd

    nodes = tree.nodes
    columns = [
        "node_id",
        "parent",
        "left_child",
        "right_child",
        "is_leaf",
        "is_left",
        "depth",
        "feature",
        "threshold",
        "bin_threshold",
        "impurity",
        "loss_valid",
        "log_weight_tree",
        "n_samples_train",
        "n_samples_valid",
        "w_samples_train",
        "w_samples_valid",
        "start_train",
        "end_train",
        "start_valid",
        "end_valid",
        "is_split_categorical",
        "bin_partition_start",
        "bin_partition_end",
    ]
    df = pd.DataFrame.from_records(
        (
            tuple(node[col] for col in columns)
            for i, node in enumerate(nodes)
            if i < tree.node_count
        ),
        columns=columns,
    )
    bin_partitions = tree.bin_partitions

    col_bin_partition = []

    for node in nodes[: tree.node_count]:
        is_split_categorical = node["is_split_categorical"]
        bin_partition_start = node["bin_partition_start"]
        bin_partition_end = node["bin_partition_end"]
        bin_threshold = node["bin_threshold"]
        if is_split_categorical:
            bin_partition = bin_partitions[bin_partition_start:bin_partition_end]
            col_bin_partition.append(str(bin_partition))
        else:
            col_bin_partition.append(str(bin_threshold))

    df["bin_partition"] = col_bin_partition
    return df


def get_nodes_regressor(tree):
    nodes = get_nodes(tree)
    y_pred = tree.y_pred[: tree.node_count]
    nodes["y_pred"] = y_pred
    return nodes


def get_nodes_classifier(tree):
    nodes = get_nodes(tree)
    y_pred = tree.y_pred[: tree.node_count]
    col_scores = []
    for scores in y_pred:
        col_scores.append(np.array2string(scores, precision=2))
    nodes["y_pred"] = col_scores

    return nodes


@jit(
    [void(TreeClassifierType, uintp), void(TreeRegressorType, uintp)],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
)
def resize_tree_(tree, capacity):
    """Resizes and updates the tree to have the required capacity. This functions
    resizes the tree no matter what (no test is performed here).

    Parameters
    ----------
    tree : TreeClassifier or TreeRegressor
        The tree to be resized

    capacity : int
        The new desired capacity (maximum number of nodes it can contain) of the tree
    """
    tree.nodes = resize(tree.nodes, capacity)
    tree.y_pred = resize(tree.y_pred, capacity, zeros=True)
    tree.capacity = capacity


@jit(
    [
        void(TreeClassifierType, optional(uintp)),
        void(TreeRegressorType, optional(uintp)),
    ],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
)
def resize_tree(tree, capacity=None):
    """Resizes and updates the tree to have the required capacity if necessary. By
    default, it doubles the current capacity of the tree if no capacity is specified
    and set it to 3 if the tree is empty.

    Parameters
    ----------
    tree : TreeClassifier or TreeRegressor
        The tree to be resized

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
    [void(TreeClassifierType, uintp), void(TreeRegressorType, uintp)],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
)
def resize_tree_bin_partitions_(tree, capacity):
    """Resizes and updates `bin_partitions` ndarray of tree attribute

    Parameters
    ----------
    tree : TreeClassifierType or TreeRegressorType
        The tree whose `bin_partitions` needs to be resized

    capacity : int
        The new desired capacity of `tree.bin_partitions`.
    """
    tree.bin_partitions = resize(tree.bin_partitions, capacity)
    tree.bin_partitions_capacity = capacity


@jit(
    [
        void(TreeClassifierType, optional(uintp)),
        void(TreeRegressorType, optional(uintp)),
    ],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
)
def resize_tree_bin_partitions(tree, capacity=None):
    """Resizes and updates the tree's bin_partitions

    Parameters
    ----------
    tree : TreeClassifierType or TreeRegressorType
        The tree whose `bin_partitions` needs to be resized

    capacity : int
        The new desired capacity of `tree.bin_partitions`.
        If None, then it doubles the capacity of the tree.
    """
    if capacity is None:
        if tree.bin_partitions_capacity == 0:
            # If no capacity is specified and there is no node in the tree yet,
            # we set it to 256
            resize_tree_bin_partitions_(tree, 256)
        else:
            # If no capacity is specified we double the current capacity
            resize_tree_bin_partitions_(tree, 2 * tree.bin_partitions_capacity)
    else:
        if capacity <= tree.bin_partitions_capacity and tree.bin_partitions.size > 0:
            # If the capacity of the tree is already large enough, we no nothing
            return
        else:
            # Otherwise, we resize using the specified capacity
            resize_tree_bin_partitions_(tree, capacity)


@jit(
    [
        uintp(
            TreeClassifierType,
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
            boolean,
            optional(uint8[::1]),
            uint8,
        ),
        uintp(
            TreeRegressorType,
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
            boolean,
            optional(uint8[::1]),
            uint8,
        ),
    ],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={
        "node_idx": uintp,
        "nodes": node_type[::1],
        "node": node_type,
        "bin_partition_start": uintp,
        "bin_partition_end": uintp,
    },
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
    is_split_categorical,
    bin_partition,
    bin_partition_size,
):
    """Adds a node in the tree.

    Parameters
    ----------
    tree : TreeClassifier or TreeRegressor
        The tree in which we want to add a node

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

    is_split_categorical : bool
        True if the node split on a categorical feature, False otherwise

    bin_partition : ndarray
        Array of shape (128,) with uint8 dtype.
        Whenever the split is on a categorical features, ndarray such that the bins in
        bin_partition[:bin_partition_size] go to the left child while the others
        go to the right child. For a leaf, `bin_partition=None`

    bin_partition_size : int
        Whenever the split is on a categorical features, integer such that the bins in
        bin_partition[:bin_partition_size] go to the left child while the others
        go to the right child. For a leaf, `bin_partition_size=0`
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

    node["is_split_categorical"] = is_split_categorical

    # TODO: bin_partition_size==0 should not be possible, right ?
    if is_split_categorical and bin_partition_size > 0:
        bin_partition_start = tree.bin_partitions_end
        bin_partition_end = bin_partition_start + bin_partition_size
        if bin_partition_end > tree.bin_partitions_capacity:
            resize_tree_bin_partitions(tree, None)
        tree.bin_partitions[bin_partition_start:bin_partition_end] = bin_partition[
            :bin_partition_size
        ]
        node["bin_partition_start"] = bin_partition_start
        node["bin_partition_end"] = bin_partition_end
        tree.bin_partitions_end = bin_partition_end
    else:
        node["bin_partition_start"] = 0
        node["bin_partition_end"] = 0

    tree.node_count += 1
    return node_idx


@jit(
    [uintp(TreeClassifierType, uint8[:]), uintp(TreeRegressorType, uint8[:])],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={
        "nodes": node_type[::1],
        "idx_leaf": uintp,
        "node": node_type,
        "xif_in_partition": boolean,
        "bin_partitions": uint8[::1],
        "bin_partition": uint8[::1],
    },
)
def find_leaf(tree, xi):
    """Find the leaf index containing the given features vector.

    Parameters
    ----------
    tree : TreeClassifier or TreeRegressor
         The tree

    xi : ndarray
        Array of input features with shape (n_features,) and uint8 dtype

    Returns
    -------
    output : int
        Index of the leaf node containing the input features vector
    """
    leaf_idx = 0
    nodes = tree.nodes
    node = nodes[leaf_idx]
    bin_partitions = tree.bin_partitions
    while not node["is_leaf"]:
        xi_f = xi[node["feature"]]
        if node["is_split_categorical"]:
            # If the bin is on a categorical features, we use its bin_partition
            bin_partition_start = node["bin_partition_start"]
            bin_partition_end = node["bin_partition_end"]
            bin_partition = bin_partitions[bin_partition_start:bin_partition_end]
            if is_bin_in_partition(xi_f, bin_partition):
                leaf_idx = node["left_child"]
            else:
                leaf_idx = node["right_child"]
        else:
            # If the split is on a continuous feature, we use its bin_threshold
            if xi_f <= node["bin_threshold"]:
                leaf_idx = node["left_child"]
            else:
                leaf_idx = node["right_child"]
        node = nodes[leaf_idx]
    return leaf_idx


def path_leaf(tree, X):
    paths = []
    for xi in X:
        path = sample_path_leaf(tree, xi)
        paths.append(path)
    return paths


def sample_path_leaf(tree, xi):
    import pandas as pd

    leaf_idx = 0
    nodes = tree.nodes
    node = nodes[leaf_idx]
    bin_partitions = tree.bin_partitions

    col_depth = []
    col_feature = []
    col_bin_partition = []
    col_bin_threshold = []
    col_is_split_categorical = []
    col_decision = []

    while True:
        bin_partition_start = node["bin_partition_start"]
        bin_partition_end = node["bin_partition_end"]
        is_split_categorical = node["is_split_categorical"]

        col_depth.append(node["depth"])
        col_feature.append(node["feature"])
        col_is_split_categorical.append(is_split_categorical)

        xi_f = xi[node["feature"]]
        if is_split_categorical:
            # If the split is on a categorical feature
            bin_partition = bin_partitions[bin_partition_start:bin_partition_end]
            col_bin_partition.append(str(bin_partition))
            col_bin_threshold.append(None)
            if is_bin_in_partition(xi_f, bin_partition):
                leaf_idx = node["left_child"]
                decision = "left: %d in %s" % (xi_f, str(bin_partition))
            else:
                leaf_idx = node["right_child"]
                decision = "right: %d not in %s" % (xi_f, str(bin_partition))
        else:
            # If the split is on a continuous feature
            if xi_f <= node["bin_threshold"]:
                leaf_idx = node["left_child"]
                decision = "left: %d <= %d" % (xi_f, node["bin_threshold"])
            else:
                leaf_idx = node["right_child"]
                decision = "right: %d > %d" % (xi_f, node["bin_threshold"])
            col_bin_partition.append(None)
            col_bin_threshold.append(node["bin_threshold"])

        col_decision.append(decision)
        node = nodes[leaf_idx]

        if node["is_leaf"]:
            break

    df = pd.DataFrame(
        {
            "depth": col_depth,
            "feature": col_feature,
            "bin_partition": col_bin_partition,
            "bin_threshold": col_bin_threshold,
            "decision": col_decision,
            "is_split_categorical": col_is_split_categorical,
        }
    )
    return df


@jit(
    [
        uintp[::1](TreeClassifierType, uint8[:, :]),
        uintp[::1](TreeRegressorType, uint8[:, :]),
    ],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={"n_samples": uintp, "out": uintp[::1], "i": uintp, "idx_leaf": uintp},
)
def tree_apply(tree, X):
    """Finds the indexes of the leaves containing each input vector of features (rows
    of the input matrix of features)

    Parameters
    ----------
    tree : TreeClassifier or TreeRegressor
        The tree

    X : ndarray
        Input matrix of features with shape (n_samples, n_features) and uint8 dtype

    Returns
    -------
    output : ndarray
        An array of shape (n_samples,) and uintp dtype containing the indexes of the
        leaves containing each input vector of features
    """
    n_samples = X.shape[0]
    out = np.zeros((n_samples,), dtype=uintp)
    for i in range(n_samples):
        idx_leaf = find_leaf(tree, X[i])
        out[i] = idx_leaf

    return out


@jit(
    float32[:, ::1](TreeClassifierType, uint8[:, :], boolean, float32),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={
        "n_samples": uintp,
        "n_classes": uintp,
        "nodes": node_type[::1],
        "y_pred": float32[:, ::1],
        "out": float32[:, ::1],
        "i": uintp,
        "idx_current": uintp,
        "pred_i": float32[::1],
        "node": node_type,
        "node_pred": float32[::1],
        "loss": float32,
        "log_weight_tree": float32,
        "alpha": float32,
    },
)
def tree_classifier_predict_proba(tree, X, aggregation, step):
    """Predicts class probabilities for the input matrix of features.

    Parameters
    ----------
    tree : TreeClassifier
        The tree

    X : ndarray
        Input matrix of features with shape (n_samples, n_features) and uint8 dtype

    aggregation : bool
        If True we predict the class probabilities using the aggregation algorithm.
        Otherwise, we simply use the prediction given by the leaf node containing the
        input features.

    step : float
        Step-size used for the computation of the aggregation weights. Used only if
        aggregation=True

    Returns
    -------
    output : ndarray
        An array of shape (n_samples, n_classes) and float32 dtype containing the
        predicted class probabilities
    """
    n_samples = X.shape[0]
    n_classes = tree.n_classes
    nodes = tree.nodes
    y_pred = tree.y_pred
    out = np.zeros((n_samples, n_classes), dtype=float32)
    for i in range(n_samples):
        # Find the leaf containing X[i]
        idx_current = find_leaf(tree, X[i])
        # Get a view to save the prediction for X[i]
        pred_i = out[i]
        # First, we get the prediction of the leaf
        pred_i[:] = y_pred[idx_current]
        if aggregation:
            while idx_current != 0:
                # Get the parent node
                idx_current = nodes[idx_current]["parent"]
                # TODO: is there a bug when node[0] has no child ?
                node = nodes[idx_current]
                # Prediction of this node
                node_pred = y_pred[idx_current]
                # logarithm of the aggregation weight
                loss = -step * node["loss_valid"]
                log_weight_tree = node["log_weight_tree"]
                alpha = 0.5 * exp(loss - log_weight_tree)
                # Context tree weighting dark magic
                pred_i[:] = alpha * node_pred + (1 - alpha) * pred_i

    return out


@jit(
    float32[:](TreeRegressorType, uint8[:, :], boolean, float32),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={
        "n_samples": uintp,
        "nodes": node_type[::1],
        "y_pred": float32[::1],
        "out": float32[::1],
        "i": uintp,
        "idx_current": uintp,
        "pred_i": float32,
        "node": node_type,
        "node_pred": float32,
        "loss": float32,
        "log_weight_tree": float32,
        "alpha": float32,
    },
)
def tree_regressor_predict(tree, X, aggregation, step):
    """Predicts the labels for the input matrix of features.

    Parameters
    ----------
    tree : TreeRegressor
        The tree

    X : ndarray
        Input matrix of features with shape (n_samples, n_features) and uint8 dtype

    aggregation : bool
        If True we predict the labels using the aggregation algorithm.
        Otherwise, we simply use the prediction given by the leaf node containing the
        input features.

    step : float
        Step-size used for the computation of the aggregation weights. Used only if
        aggregation=True

    Returns
    -------
    output : ndarray
        An array of shape (n_samples,) and float32 dtype containing the
        predicted labels
    """
    n_samples = X.shape[0]
    nodes = tree.nodes
    y_pred = tree.y_pred
    out = np.zeros(n_samples, dtype=float32)

    for i in range(n_samples):
        idx_current = find_leaf(tree, X[i])
        # First, we get the prediction of the leaf
        pred_i = y_pred[idx_current]
        if aggregation:
            while idx_current != 0:
                # Get the parent node
                idx_current = nodes[idx_current]["parent"]
                node = nodes[idx_current]
                # Prediction of this node
                node_pred = y_pred[idx_current]
                # logarithm of the aggregation weight
                loss = -step * node["loss_valid"]
                log_weight_tree = node["log_weight_tree"]
                # Compute the aggregation weight for this subtree
                alpha = 0.5 * exp(loss - log_weight_tree)
                # Context tree weighting dark magic
                pred_i = alpha * node_pred + (1 - alpha) * pred_i

        out[i] = pred_i

    return out


# TODO: code also a tree_classifier_weighted_depth or change the apply function with
#  an aggregation option


@jit(
    float32[:](TreeRegressorType, uint8[:, :], float32),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={
        "n_samples": uintp,
        "nodes": node_type[::1],
        "out": float32[::1],
        "i": uintp,
        "idx_current": uintp,
        "node": node_type,
        "weighted_depth": float32,
        "node_pred": float32,
        "loss": float32,
        "log_weight_tree": float32,
        "alpha": float32,
    },
)
def tree_regressor_weighted_depth(tree, X, step):
    """Compute the weighted depth used by the aggregation algorithm for the
    input matrix of features.

    Parameters
    ----------
    tree : TreeRegressor
        The tree

    X : ndarray
        Input matrix of features with shape (n_samples, n_features) and uint8 dtype

    step : float
        Step-size used for the computation of the aggregation weights. Used only if
        aggregation=True

    Returns
    -------
    output : ndarray
        An array of shape (n_samples,) and float32 dtype containing the
        predicted labels
    """
    n_samples = X.shape[0]
    nodes = tree.nodes
    out = np.zeros(n_samples, dtype=float32)
    for i in range(n_samples):
        idx_current = find_leaf(tree, X[i])
        node = nodes[idx_current]
        weighted_depth = float32(node["depth"])
        while idx_current != 0:
            idx_current = nodes[idx_current]["parent"]
            node = nodes[idx_current]
            depth_new = node["depth"]
            loss = -step * node["loss_valid"]
            log_weight_tree = node["log_weight_tree"]
            alpha = 0.5 * exp(loss - log_weight_tree)
            weighted_depth = alpha * depth_new + (1 - alpha) * weighted_depth
        out[i] = weighted_depth

    return out
