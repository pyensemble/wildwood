# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains all required container types for nodes (including the
node numpy dtypes and several functions operating locally in nodes.
"""

from math import log
import numpy as np
from numba import from_dtype, jit, boolean, uint8, intp, uintp, float32, void
from numba.experimental import jitclass

from ._utils import get_type, sample_without_replacement
from ._tree_context import TreeClassifierContextType, TreeRegressorContextType


# Global jit decorator options
NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False


# This data type describes all the information saved about a node. It is used in
#  _tree.py where the TreeClassifier and TreeRegressor dataclasses contain
#  an attribute called nodes that uses this data type.
node_dtype = np.dtype(
    [
        # Index of the node in the nodes array
        ("node_id", np.uintp),
        #
        # Index of the parent node
        ("parent", np.uintp),
        #
        # Index of the left child node (it's TREE_LEAF is node is a leaf)
        ("left_child", np.intp),
        #
        # Index of the right child node (it's TREE_LEAF is node is a leaf)
        ("right_child", np.intp),
        #
        # Is the node a leaf ?
        ("is_leaf", np.bool),
        #
        # Is the node a left child ? (it's ? for the root node)
        ("is_left", np.bool),
        #
        # Depth of the node in the tree
        ("depth", np.uintp),
        #
        # Feature used for splitting the node (it's ??? if node is a leaf)
        ("feature", np.uintp),
        #
        # Continuous threshold used for splitting the node (not used for now)
        ("threshold", np.float32),
        #
        # Index of the bin threshold used for splitting the node
        ("bin_threshold", np.uint8),
        #
        # Impurity of the node. Used to avoid to split a "pure" node (with impurity=0).
        #   Note that impurity is computed using training samples only.
        ("impurity", np.float32),
        #
        # Validation loss of the node, computed on validation (out-of-the-bag) samples
        ("loss_valid", np.float32),
        #
        # Logarithm of the subtree aggregation weight
        ("log_weight_tree", np.float32),
        #
        # Number of training samples in the node
        ("n_samples_train", np.uintp),
        #
        # Number of validation (out-of-the-bag) samples in the node
        ("n_samples_valid", np.uintp),
        #
        # Weighted number of training samples in the node
        ("w_samples_train", np.float32),
        #
        # Weighted number of validation (out-of-the-bag) samples in the node
        ("w_samples_valid", np.float32),
        #
        # Index of the first training sample in the node. We have that
        #   partition_train[start_train:end_train] contains the indexes of the node's
        #   training samples
        ("start_train", np.uintp),
        #
        # End-index of the slice containing the node's training samples indexes
        ("end_train", np.uintp),
        #
        # Index of the first validation (out-of-the-bag) sample in the node. We have
        #   that partition_valid[start_valid:end_valid] contains the indexes of the
        #   node's validation samples
        ("start_valid", np.uintp),
        #
        # End-index of the slice containing the node's validation samples indexes
        ("end_valid", np.uintp),
        #
        # Is the split on a categorical feature ?
        ("is_split_categorical", np.bool),
        #
        # Whenever the split is on a categorical features: start index of the bin
        # partition for this node. The tree has a bin_partitions array,
        # and bin_partitions[bin_partition_start:bin_partition_end] contains
        # the bins that go to the left child
        ("bin_partition_start", np.uintp),
        #
        # Whenever the split is on a categorical features: end index of the bin
        # partition for this node. The tree has a bin_partitions array,
        # and bin_partitions[bin_partition_start:bin_partition_end] contains
        # the bins that go to the left child.
        ("bin_partition_end", np.uintp),
    ]
)


# Convert this dtype to a numba-usable data-type
node_type = from_dtype(node_dtype)


# TODO: many attributes are not used in the node_context ?

# A node_context contains all the data required to find the best split of the node
node_context_type = [
    # This array contains the index of all the features. This will be modified
    # inplace when sampling features to be considered for splits
    ("features_pool", uintp[::1]),
    #
    # This array will contain the features sampled uniformly at random (without
    # replacement) to be considered for splits
    ("features_sampled", uintp[::1]),
    #
    # Do we need to perform features sampling ?
    ("sample_features", boolean),
    #
    # Number of training samples in the node
    ("n_samples_train", intp),
    #
    # Number of validation (out-of-the-bag) samples in the node
    ("n_samples_valid", intp),
    #
    # Weighted number of training samples in the node
    ("w_samples_train", float32),
    #
    # Weighted number of validation (out-of-the-bag) samples in the node
    ("w_samples_valid", float32),
    #
    # Index of the first training sample in the node. We have that
    #   partition_train[start_train:end_train] contains the indexes of the node's
    #   training samples
    ("start_train", intp),
    #
    # End-index of the slice containing the node's training samples indexes
    ("end_train", intp),
    #
    # Index of the first validation (out-of-the-bag) sample in the node. We have
    #   that partition_valid[start_valid:end_valid] contains the indexes of the
    #   node's validation samples
    ("start_valid", intp),
    #
    # End-index of the slice containing the node's validation samples indexes
    ("end_valid", intp),
    #
    # Validation loss of the node, computed on validation (out-of-the-bag) samples
    ("loss_valid", float32),
    #
    # Number of training samples for each (feature, bin) in the node
    ("n_samples_train_in_bins", uintp[:, ::1]),
    #
    # Weighted number of training samples for each (feature, bin) in the node
    ("w_samples_train_in_bins", float32[:, ::1]),
    #
    # Number of validation samples for each (feature, bin) in the node
    ("n_samples_valid_in_bins", uintp[:, ::1]),
    #
    # Weighted number of validation samples for each (feature, bin) in the node
    ("w_samples_valid_in_bins", float32[:, ::1]),
    #
    # A vector that counts the number of non-empty bins for each feature on train data.
    ("non_empty_bins_train_count", uint8[::1]),
    #
    # A vector that counts the number of non-empty bins for each feature on valid data.
    ("non_empty_bins_valid_count", uint8[::1]),
    #
    # An array that saves the indices of non-empty bins for each feature on train data.
    ("non_empty_bins_train", uint8[:, ::1]),
    #
    # An array that saves the indices of non-empty bins for each feature on valid data.
    ("non_empty_bins_valid", uint8[:, ::1]),
]


node_classifier_context_type = [
    *node_context_type,
    #
    # Weighted number of training samples for each (feature, bin, label) in the node
    ("y_sum", float32[:, :, ::1]),
    #
    # Prediction produced by the node using the training data it contains
    ("y_pred", float32[::1]),
]


node_regressor_context_type = [
    *node_context_type,
    #
    # Weighted sum of the labels for each (feature, bin) in the node
    ("y_sum", float32[:, ::1]),
    #
    # Weighted sum of the squared labels for each (feature, bin) in the node
    ("y_sq_sum", float32[:, ::1]),
    #
    # Prediction produced by the node using the training data it contains
    ("y_pred", float32),
]


@jitclass(node_classifier_context_type)
class NodeClassifierContext:
    """A node_context contains all the data required to find the best split of the
    node for a classification tree.

    Parameters
    ----------
    tree_context : TreeClassifierContext
        An object describing the context of the tree

    Attributes
    ----------
    features_pool : ndarray
        Array of shape (n_features,) of intp dtype containing all the features
        indexes, namely [0, ..., n_features-1]

    features_sampled : ndarray
        Array of shape (max_features,) of intp dtype containing sampled features to
        be considered for possible splitting

    sample_features : bool
        Do we need to perform features sampling ? (This if True if n_features !=
        max_features and False otherwise)

    features : ndarray
        Array of shape (max_features,) of intp dtype containing the candidate features
        for splitting. These features are chosen uniformly at random each time a
        split is looked for

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

    n_samples_train_in_bins : ndarray
        Number of training samples for each (feature, bin) in the node

    w_samples_train_in_bins : ndarray
        Weighted number of training samples for each (feature, bin) in the node

    n_samples_valid_in_bins : ndarray
        Number of validation samples for each (feature, bin) in the node

    w_samples_valid_in_bins : ndarray
        Weighted number of validation samples for each (feature, bin) in the node

    non_empty_bins_train_count : ndarray
        A vector that counts the number of non-empty bins for each feature on train
         data.

    non_empty_bins_valid_count : ndarray
        A vector that counts the number of non-empty bins for each feature on valid
         data.

    non_empty_bins_train : ndarray
        An array that saves the indices of non-empty bins for each feature on train
        data.

    non_empty_bins_valid : ndarray
        An array that saves the indices of non-empty bins for each feature on valid
        data.

    y_sum : ndarray
        Weighted number of training samples for each (feature, bin, label) in the node

    y_pred : ndarray
        Prediction produced by the node using the training data it contains
    """
    def __init__(self, tree_context):
        init_node_context(tree_context, self)
        max_features = tree_context.max_features
        max_bins = tree_context.max_bins
        n_classes = tree_context.n_classes
        self.y_sum = np.empty((max_features, max_bins, n_classes), dtype=np.float32)
        self.y_pred = np.empty(n_classes, dtype=np.float32)


@jitclass(node_regressor_context_type)
class NodeRegressorContext:
    """A node_context contains all the data required to find the best split of the
    node for a regression tree.

    Parameters
    ----------
    tree_context : TreeRegressorContext
        An object describing the context of the tree

    Attributes
    ----------
    features_pool : ndarray
        Array of shape (n_features,) of intp dtype containing all the features
        indexes, namely [0, ..., n_features-1]

    features_sampled : ndarray
        Array of shape (max_features,) of intp dtype containing sampled features to
        be considered for possible splitting

    sample_features : bool
        Do we need to perform features sampling ? (This if True if n_features !=
        max_features and False otherwise)

    features : ndarray
        Array of shape (max_features,) of intp dtype containing the candidate features
        for splitting. These features are chosen uniformly at random each time a
        split is looked for

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

    n_samples_train_in_bins : ndarray
        Number of training samples for each (feature, bin) in the node

    w_samples_train_in_bins : ndarray
        Weighted number of training samples for each (feature, bin) in the node

    n_samples_valid_in_bins : ndarray
        Number of validation samples for each (feature, bin) in the node

    w_samples_valid_in_bins : ndarray
        Weighted number of validation samples for each (feature, bin) in the node

    non_empty_bins_train_count : ndarray
        A vector that counts the number of non-empty bins for each feature on train
         data.

    non_empty_bins_valid_count : ndarray
        A vector that counts the number of non-empty bins for each feature on valid
         data.

    non_empty_bins_train : ndarray
        An array that saves the indices of non-empty bins for each feature on train
        data.

    non_empty_bins_valid : ndarray
        An array that saves the indices of non-empty bins for each feature on valid
        data.

    y_sum : ndarray
        Weighted sum of the labels for each (feature, bin) in the node

    y_sq_sum : ndarray
        Weighted sum of the squared labels for each (feature, bin) in the node

    y_pred : float
        Prediction produced by the node using the training data it contains
    """

    def __init__(self, tree_context):
        init_node_context(tree_context, self)
        max_features = tree_context.max_features
        max_bins = tree_context.max_bins
        self.y_sum = np.empty((max_features, max_bins), dtype=np.float32)
        self.y_sq_sum = np.empty((max_features, max_bins), dtype=np.float32)
        self.y_pred = 0.0


NodeClassifierContextType = get_type(NodeClassifierContext)
NodeRegressorContextType = get_type(NodeRegressorContext)


@jit(
    [
        void(TreeClassifierContextType, NodeClassifierContextType),
        void(TreeRegressorContextType, NodeRegressorContextType),
    ],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
)
def init_node_context(tree_context, node_context):
    """A common initializer for NodeContextClassifier and NodeContextRegressor.
    Since numba's jitclass support for inheritance is inexistant, we use this
    function both in the constructors of NodeContextClassifier and NodeContextRegressor.

    Parameters
    ----------
    tree_context : TreeContext
        An object describing the context of the tree

    node_context : NodeClassifierContext or NodeRegressorContext
        The node context that we be partly initialized by this function
    """
    max_features = tree_context.max_features
    max_bins = tree_context.max_bins
    n_features = tree_context.n_features
    # If max_features is the same as n_features, we don't need to sample features
    node_context.sample_features = max_features != n_features
    # This array contains the index of all the features. This will be modified
    # inplace when sampling features to be considered for splits
    node_context.features_pool = np.arange(0, n_features, dtype=np.uintp)
    # This array will contain the features sampled uniformly at random (without
    # replacement) to be considered for splits
    node_context.features_sampled = np.arange(0, max_features, dtype=np.uintp)
    node_context.n_samples_train_in_bins = np.empty(
        (max_features, max_bins), dtype=np.uintp
    )
    node_context.w_samples_train_in_bins = np.empty(
        (max_features, max_bins), dtype=np.float32
    )
    node_context.n_samples_valid_in_bins = np.empty(
        (max_features, max_bins), dtype=np.uintp
    )
    node_context.w_samples_valid_in_bins = np.empty(
        (max_features, max_bins), dtype=np.float32
    )
    node_context.non_empty_bins_train = np.empty((max_features, max_bins), dtype=uint8)
    node_context.non_empty_bins_train_count = np.empty((max_features,), dtype=uint8)
    node_context.non_empty_bins_valid = np.empty((max_features, max_bins), dtype=uint8)
    node_context.non_empty_bins_valid_count = np.empty((max_features,), dtype=uint8)


@jit(
    void(
        TreeClassifierContextType, NodeClassifierContextType, uintp, uintp, uintp, uintp
    ),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={
        "n_samples_train_in_bins": uintp[:, ::1],
        "n_samples_valid_in_bins": uintp[:, ::1],
        "w_samples_train_in_bins": float32[:, ::1],
        "w_samples_valid_in_bins": float32[:, ::1],
        "non_empty_bins_train": uint8[:, ::1],
        "non_empty_bins_valid": uint8[:, ::1],
        "non_empty_bins_train_count": uint8[::1],
        "non_empty_bins_valid_count": uint8[::1],
        "y_sum": float32[:, :, ::1],
        "y_pred": float32[::1],
        "features": uintp[::1],
        "X": uint8[:, :],
        "y": float32[::1],
        "sample_weights": float32[::1],
        "partition_train": uintp[::1],
        "partition_valid": uintp[::1],
        "n_classes": uintp,
        "dirichlet": float32,
        "aggregation": boolean,
        "train_indices": uintp[::1],
        "valid_indices": uintp[::1],
        "w_samples_train": float32,
        "w_samples_valid": float32,
        "f": uintp,
        "loss_valid": float32,
        "feature": uintp,
        "sample": uintp,
        "bin": uint8,
        "label": uintp,
        "sample_weight": float32,
        "k": uintp,
    },
)
def compute_node_classifier_context(
    tree_context, node_context, start_train, end_train, start_valid, end_valid
):
    """Computes the node context from the data and from the tree context for
    classification. Computations are saved in the passed node_context.

    Parameters
    ----------
    tree_context : TreeContext
        The tree context

    node_context : NodeClassifierContext
        The node context that this function will compute

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
    """
    n_samples_train_in_bins = node_context.n_samples_train_in_bins
    n_samples_valid_in_bins = node_context.n_samples_valid_in_bins
    w_samples_train_in_bins = node_context.w_samples_train_in_bins
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins
    non_empty_bins_train = node_context.non_empty_bins_train
    non_empty_bins_train_count = node_context.non_empty_bins_train_count
    non_empty_bins_valid = node_context.non_empty_bins_valid
    non_empty_bins_valid_count = node_context.non_empty_bins_valid_count
    y_sum = node_context.y_sum
    y_pred = node_context.y_pred

    # If necessary, sample the features
    if node_context.sample_features:
        sample_without_replacement(
            node_context.features_pool, node_context.features_sampled
        )

    features = node_context.features_sampled
    n_samples_train_in_bins.fill(0)
    n_samples_valid_in_bins.fill(0)
    w_samples_train_in_bins.fill(0.0)
    w_samples_valid_in_bins.fill(0.0)
    non_empty_bins_train.fill(0)
    non_empty_bins_train_count.fill(0)
    non_empty_bins_valid.fill(0)
    non_empty_bins_valid_count.fill(0)
    y_sum.fill(0.0)
    y_pred.fill(0.0)

    # Get information from the tree context
    X = tree_context.X
    y = tree_context.y
    sample_weights = tree_context.sample_weights
    partition_train = tree_context.partition_train
    partition_valid = tree_context.partition_valid
    n_classes = tree_context.n_classes
    dirichlet = tree_context.dirichlet
    aggregation = tree_context.aggregation

    # The indices of the training samples contained in the node
    train_indices = partition_train[start_train:end_train]
    valid_indices = partition_valid[start_valid:end_valid]

    # Weighted number of training and validation samples
    w_samples_train = 0.0
    w_samples_valid = 0.0

    # A counter for the features
    f = 0
    # The validation loss
    loss_valid = 0.0

    # TODO: unrolling the for loop could be faster
    # For-loop on features first and then samples (X is F-major)
    for feature in features:
        # Compute statistics about training samples
        for sample in train_indices:
            bin = X[sample, feature]
            label = uintp(y[sample])
            sample_weight = sample_weights[sample]
            if f == 0:
                w_samples_train += sample_weight
                y_pred[label] += sample_weight

            if n_samples_train_in_bins[f, bin] == 0:
                # It's the first time we find a train sample for this (feature, bin)
                # We save the bin number at index non_empty_bins_train_count[f]
                non_empty_bins_train[f, non_empty_bins_train_count[f]] = bin
                # We increase the count of non-empty bins for this feature
                non_empty_bins_train_count[f] += 1

            # One more sample in this bin for the current feature
            n_samples_train_in_bins[f, bin] += 1
            w_samples_train_in_bins[f, bin] += sample_weight
            # One more sample in this bin for the current feature with this label
            y_sum[f, bin, label] += sample_weight

        # TODO: we should put this outside so that we can change the dirichlet
        #  parameter without re-growing the tree
        # The prediction is given by the formula
        #   y_k = (n_k + dirichlet) / (n_samples + dirichlet * n_classes)
        # where n_k is the number of samples with label class k
        if f == 0:
            for k in range(n_classes):
                y_pred[k] = (y_pred[k] + dirichlet) / (
                    w_samples_train + n_classes * dirichlet
                )

        # Compute sample counts about validation samples
        if aggregation:
            for sample in valid_indices:
                bin = X[sample, feature]
                sample_weight = sample_weights[sample]
                if f == 0:
                    w_samples_valid += sample_weight
                    label = uintp(y[sample])
                    # TODO: aggregation loss is hard-coded here. Call a function instead
                    #  when implementing other losses
                    loss_valid -= sample_weight * log(y_pred[label])

                if n_samples_valid_in_bins[f, bin] == 0.0:
                    # It's the first time we find a valid sample for this (feature, bin)
                    # We save the bin number at index non_empty_bins_valid_count[f]
                    non_empty_bins_valid[f, non_empty_bins_valid_count[f]] = bin
                    # We increase the count of non-empty bins for this feature
                    non_empty_bins_valid_count[f] += 1

                n_samples_valid_in_bins[f, bin] += 1
                w_samples_valid_in_bins[f, bin] += sample_weight

        f += 1

    # Save remaining things in the node context
    node_context.n_samples_train = end_train - start_train
    node_context.n_samples_valid = end_valid - start_valid
    node_context.loss_valid = loss_valid
    node_context.w_samples_train = w_samples_train
    node_context.w_samples_valid = w_samples_valid


@jit(
    void(
        TreeRegressorContextType, NodeRegressorContextType, uintp, uintp, uintp, uintp
    ),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={
        "n_samples_train_in_bins": uintp[:, ::1],
        "n_samples_valid_in_bins": uintp[:, ::1],
        "w_samples_train_in_bins": float32[:, ::1],
        "w_samples_valid_in_bins": float32[:, ::1],
        "non_empty_bins_train": uint8[:, ::1],
        "non_empty_bins_valid": uint8[:, ::1],
        "non_empty_bins_train_count": uint8[::1],
        "non_empty_bins_valid_count": uint8[::1],
        "y_sum": float32[:, ::1],
        "y_sq_sum": float32[:, ::1],
        "y_pred": float32,
        "features": uintp[::1],
        "X": uint8[:, :],
        "y": float32[::1],
        "sample_weights": float32[::1],
        "partition_train": uintp[::1],
        "partition_valid": uintp[::1],
        "aggregation": boolean,
        "train_indices": uintp[::1],
        "valid_indices": uintp[::1],
        "w_samples_train": float32,
        "w_samples_valid": float32,
        "f": uintp,
        "loss_valid": float32,
        "feature": uintp,
        "sample": uintp,
        "bin": uint8,
        "label": float32,
        "sample_weight": float32,
        "k": uintp,
        "w_y": float32,
    },
)
def compute_node_regressor_context(
    tree_context, node_context, start_train, end_train, start_valid, end_valid
):
    """Computes the node context from the data and from the tree context for regression.
    Computations are saved in the passed node_context.

    Parameters
    ----------
    tree_context : TreeContext
        The tree context

    node_context : NodeRegressorContext
        The node context that this function will compute

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
    """
    # Initialize the things from the node context
    n_samples_train_in_bins = node_context.n_samples_train_in_bins
    n_samples_valid_in_bins = node_context.n_samples_valid_in_bins
    w_samples_train_in_bins = node_context.w_samples_train_in_bins
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins
    non_empty_bins_train = node_context.non_empty_bins_train
    non_empty_bins_train_count = node_context.non_empty_bins_train_count
    non_empty_bins_valid = node_context.non_empty_bins_valid
    non_empty_bins_valid_count = node_context.non_empty_bins_valid_count
    y_sum = node_context.y_sum
    y_sq_sum = node_context.y_sq_sum

    # If necessary, sample the features
    if node_context.sample_features:
        sample_without_replacement(
            node_context.features_pool, node_context.features_sampled
        )

    features = node_context.features_sampled
    n_samples_train_in_bins.fill(0)
    n_samples_valid_in_bins.fill(0)
    w_samples_train_in_bins.fill(0.0)
    w_samples_valid_in_bins.fill(0.0)
    non_empty_bins_train.fill(0)
    non_empty_bins_train_count.fill(0)
    non_empty_bins_valid.fill(0)
    non_empty_bins_valid_count.fill(0)
    y_sum.fill(0.0)
    y_sq_sum.fill(0.0)
    y_pred = 0.0

    # Get information from the tree context
    X = tree_context.X
    y = tree_context.y
    sample_weights = tree_context.sample_weights
    partition_train = tree_context.partition_train
    partition_valid = tree_context.partition_valid
    aggregation = tree_context.aggregation

    # The indices of the training samples contained in the node
    train_indices = partition_train[start_train:end_train]
    valid_indices = partition_valid[start_valid:end_valid]

    # Weighted number of training and validation samples
    w_samples_train = 0.0
    w_samples_valid = 0.0

    # A counter for the features
    f = 0
    # The validation loss
    loss_valid = 0.0

    # TODO: unrolling the for loop could be faster
    # For-loop on features first and then samples (X is F-major)
    for feature in features:
        # Compute statistics about training samples
        for sample in train_indices:
            bin = X[sample, feature]
            label = y[sample]
            sample_weight = sample_weights[sample]
            w_y = sample_weight * label
            if f == 0:
                w_samples_train += sample_weight
                y_pred += w_y

            if n_samples_train_in_bins[f, bin] == 0:
                # It's the first time we find a train sample for this (feature, bin)
                # We save the bin number at index non_empty_bins_train_count[f]
                non_empty_bins_train[f, non_empty_bins_train_count[f]] = bin
                # We increase the count of non-empty bins for this feature
                non_empty_bins_train_count[f] += 1

            # One more sample in this bin for the current feature
            n_samples_train_in_bins[f, bin] += 1
            w_samples_train_in_bins[f, bin] += sample_weight
            # One more sample in this bin for the current feature with this label

            y_sum[f, bin] += w_y
            y_sq_sum[f, bin] += w_y * label

        # The prediction is simply the weighted average of labels
        if f == 0:
            y_pred /= w_samples_train

        # Compute sample counts about validation samples
        if aggregation:
            for sample in valid_indices:
                bin = X[sample, feature]
                sample_weight = sample_weights[sample]
                if f == 0:
                    w_samples_valid += sample_weight
                    label = y[sample]
                    # TODO: aggregation loss is hard-coded here. Call a function instead
                    #  when implementing other losses
                    loss_valid += sample_weight * (label - y_pred) * (label - y_pred)

                if n_samples_valid_in_bins[f, bin] == 0:
                    # It's the first time we find a valid sample for this (feature, bin)
                    # We save the bin number at index non_empty_bins_valid_count[f]
                    non_empty_bins_valid[f, non_empty_bins_valid_count[f]] = bin
                    # We increase the count of non-empty bins for this feature
                    non_empty_bins_valid_count[f] += 1

                n_samples_valid_in_bins[f, bin] += 1
                w_samples_valid_in_bins[f, bin] += sample_weight

        f += 1

    # Save remaining things in the node context
    node_context.y_pred = y_pred
    node_context.n_samples_train = end_train - start_train
    node_context.n_samples_valid = end_valid - start_valid
    node_context.loss_valid = loss_valid
    node_context.w_samples_train = w_samples_train
    node_context.w_samples_valid = w_samples_valid
