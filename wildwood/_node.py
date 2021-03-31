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
from ._tree_context import TreeContextType


# TODO: Some things in the node_tree could be removed

# This data-type describes all the information saved about a node.
# It's used in _tree.py where the Tree dataclass contains an attribute called nodes
# that uses this data type.
node_dtype = np.dtype(
    [
        # Index of the node in the nodes array
        ("node_id", np.uintp),
        # Index of the parent node
        ("parent", np.uintp),
        # Index of the left child node (it's TREE_LEAF is node is a leaf)
        ("left_child", np.intp),
        # Index of the right child node (it's TREE_LEAF is node is a leaf)
        ("right_child", np.intp),
        # Is the node a leaf ?
        ("is_leaf", np.bool),
        # Is the node a left child ? (it's ? for the root node)
        ("is_left", np.bool),
        # Depth of the node in the tree
        ("depth", np.uintp),
        # Feature used for splitting the node (it's ??? if node is a leaf)
        ("feature", np.uintp),
        # Continuous threshold used for splitting the node (not used for now)
        ("threshold", np.float32),
        # Index of the bin threshold used for splitting the node
        ("bin_threshold", np.uint8),
        # Impurity of the node. Used to avoid to split a "pure" node (with impurity=0).
        #   Note that impurity is computed using training samples (as all tree-growing
        #   related things)
        ("impurity", np.float32),
        # Validation loss of the node, computed on validation (out-of-the-bag) samples
        ("loss_valid", np.float32),
        # Logarithm of the subtree aggregation weight
        ("log_weight_tree", np.float32),
        # Number of training samples in the node
        ("n_samples_train", np.uintp),
        # Number of validation (out-of-the-bag) samples in the node
        ("n_samples_valid", np.uintp),
        # Weighted number of training samples in the node
        ("w_samples_train", np.float32),
        # Weighted number of validation (out-of-the-bag) samples in the node
        ("w_samples_valid", np.float32),
        # Index of the first training sample in the node. We have that
        #   partition_train[start_train:end_train] contains the indexes of the node's
        #   training samples
        ("start_train", np.uintp),
        # End-index of the slice containing the node's training samples indexes
        ("end_train", np.uintp),
        # Index of the first validation (out-of-the-bag) sample in the node. We have
        #   that partition_valid[start_valid:end_valid] contains the indexes of the
        #   node's validation samples
        ("start_valid", np.uintp),
        # End-index of the slice containing the node's validation samples indexes
        ("end_valid", np.uintp),
        # If the split is on a categorical feature?
        ("is_split_categorical", np.bool),
        # In case that the split is on a categorical feature,
        #    index of start of index of bins
        ("permutation_start", np.uint8),
        # In case that the split is on a categorical feature,
        #    index of end of index of bins
        ("permutation_end", np.uint8),
        # In case that the split is on a categorical feature,
        #    is bins within [start, end] go to left child?
        #    False means bins within [start, end] go to right child
        ("is_perm_left", np.bool),
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
    # This array will contain the features sampled uniformly at random (without
    # replacement) to be considered for splits
    ("features_sampled", uintp[::1]),
    # Do we need to perform features sampling ?
    ("sample_features", boolean),
    # Number of training samples in the node
    ("n_samples_train", intp),
    # Number of validation (out-of-the-bag) samples in the node
    ("n_samples_valid", intp),
    # Weighted number of training samples in the node
    ("w_samples_train", float32),
    # Weighted number of validation (out-of-the-bag) samples in the node
    ("w_samples_valid", float32),
    # Index of the first training sample in the node. We have that
    #   partition_train[start_train:end_train] contains the indexes of the node's
    #   training samples
    # ("start_train", intp),
    # End-index of the slice containing the node's training samples indexes
    # ("end_train", intp),
    # Index of the first validation (out-of-the-bag) sample in the node. We have
    #   that partition_valid[start_valid:end_valid] contains the indexes of the
    #   node's validation samples
    # ("start_valid", intp),
    # End-index of the slice containing the node's validation samples indexes
    # ("end_valid", intp),
    # Validation loss of the node, computed on validation (out-of-the-bag) samples
    ("loss_valid", float32),
    # Weighted number of training samples for each (feature, bin) in the node
    ("w_samples_train_in_bins", float32[:, ::1]),
    # Weighted number of validation samples for each (feature, bin) in the node
    ("w_samples_valid_in_bins", float32[:, ::1]),
    # Weighted number of training samples for each (feature, bin, label) in the node
    ("y_sum", float32[:, :, ::1]),
    # Prediction produced by the node using the training data it contains
    ("y_pred", float32[::1]),
]


@jitclass(node_context_type)
class NodeContext:
    """A node_context contains all the data required to find the best split of the node.

    Parameters
    ----------
    tree_context : TreeContext
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

    w_samples_train_in_bins : ndarray
        Weighted number of training samples for each (feature, bin) in the node

    w_samples_valid_in_bins : ndarray
        Weighted number of validation samples for each (feature, bin) in the node

    y_sum : ndarray
        Weighted number of training samples for each (feature, bin, label) in the node

    y_pred : ndarray
        Prediction produced by the node using the training data it contains
    """

    def __init__(self, tree_context):
        max_features = tree_context.max_features
        max_bins = tree_context.max_bins
        n_classes = tree_context.n_classes
        n_features = tree_context.n_features
        # If max_features is the same as n_features, we don't need to sample features
        self.sample_features = max_features != n_features
        # This array contains the index of all the features. This will be modified
        # inplace when sampling features to be considered for splits
        self.features_pool = np.arange(0, n_features, dtype=np.uintp)
        # This array will contain the features sampled uniformly at random (without
        # replacement) to be considered for splits
        self.features_sampled = np.arange(0, max_features, dtype=np.uintp)

        self.w_samples_train_in_bins = np.empty(
            (max_features, max_bins), dtype=np.float32
        )
        self.w_samples_valid_in_bins = np.empty(
            (max_features, max_bins), dtype=np.float32
        )
        self.y_sum = np.empty((max_features, max_bins, n_classes), dtype=np.float32)
        self.y_pred = np.empty(n_classes, dtype=np.float32)


NodeContextType = get_type(NodeContext)


@jit(
    void(TreeContextType, NodeContextType, uintp, uintp, uintp, uintp),
    nopython=True,
    nogil=True,
    boundscheck=False,
    locals={
        "w_samples_train_in_bins": float32[:, ::1],
        "w_samples_valid_in_bins": float32[:, ::1],
        "y_sum": float32[:, :, ::1],
        "y_pred": float32[::1],
        "features": uintp[::1],
        "X": uint8[::1, :],
        "y": float32[::1],
        "sample_weights": float32[::1],
        "partition_train": uintp[::1],
        "partition_valid": uintp[::1],
        "n_classes": uintp,
        "dirichlet": float32,
        "is_categorical": boolean[::1],
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
def compute_node_context(
    tree_context, node_context, start_train, end_train, start_valid, end_valid
):
    """Computes the node context from the data and from the tree context.
    Computations are saved in the passed node_context.

    Parameters
    ----------
    tree_context : TreeContext
        The tree context

    node_context : NodeContext
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
    w_samples_train_in_bins = node_context.w_samples_train_in_bins
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins
    y_sum = node_context.y_sum
    y_pred = node_context.y_pred

    # If necessary, sample the features
    if node_context.sample_features:
        sample_without_replacement(
            node_context.features_pool, node_context.features_sampled
        )

    features = node_context.features_sampled
    w_samples_train_in_bins.fill(0.0)
    w_samples_valid_in_bins.fill(0.0)
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
    is_categorical = tree_context.is_categorical

    # The indices of the training samples contained in the node
    train_indices = partition_train[start_train:end_train]
    print('train_indices=', train_indices)
    valid_indices = partition_valid[start_valid:end_valid]
    print("valid_indices=", valid_indices)

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
            # One more sample in this bin for the current feature
            w_samples_train_in_bins[f, bin] += sample_weight
            # One more sample in this bin for the current feature with this label
            y_sum[f, bin, label] += sample_weight

        # The prediction is given by the formula
        #   y_k = (n_k + dirichlet) / (n_samples + dirichlet * n_classes)
        # where n_k is the number of samples with label class k
        if f == 0:
            for k in range(n_classes):
                y_pred[k] = (y_pred[k] + dirichlet) / (
                    w_samples_train + n_classes * dirichlet
                )

        # Compute sample counts about validation samples
        for sample in valid_indices:
            bin = X[sample, feature]
            sample_weight = sample_weights[sample]
            if f == 0:
                w_samples_valid += sample_weight
                label = uintp(y[sample])
                loss_valid += -log(y_pred[label])

            w_samples_valid_in_bins[f, bin] += sample_weight

        f += 1

    # Save remaining things in the node context
    node_context.n_samples_train = end_train - start_train
    node_context.n_samples_valid = end_valid - start_valid
    # Don't forget to normalize the validation loss
    node_context.loss_valid = loss_valid / node_context.n_samples_valid
    node_context.w_samples_train = w_samples_train
    node_context.w_samples_valid = w_samples_valid
    print("node_context: w_samples_train=", w_samples_train,
          "w_samples_valid=", w_samples_valid,
          "n_samples_train=", node_context.n_samples_train,
          "n_samples_valid=", node_context.n_samples_valid)
