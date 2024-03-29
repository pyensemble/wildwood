# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This contains the data structure and type for a tree context.
"""

import numpy as np
from numba import (
    jit,
    void,
    boolean,
    uint8,
    uint64,
    intp,
    uintp,
    float32,
)
from numba.experimental import jitclass
from ._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, get_type
from .preprocessing.features_bitarray import FeaturesBitArray, FeaturesBitArrayType


# A pure data class which contains global context information, such as the datasets,
# training and validation indices, etc.
tree_context_type = [
    # The maximum number of actual bins used by the features bitarray
    ("max_n_bins", uint64),
    #
    # The features bitarray
    ("features_bitarray", FeaturesBitArrayType),
    #
    # The vector of labels
    ("y", float32[::1]),
    #
    # Sample weights
    ("sample_weights", float32[::1]),
    #
    # Training sample indices for tree growth
    ("train_indices", uintp[::1]),
    #
    # Validation sample indices for tree aggregation
    ("valid_indices", uintp[::1]),
    #
    # Total sample size
    ("n_samples", uintp),
    #
    # Maximum node depth
    ("max_depth", uintp),
    #
    # Training sample size
    ("n_samples_train", uintp),
    #
    # Validation sample size
    ("n_samples_valid", uintp),
    #
    # The total number of features
    ("n_features", uintp),
    #
    # Maximum number of features to try for splitting
    ("max_features", uintp),
    #
    # The minimum number of train samples and valid samples required to split a node
    ("min_samples_split", uintp),
    #
    # A split is considered only if it would lead to left and right childs with at
    # least this number of train and this number of valid samples
    ("min_samples_leaf", uintp),
    #
    # Is aggregation used ?
    ("aggregation", boolean),
    #
    # Step-size used in the aggregation weights
    ("step", float32),
    #
    # Categorical features indicator of shape (n_features,)
    ("is_categorical", boolean[::1]),
    #
    # A node in the tree contains start_train and end_train indices such that
    #  partition_train[start_train:end_train] contains the indexes of the node's
    #  training samples. This array is updated each time a node is split by the
    #  `split_indices` function from the _split.py module
    ("partition_train", uintp[::1]),
    #
    # A node in the tree contains start_valid and end_valid indices such that
    #  partition_valid[start_valid:end_valid] contains the indexes of the node's
    #  validation samples. This array is updated each time a node is split by the
    #  `split_indices` function from the _split.py module
    ("partition_valid", uintp[::1]),
    #
    # A "buffer" used in the split_indices function
    ("left_buffer", uintp[::1]),
    #
    # A "buffer" used in the split_indices function
    ("right_buffer", uintp[::1]),
    #
    # Criterion: "gini", "entropy" or "mse", encoded as a uint8 (following the
    # mapping from _utils.criteria_mapping)
    ("criterion", uint8),
]


tree_classifier_context_type = [
    *tree_context_type,
    #
    # The number of classes
    ("n_classes", uintp),
    #
    # categorical split strategy
    ("cat_split_strategy", uint8),
    #
    # Dirichlet parameter
    ("dirichlet", float32),
]

tree_regressor_context_type = [
    *tree_context_type,
]


@jitclass(tree_classifier_context_type)
class TreeClassifierContext:
    """
    The splitting context holds all the useful data for splitting
    """

    def __init__(
        self,
        features_bitarray,
        y,
        sample_weights,
        train_indices,
        valid_indices,
        n_classes,
        max_features,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        aggregation,
        dirichlet,
        step,
        is_categorical,
        cat_split_strategy,
        criterion,
    ):
        init_tree_context(
            self,
            features_bitarray,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            max_features,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            aggregation,
            step,
            is_categorical,
            criterion,
        )
        self.n_classes = n_classes
        self.dirichlet = dirichlet
        self.cat_split_strategy = cat_split_strategy


@jitclass(tree_regressor_context_type)
class TreeRegressorContext:
    """
    The splitting context holds all the useful data for splitting
    """

    def __init__(
        self,
        features_bitarray,
        y,
        sample_weights,
        train_indices,
        valid_indices,
        max_features,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        aggregation,
        step,
        is_categorical,
        criterion,
    ):
        init_tree_context(
            self,
            features_bitarray,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            max_features,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            aggregation,
            step,
            is_categorical,
            criterion,
        )


TreeClassifierContextType = get_type(TreeClassifierContext)
TreeRegressorContextType = get_type(TreeRegressorContext)


@jit(
    [
        void(
            TreeClassifierContextType,  # tree_context
            FeaturesBitArrayType,       # features_bitarray
            float32[::1],               # y
            float32[::1],               # sample_weights
            uintp[::1],                 # train_indices
            uintp[::1],                 # valid_indices
            intp,                       # max_features
            uintp,                       # max_depth
            intp,                       # min_samples_split
            uintp,                      # min_samples_leaf
            boolean,                    # aggregation
            float32,                    # step
            boolean[::1],               # is_categorical
            uint8,                      # criterion
        ),
        void(
            TreeRegressorContextType,
            FeaturesBitArrayType,
            float32[::1],
            float32[::1],
            uintp[::1],
            uintp[::1],
            intp,
            uintp,
            intp,
            uintp,
            boolean,
            float32,
            boolean[::1],
            uint8,
        ),
    ],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    fastmath=FASTMATH,
)
def init_tree_context(
    tree_context,
    features_bitarray,
    y,
    sample_weights,
    train_indices,
    valid_indices,
    max_features,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    aggregation,
    step,
    is_categorical,
    criterion,
):

    # The maximum number of actual bins used by the features bitarray
    max_n_bins = features_bitarray.max_values.max() + 1
    tree_context.max_n_bins = max_n_bins
    tree_context.features_bitarray = features_bitarray
    tree_context.y = y
    tree_context.sample_weights = sample_weights
    tree_context.max_features = max_features
    tree_context.min_samples_split = min_samples_split
    tree_context.min_samples_leaf = min_samples_leaf
    tree_context.train_indices = train_indices
    tree_context.valid_indices = valid_indices
    tree_context.aggregation = aggregation
    tree_context.step = step
    tree_context.criterion = criterion
    tree_context.partition_train = train_indices.copy()
    tree_context.partition_valid = valid_indices.copy()
    tree_context.is_categorical = is_categorical.copy()

    n_samples = features_bitarray.n_samples
    n_features = features_bitarray.n_features
    tree_context.n_samples = n_samples
    tree_context.max_depth = max_depth
    tree_context.n_features = n_features
    tree_context.n_samples_train = train_indices.shape[0]
    tree_context.n_samples_valid = valid_indices.shape[0]

    # Two buffers used in the split_indices function
    tree_context.left_buffer = np.empty(n_samples, dtype=np.uintp)
    tree_context.right_buffer = np.empty(n_samples, dtype=np.uintp)
