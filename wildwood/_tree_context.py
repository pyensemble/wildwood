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
    intp,
    uintp,
    float32,
)
from numba.experimental import jitclass
from ._utils import get_type


# TODO: X is uint8[:, :] while it could be uint8[::1, :] namely forced F-major,
#  but if X has shape (n_samples, 1) (only one feature) then it is both F and C and
#  this raises a numba compulation error. But, it should not affect performance

# A pure data class which contains global context information, such as the dataset,
# training and validation indices, etc.
tree_context_type = [
    # The binned matrix of features
    ("X", uint8[:, :]),
    # The vector of labels
    ("y", float32[::1]),
    # Sample weights
    ("sample_weights", float32[::1]),
    # Training sample indices for tree growth
    ("train_indices", uintp[::1]),
    # Validation sample indices for tree aggregation
    ("valid_indices", uintp[::1]),
    # Total sample size
    ("n_samples", uintp),
    # Training sample size
    ("n_samples_train", uintp),
    # Validation sample size
    ("n_samples_valid", uintp),
    # The total number of features
    ("n_features", uintp),
    # Maximum number of bins
    ("max_bins", intp),
    # Actual number of bins used for each feature
    ("n_bins_per_feature", intp[::1]),
    # Maximum number of features to try for splitting
    ("max_features", uintp),
    ("aggregation", boolean),
    # Step-size used in the aggregation weights
    ("step", float32),
    ("partition_train", uintp[::1]),
    ("partition_valid", uintp[::1]),
    ("left_buffer", uintp[::1]),
    ("right_buffer", uintp[::1]),
]


tree_classifier_context_type = [
    *tree_context_type,
    # The number of classes
    ("n_classes", uintp),
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
        X,
        y,
        sample_weights,
        train_indices,
        valid_indices,
        n_classes,
        max_bins,
        n_bins_per_feature,
        max_features,
        aggregation,
        dirichlet,
        step,
    ):
        init_tree_context(
            self,
            X,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            max_bins,
            n_bins_per_feature,
            max_features,
            aggregation,
            step,
        )
        self.n_classes = n_classes
        self.dirichlet = dirichlet


@jitclass(tree_regressor_context_type)
class TreeRegressorContext:
    """
    The splitting context holds all the useful data for splitting
    """

    def __init__(
        self,
        X,
        y,
        sample_weights,
        train_indices,
        valid_indices,
        max_bins,
        n_bins_per_feature,
        max_features,
        aggregation,
        step,
    ):
        init_tree_context(
            self,
            X,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            max_bins,
            n_bins_per_feature,
            max_features,
            aggregation,
            step,
        )


TreeClassifierContextType = get_type(TreeClassifierContext)
TreeRegressorContextType = get_type(TreeRegressorContext)


@jit(
    [
        void(
            TreeClassifierContextType,
            uint8[:, :],
            float32[::1],
            float32[::1],
            uintp[::1],
            uintp[::1],
            intp,
            intp[::1],
            uintp,
            boolean,
            float32,
        ),
        void(
            TreeRegressorContextType,
            uint8[:, :],
            float32[::1],
            float32[::1],
            uintp[::1],
            uintp[::1],
            intp,
            intp[::1],
            uintp,
            boolean,
            float32,
        ),
    ],
    nopython=True,
    nogil=True,
)
def init_tree_context(
    tree_context,
    X,
    y,
    sample_weights,
    train_indices,
    valid_indices,
    max_bins,
    n_bins_per_feature,
    max_features,
    aggregation,
    step,
):
    tree_context.X = X
    tree_context.y = y
    tree_context.sample_weights = sample_weights
    tree_context.max_bins = max_bins
    tree_context.n_bins_per_feature = n_bins_per_feature
    tree_context.max_features = max_features
    tree_context.train_indices = train_indices
    tree_context.valid_indices = valid_indices
    tree_context.aggregation = aggregation
    tree_context.step = step
    tree_context.partition_train = train_indices.copy()
    tree_context.partition_valid = valid_indices.copy()

    n_samples, n_features = X.shape
    tree_context.n_samples = n_samples
    tree_context.n_features = n_features
    tree_context.n_samples_train = train_indices.shape[0]
    tree_context.n_samples_valid = valid_indices.shape[0]

    # Two buffers used in the split_indices function
    tree_context.left_buffer = np.empty(n_samples, dtype=np.uintp)
    tree_context.right_buffer = np.empty(n_samples, dtype=np.uintp)
