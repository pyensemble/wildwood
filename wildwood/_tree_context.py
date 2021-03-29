# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This contains the data structure and type for a tree context.
"""

import numpy as np
from numba import (
    boolean,
    uint8,
    intp,
    uintp,
    float32,
)
from numba.experimental import jitclass
from ._utils import get_type


# A pure data class which contains global context information, such as the dataset,
# training and validation indices, etc.
spec_tree_context = [
    # The binned matrix of features
    ("X", uint8[::1, :]),
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
    # The number of classes
    ("n_classes", uintp),
    # Maximum number of bins
    ("max_bins", intp),
    # Actual number of bins used for each feature
    ("n_bins_per_feature", intp[::1]),
    # Maximum number of features to try for splitting
    ("max_features", uintp),
    # TODO: only for classification
    # Dirichlet parameter
    ("aggregation", boolean),
    # Dirichlet parameter
    ("dirichlet", float32),
    # Step-size used in the aggregation weights
    ("step", float32),
    # categorical features indicator
    ("is_categorical", boolean[::1]),  # TODO Yiyang check this type
    ("partition_train", uintp[::1]),
    ("partition_valid", uintp[::1]),
    ("left_buffer", uintp[::1]),
    ("right_buffer", uintp[::1]),
]


@jitclass(spec_tree_context)
class TreeContext:
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
        is_categorical,
    ):
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.n_classes = n_classes
        self.max_bins = max_bins
        self.n_bins_per_feature = n_bins_per_feature
        self.max_features = max_features
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.aggregation = aggregation
        self.dirichlet = dirichlet
        self.step = step
        self.partition_train = train_indices.copy()
        self.partition_valid = valid_indices.copy()
        self.is_categorical = is_categorical

        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_samples_train = train_indices.shape[0]
        self.n_samples_valid = valid_indices.shape[0]

        # Two buffers used in the split_indices function
        self.left_buffer = np.empty(n_samples, dtype=uintp)
        self.right_buffer = np.empty(n_samples, dtype=uintp)


TreeContextType = get_type(TreeContext)
