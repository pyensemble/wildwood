# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains all functions and dataclasses required to find the best split in
a node.
"""

import numpy as np
from numba import jit, boolean, uint8, uintp, float32, void
from numba.types import Tuple
from numba.experimental import jitclass

from ._node import NodeClassifierContextType, NodeRegressorContextType
from ._tree_context import TreeClassifierContextType, TreeRegressorContextType
from ._impurity import gini_childs, mse_childs, information_gain_proxy
from ._utils import get_type


split_type = [
    # True if we found a split, False otherwise
    ("found_split", boolean),
    #
    # Information gain proxy obtained thanks to this split
    ("gain_proxy", float32),
    #
    # Feature used to split the node
    ("feature", uintp),
    #
    # Index of the bin threshold used to split the node
    ("bin_threshold", uint8),
    #
    # Number of training samples in the left child node
    ("n_samples_train_left", uintp),
    #
    # Number of training samples in the right child node
    ("n_samples_train_right", uintp),
    #
    # Weighted number of training samples in the left child node
    ("w_samples_train_left", float32),
    #
    # Weighted number of training samples in the right child node
    ("w_samples_train_right", float32),
    #
    # Number of validation samples in the left child node
    ("n_samples_valid_left", uintp),
    #
    # Number of validation samples in the right child node
    ("n_samples_valid_right", uintp),
    #
    # Weighted number of validation samples in the left child node
    ("w_samples_valid_left", float32),
    #
    # Weighted number of validation samples in the right child node
    ("w_samples_valid_right", float32),
    #
    # Impurity of the left child node
    ("impurity_left", float32),
    #
    # Impurity of the right child node
    ("impurity_right", float32),
]


split_classifier_type = [
    *split_type,
    #
    # Weighted number of training samples for each label class in the left child node
    ("y_sum_left", float32[::1]),
    #
    # Weighted number of training samples for each label class in the right child node
    ("y_sum_right", float32[::1]),
]


split_regressor_type = [
    *split_type,
    #
    # Weighted sum of training labels in the left child node
    ("y_sum_left", float32),
    #
    # Weighted sum of training labels in the right child node
    ("y_sum_right", float32),
    #
    # Weighted sum of squared training labels in the left child node
    ("y_sq_sum_left", float32),
    #
    # Weighted sum of squared training labels in the right child node
    ("y_sq_sum_right", float32),
]


@jitclass(split_classifier_type)
class SplitClassifier(object):
    """Pure dataclass to store information about a potential split for classification.

    Parameters
    ----------
    n_classes : int
        Number of label classes

    Attributes
    ----------
    found_split : bool
        True if we found a split, False otherwise

    gain_proxy : float
        Information gain proxy obtained thanks to this split

    feature : int
        Feature used to split the node

    bin_threshold : int
        Index of the bin threshold used to split the node

    n_samples_train_left : int
        Number of training samples in the left child node

    n_samples_train_right : int
        Number of training samples in the right child node

    w_samples_train_left : float
        Weighted number of training samples in the left child node

    w_samples_train_right : float
        Weighted number of training samples in the right child node

    n_samples_valid_left : int
        Number of validation samples in the left child node

    n_samples_valid_right : int
        Number of validation samples in the right child node

    w_samples_valid_left : int
        Weighted number of validation samples in the left child node

    w_samples_valid_right : int
        Weighted number of validation samples in the right child node

    y_sum_left : ndarray
        Array of shape (n_classes,) with float32 dtype containing the weighted number
        of training samples for each label class in the left child node

    y_sum_right : ndarray
        Array of shape (n_classes,) with float32 dtype containing the weighted number
        of training samples for each label class in the right child node

    impurity_left : float
        Impurity of the left child node

    impurity_right : float
        Impurity of the right child node
    """

    def __init__(self, n_classes):
        init_split(self)
        self.y_sum_left = np.empty(n_classes, dtype=np.float32)
        self.y_sum_right = np.empty(n_classes, dtype=np.float32)


@jitclass(split_regressor_type)
class SplitRegressor(object):
    """Pure dataclass to store information about a potential split for regression.

    Attributes
    ----------
    found_split : bool
        True if we found a split, False otherwise

    gain_proxy : float
        Information gain proxy obtained thanks to this split

    feature : int
        Feature used to split the node

    bin_threshold : int
        Index of the bin threshold used to split the node

    n_samples_train_left : int
        Number of training samples in the left child node

    n_samples_train_right : int
        Number of training samples in the right child node

    w_samples_train_left : float
        Weighted number of training samples in the left child node

    w_samples_train_right : float
        Weighted number of training samples in the right child node

    n_samples_valid_left : int
        Number of validation samples in the left child node

    n_samples_valid_right : int
        Number of validation samples in the right child node

    w_samples_valid_left : int
        Weighted number of validation samples in the left child node

    w_samples_valid_right : int
        Weighted number of validation samples in the right child node

    y_sum_left : float
        Weighted sum of training labels in the left child node

    y_sum_right : float
        Weighted sum of training labels in the right child node

    y_sq_sum_left : float
        Weighted sum of squared training labels in the left child node

    y_sq_sum_right : float
        Weighted sum of squared training labels in the riht child node

    impurity_left : float
        Impurity of the left child node

    impurity_right : float
        Impurity of the right child node
    """

    def __init__(self):
        init_split(self)
        self.y_sum_left = 0.0
        self.y_sum_right = 0.0
        self.y_sq_sum_left = 0.0
        self.y_sq_sum_right = 0.0


SplitClassifierType = get_type(SplitClassifier)
SplitRegressorType = get_type(SplitRegressor)


@jit(
    [void(SplitClassifierType), void(SplitRegressorType)], nopython=True, nogil=True,
)
def init_split(split):
    """A common initializer for SplitClassifier and SplitRegressor.
    Since numba's jitclass support for inheritance is inexistant, we use this
    function both in the constructors of SplitClassifier and SplitRegressor.

    Parameters
    ----------
    split : SplitClassifier or SplitRegressor
        The split to be initialized by this function
    """
    split.found_split = False
    split.gain_proxy = -np.inf
    split.feature = 0
    split.bin_threshold = 0
    split.n_samples_train_left = 0
    split.n_samples_train_right = 0
    split.w_samples_train_left = 0.0
    split.w_samples_train_right = 0.0
    split.n_samples_valid_left = 0
    split.n_samples_valid_right = 0
    split.w_samples_valid_left = 0.0
    split.w_samples_valid_right = 0.0
    split.impurity_left = 0.0
    split.impurity_right = 0.0


@jit(
    [
        void(SplitClassifierType, SplitClassifierType),
        void(SplitRegressorType, SplitRegressorType),
    ],
    nopython=True,
    nogil=True,
)
def base_copy_split(from_split, to_split):
    """Base copies split data from from_split into to_split, both for
    SplitClassifierType and SplitRegressorType

    Parameters
    ----------
    from_split : SplitClassifier or SplitRegressor
        Copy data from this split

    to_split : SplitClassifier or SplitRegressor
        Data is copied into this split
    """
    to_split.found_split = from_split.found_split
    to_split.gain_proxy = from_split.gain_proxy
    to_split.feature = from_split.feature
    to_split.bin_threshold = from_split.bin_threshold
    to_split.n_samples_train_left = from_split.n_samples_train_left
    to_split.n_samples_train_right = from_split.n_samples_train_right
    to_split.w_samples_train_left = from_split.w_samples_train_left
    to_split.w_samples_train_right = from_split.w_samples_train_right
    to_split.n_samples_valid_left = from_split.n_samples_valid_left
    to_split.n_samples_valid_right = from_split.n_samples_valid_right
    to_split.w_samples_valid_left = from_split.w_samples_valid_left
    to_split.w_samples_valid_right = from_split.w_samples_valid_right
    to_split.impurity_left = from_split.impurity_left
    to_split.impurity_right = from_split.impurity_right


@jit(void(SplitClassifierType, SplitClassifierType), nopython=True, nogil=True)
def copy_split_classifier(from_split, to_split):
    base_copy_split(from_split, to_split)
    to_split.y_sum_left[:] = from_split.y_sum_left
    to_split.y_sum_right[:] = from_split.y_sum_right


@jit(void(SplitRegressorType, SplitRegressorType), nopython=True, nogil=True)
def copy_split_regressor(from_split, to_split):
    base_copy_split(from_split, to_split)
    to_split.y_sum_left = from_split.y_sum_left
    to_split.y_sum_right = from_split.y_sum_right


@jit(
    void(
        TreeClassifierContextType,
        NodeClassifierContextType,
        uintp,
        uintp,
        SplitClassifierType,
    ),
    nopython=True,
    nogil=True,
    locals={
        "n_classes": uintp,
        "n_bins": uint8,
        "n_samples_train": uintp,
        "w_samples_train": float32,
        "w_samples_valid": float32,
        "w_samples_train_in_bins": float32[::1],
        "w_samples_valid_in_bins": float32[::1],
        "y_sum_in_bins": float32[:, :],
        "n_samples_train_left": uintp,
        "n_samples_train_right": uintp,
        "w_samples_train_left": float32,
        "w_samples_train_right": float32,
        "w_samples_valid_left": float32,
        "w_samples_valid_right": float32,
        "y_sum_left": float32[::1],
        "y_sum_right": float32[::1],
        "gain_proxy": float32,
        "best_gain_proxy": float32,
        "bin": uint8,
        "impurity_left": float32,
        "impurity_right": float32,
    },
)
def find_best_split_classifier_along_feature(
    tree_context, node_context, feature, f, best_split
):
    """Finds the best split for classification (best bin_threshold) for a feature.
    If no split can be found (this happens when we can't find a split with a large
    enough weighted number of training and validation samples in the left and right
    childs) we simply set best_split.found_split to False).

    Parameters
    ----------
    tree_context : TreeClassifierContext
        The tree context which contains all the data about the tree that is useful to
        find a split

    node_context : NodeClassifierContext
        The node context which contains all the data about the node required to find
        a best split for it

    feature : uint
        The index of the feature for which we want to find a split

    f : uint
        The index in [0, ..., max_feature-1] corresponding to the position in the
        node_context of the feature we are considering for a split

    best_split : SplitClassifier
        Data about the best split found for the feature
    """
    # TODO: We shall use the correct number of bins for each feature (for categorical
    #  features with less than 256 modalities for instance)
    n_classes = tree_context.n_classes
    n_bins = tree_context.max_bins

    n_samples_train = node_context.n_samples_train
    w_samples_train = node_context.w_samples_train
    w_samples_valid = node_context.w_samples_valid
    # Weighed number of training samples in each bin for the feature
    w_samples_train_in_bins = node_context.w_samples_train_in_bins[f]
    # Weighted number of validation samples in each bin for the feature
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins[f]
    # Get the sum of labels (counts) in each bin for the feature
    y_sum_in_bins = node_context.y_sum[f]

    # Counts and sums on the left are zero, since we go from left to right, while
    # counts and sums on the right contain everything
    n_samples_train_left = 0
    n_samples_train_right = n_samples_train
    w_samples_train_left = 0.0
    w_samples_train_right = w_samples_train
    w_samples_valid_left = 0
    w_samples_valid_right = w_samples_valid
    # TODO: we should allocate these vectors in the tree_context, but the benefit should
    #  be negligible
    y_sum_left = np.zeros(n_classes, dtype=np.float32)
    y_sum_right = np.empty(n_classes, dtype=np.float32)
    y_sum_right[:] = y_sum_in_bins.sum(axis=0)

    # The best gain proxy seen so far
    best_gain_proxy = -np.inf
    # Did we find a split ? Not for now
    best_split.found_split = False

    # We go from left to right and compute the information gain proxy of all possible
    # splits in order to find the best one
    for bin in range(n_bins):
        # On the left we accumulate the counts
        w_samples_train_left += w_samples_train_in_bins[bin]
        w_samples_valid_left += w_samples_valid_in_bins[bin]
        # On the right we remove the counts
        w_samples_train_right -= w_samples_train_in_bins[bin]
        w_samples_valid_right -= w_samples_valid_in_bins[bin]

        # Update the label sums on the left and write in the same fashion
        y_sum_left += y_sum_in_bins[bin]
        y_sum_right -= y_sum_in_bins[bin]

        # TODO: this should be parametrizable through something like min_samples_leaf
        # If the split would lead to 0 training or 0 validation samples in the left
        # child then we don't consider the split
        if (w_samples_train_left <= 0.0) or (w_samples_valid_left <= 0.0):
            continue

        # If the split would lead to 0 training or 0 validation samples in the right,
        # and since we go from left to right, no other future bin on the right would
        # lead to an acceptable split, so we break the for loop over bins.
        if (w_samples_train_right <= 0.0) or (w_samples_valid_right <= 0.0):
            break

        # TODO: we shall pass the child impurity function to handle different impurities
        # Get the impurities of the left and right childs
        impurity_left, impurity_right = gini_childs(
            n_classes,
            w_samples_train_left,
            w_samples_train_right,
            y_sum_left,
            y_sum_right,
        )
        # And compute the information gain proxy
        gain_proxy = information_gain_proxy(
            impurity_left, impurity_right, w_samples_train_left, w_samples_train_right,
        )

        if gain_proxy > best_gain_proxy:
            # We've found a split better than the current one, so we save it
            best_gain_proxy = gain_proxy
            best_split.found_split = True
            best_split.gain_proxy = gain_proxy
            best_split.feature = feature
            best_split.bin_threshold = bin
            best_split.impurity_left = impurity_left
            best_split.impurity_right = impurity_right
            best_split.n_samples_train_left = n_samples_train_left
            best_split.n_samples_train_right = n_samples_train_right
            best_split.w_samples_train_left = w_samples_train_left
            best_split.w_samples_train_right = w_samples_train_right
            best_split.y_sum_left[:] = y_sum_left
            best_split.y_sum_right[:] = y_sum_right


@jit(
    void(
        TreeRegressorContextType,
        NodeRegressorContextType,
        uintp,
        uintp,
        SplitRegressorType,
    ),
    nopython=True,
    nogil=True,
    locals={
        "n_bins": uint8,
        "n_samples_train": uintp,
        "w_samples_train": float32,
        "w_samples_valid": float32,
        "w_samples_train_in_bins": float32[::1],
        "w_samples_valid_in_bins": float32[::1],
        "n_samples_train_left": uintp,
        "n_samples_train_right": uintp,
        "w_samples_train_left": float32,
        "w_samples_train_right": float32,
        "w_samples_valid_left": float32,
        "w_samples_valid_right": float32,
        "y_sum_left": float32,
        "y_sum_right": float32,
        "y_sq_sum_left": float32,
        "y_sq_sum_right": float32,
        "gain_proxy": float32,
        "best_gain_proxy": float32,
        "bin": uint8,
        "impurity_left": float32,
        "impurity_right": float32,
    },
)
def find_best_split_regressor_along_feature(
    tree_context, node_context, feature, f, best_split
):
    """Finds the best split for regression (best bin_threshold) for a feature. If no
    split can be found (this happens when we can't find a split with a large enough
    weighted number of training and validation samples in the left and right childs)
    we simply set best_split.found_split to False).

    Parameters
    ----------
    tree_context : TreeRegressorContext
        The tree context which contains all the data about the tree that is useful to
        find a split

    node_context : NodeRegressorContext
        The node context which contains all the data about the node required to find
        a best split for it

    feature : uint
        The index of the feature for which we want to find a split

    f : uint
        The index in [0, ..., max_feature-1] corresponding to the position in the
        node_context of the feature we are considering for a split

    best_split : SplitRegressor
        Data about the best split found for the feature
    """
    # TODO: We shall use the correct number of bins for each feature (for categorical
    #  features with less than 256 modalities for instance)
    n_bins = tree_context.max_bins

    n_samples_train = node_context.n_samples_train
    w_samples_train = node_context.w_samples_train
    w_samples_valid = node_context.w_samples_valid
    w_samples_train_in_bins = node_context.w_samples_train_in_bins[f]
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins[f]
    # Weighted sum of labels in each bin for the feature
    y_sum_in_bins = node_context.y_sum[f]
    # Weighted sum of squared labels in each bin for the feature
    y_sq_sum_in_bins = node_context.y_sq_sum[f]

    # Counts and sums on the left are zero, since we go from left to right, while
    # counts and sums on the right contain everything
    n_samples_train_left = 0
    n_samples_train_right = n_samples_train
    w_samples_train_left = 0.0
    w_samples_train_right = w_samples_train
    w_samples_valid_left = 0
    w_samples_valid_right = w_samples_valid
    # TODO: we should allocate these vectors in the tree_context, but the benefit should
    #  be negligible

    y_sum_left = 0.0
    y_sum_right = y_sum_in_bins.sum(axis=0)
    y_sq_sum_left = 0.0
    y_sq_sum_right = y_sq_sum_in_bins.sum(axis=0)

    # The best gain proxy seen so far
    best_gain_proxy = -np.inf
    # Did we find a split ? Not for now
    best_split.found_split = False

    # We go from left to right and compute the information gain proxy of all possible
    # splits in order to find the best one
    for bin in range(n_bins):
        w_samples_train_left += w_samples_train_in_bins[bin]
        w_samples_valid_left += w_samples_valid_in_bins[bin]
        w_samples_train_right -= w_samples_train_in_bins[bin]
        w_samples_valid_right -= w_samples_valid_in_bins[bin]

        y_sum_left += y_sum_in_bins[bin]
        y_sum_right -= y_sum_in_bins[bin]
        y_sq_sum_left += y_sq_sum_in_bins[bin]
        y_sq_sum_right -= y_sq_sum_in_bins[bin]

        # TODO: this should be parametrizable through something like min_samples_leaf
        # If the split would lead to 0 training or 0 validation samples in the left
        # child then we don't consider the split
        if (w_samples_train_left <= 0.0) or (w_samples_valid_left <= 0.0):
            continue

        # If the split would lead to 0 training or 0 validation samples in the right,
        # and since we go from left to right, no other future bin on the right would
        # lead to an acceptable split, so we break the for loop over bins.
        if (w_samples_train_right <= 0.0) or (w_samples_valid_right <= 0.0):
            break

        impurity_left, impurity_right = mse_childs(
            w_samples_train_left,
            w_samples_train_right,
            y_sum_left,
            y_sum_right,
            y_sq_sum_left,
            y_sq_sum_right,
        )
        # And compute the information gain proxy
        gain_proxy = information_gain_proxy(
            impurity_left, impurity_right, w_samples_train_left, w_samples_train_right,
        )

        if gain_proxy > best_gain_proxy:
            # We've found a split better than the current one, so we save it
            best_gain_proxy = gain_proxy
            best_split.found_split = True
            best_split.gain_proxy = gain_proxy
            best_split.feature = feature
            best_split.bin_threshold = bin
            best_split.impurity_left = impurity_left
            best_split.impurity_right = impurity_right
            best_split.n_samples_train_left = n_samples_train_left
            best_split.n_samples_train_right = n_samples_train_right
            best_split.w_samples_train_left = w_samples_train_left
            best_split.w_samples_train_right = w_samples_train_right
            best_split.y_sum_left = y_sum_left
            best_split.y_sum_right = y_sum_right
            best_split.y_sq_sum_left = y_sq_sum_left
            best_split.y_sq_sum_right = y_sq_sum_right


# TODO: no signature for this function this I don't know how to type first-order
#  functions with numba
@jit(
    nopython=True,
    nogil=True,
    locals={
        "features": uintp[::1],
        "best_gain_proxy": float32,
        "feature": uintp,
        "f": uintp,
    },
)
def find_node_split(
    tree_context,
    node_context,
    find_best_split_along_feature,
    copy_split,
    best_split,
    candidate_split,
):
    """Finds the best split (best bin_threshold) given the node context.
    If no split can be found (this happens when we can't find a split with a large
    enough weighted number of training and validation samples in the left and right
    childs) we simply set best_split.found_split to False).

    Parameters
    ----------
    tree_context : TreeClassifierContext or TreeRegressorContext
        The tree context which contains all the data about the tree that is useful to
        find a split

    node_context : NodeClassifierContext or NodeRegressorContext
        The node context which contains all the data about the node required to find
        a best split for it

    find_best_split_along_feature : function
        The function that looks for the best split along a feature

    copy_split : function
        A function that allows to copy a split of the correct type

    best_split : SplitClassifier or SplitRegressor
        The best split found. If there no best split, then best_split.found_split is
        set to False

    candidate_split : SplitClassifier or SplitRegressor
        A temporary split object used for computations only
    """
    # Get the set of features to try out
    features = node_context.features_sampled
    # Loop over the possible features
    best_gain_proxy = -np.inf
    init_split(best_split)
    init_split(candidate_split)
    f = 0
    for feature in features:
        # Compute the best bin and gain proxy obtained for the feature
        find_best_split_along_feature(
            tree_context, node_context, feature, f, candidate_split
        )
        # If we found a candidate split along the feature
        if candidate_split.found_split:
            # And if it's better than the current one
            if candidate_split.gain_proxy >= best_gain_proxy:
                # Then we replace the best current split
                copy_split(candidate_split, best_split)
                best_gain_proxy = candidate_split.gain_proxy
        f += 1


@jit(
    [
        Tuple((uintp, uintp))(
            TreeClassifierContextType, SplitClassifierType, uintp, uintp, uintp, uintp
        ),
        Tuple((uintp, uintp))(
            TreeRegressorContextType, SplitRegressorType, uintp, uintp, uintp, uintp
        ),
    ],
    nopython=True,
    nogil=True,
    locals={
        "feature": uintp,
        "bin_threshold": uint8,
        "Xf": uint8[:],
        "left_buffer": uintp[::1],
        "right_buffer": uintp[::1],
        "partition_train": uintp[::1],
        "partition_valid": uintp[::1],
        "n_samples_train_left": uintp,
        "n_samples_train_right": uintp,
        "i": uintp,
        "pos_train": uintp,
        "n_samples_valid_left": uintp,
        "n_samples_valid_right": uintp,
        "pos_valid": uintp,
    },
)
def split_indices(tree_context, split, start_train, end_train, start_valid, end_valid):
    """This functions updates both partition_train, partition_valid, pos_train,
    pos_valid so that the following is satisfied:

    - partition_train[start_train:pos_train] contains the training sample indices of
      the left child
    - partition_train[pos_train:end_train] contains the training sample indices of
      the right child
    - partition_valid[start_valid:pos_valid] contains the validation sample indices
      of the left child
    - partition_valid[pos_valid:end_valid] contains the validation sample indices of
      the right child

    Parameters
    ----------
    tree_context : TreeClassifierContext or TreeRegressorContext
        The tree context which contains all the data about the tree that is useful to
        find a split

    split : SplitClassifier or SplitRegressor
        Data about the best split found for the node

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

    Returns
    -------
    output : tuple
        A tuple with two elements containing:
        - pos_train : int
            The index such that partition_train[start_train:pos_train] contains the
            training sample indices of the left child and
            partition_train[pos_train:end_train] contains the training sample indices of
            the right child
        - pos_train : int
            The index such that partition_valid[start_valid:pos_valid] contains the
            validation sample indices of the left child and
            partition_valid[pos_valid:end_valid] contains the validation sample
            indices of the right child
    """
    # The feature and the bin threshold used for this split
    feature = split.feature
    bin_threshold = split.bin_threshold
    Xf = tree_context.X.T[feature]

    left_buffer = tree_context.left_buffer
    right_buffer = tree_context.right_buffer

    # The current training sample indices in the node
    partition_train = tree_context.partition_train
    # The current validation sample indices in the node
    partition_valid = tree_context.partition_valid

    n_samples_train_left = 0
    n_samples_train_right = 0

    for i in partition_train[start_train:end_train]:
        if Xf[i] <= bin_threshold:
            left_buffer[n_samples_train_left] = i
            n_samples_train_left += 1
        else:
            right_buffer[n_samples_train_right] = i
            n_samples_train_right += 1

    pos_train = start_train + n_samples_train_left

    partition_train[start_train:pos_train] = left_buffer[:n_samples_train_left]
    partition_train[pos_train:end_train] = right_buffer[:n_samples_train_right]

    # We must have start_train + n_samples_train_left + n_samples_train_right ==
    # end_train
    n_samples_valid_left = 0
    n_samples_valid_right = 0
    for i in partition_valid[start_valid:end_valid]:
        if Xf[i] <= bin_threshold:
            left_buffer[n_samples_valid_left] = i
            n_samples_valid_left += 1
        else:
            right_buffer[n_samples_valid_right] = i
            n_samples_valid_right += 1

    pos_valid = start_valid + n_samples_valid_left

    partition_valid[start_valid:pos_valid] = left_buffer[:n_samples_valid_left]
    partition_valid[pos_valid:end_valid] = right_buffer[:n_samples_valid_right]

    return pos_train, pos_valid
