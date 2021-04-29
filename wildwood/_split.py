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


eps = np.finfo("float32").eps


split_type = [
    # The ordered bins in the feature used by the split (in bins[:bins_count))
    ("bins", uint8[::1]),
    #
    # The number of (non-empty) bins in the feature used by the split
    ("bins_count", uint8),
    #
    # True if we found a split, False otherwise
    ("found_split", boolean),
    #
    # Is the split on a categorical feature?
    ("is_split_categorical", boolean),
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
    # Whenever the split is on a categorical features, array such that the bins in
    #  bin_partition[:bin_partition_size] go to the left child while the others
    #  go to the right child
    ("bin_partition", uint8[::1]),
    #
    # Whenever the split is on a categorical features, integer such that the bins in
    #  bin_partition[:bin_partition_size] go to the left child while the others
    #  go to the right child
    ("bin_partition_size", uint8),
    #
    # Weighted number of training samples in the left child node
    ("w_samples_train_left", float32),
    #
    # Weighted number of training samples in the right child node
    ("w_samples_train_right", float32),
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
    bins : ndarray
        The ordered bins in the feature used by the split (in bins[:bins_count))

    bins_count : int
        The number of (non-empty) bins in the feature used by the split

    found_split : bool
        True if we found a split, False otherwise

    is_split_categorical : bool
        Is the split on a categorical feature?

    gain_proxy : float
        Information gain proxy obtained thanks to this split

    feature : int
        Feature used to split the node

    bin_threshold : int
        Index of the bin threshold used to split the node

    bin_partition : ndarray
        Array of shape (128,) with uint8 dtype.
        Whenever the split is on a categorical features, ndarray such that the bins in
        bin_partition[:bin_partition_size] go to the left child while the others
        go to the right child

    bin_partition_size : int
        Whenever the split is on a categorical features, integer such that the bins in
        bin_partition[:bin_partition_size] go to the left child while the others
        go to the right child

    w_samples_train_left : float
        Weighted number of training samples in the left child node

    w_samples_train_right : float
        Weighted number of training samples in the right child node

    w_samples_valid_left : int
        Weighted number of validation samples in the left child node

    w_samples_valid_right : int
        Weighted number of validation samples in the right child node

    impurity_left : float
        Impurity of the left child node

    impurity_right : float
        Impurity of the right child node

    y_sum_left : ndarray
        Array of shape (n_classes,) with float32 dtype containing the weighted number
        of training samples for each label class in the left child node

    y_sum_right : ndarray
        Array of shape (n_classes,) with float32 dtype containing the weighted number
        of training samples for each label class in the right child node
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
    bins : ndarray
        The ordered bins in the feature used by the split (in bins[:bins_count))

    bins_count : int
        The number of (non-empty) bins in the feature used by the split

    found_split : bool
        True if we found a split, False otherwise

    is_split_categorical : bool
        Is the split on a categorical feature?

    gain_proxy : float
        Information gain proxy obtained thanks to this split

    feature : int
        Feature used to split the node

    bin_threshold : int
        Index of the bin threshold used to split the node

    bin_partition : ndarray
        Array of shape (128,) with uint8 dtype.
        Whenever the split is on a categorical features, ndarray such that the bins in
        bin_partition[:bin_partition_size] go to the left child while the others
        go to the right child

    bin_partition_size : int
        Whenever the split is on a categorical features, integer such that the bins in
        bin_partition[:bin_partition_size] go to the left child while the others
        go to the right child

    w_samples_train_left : float
        Weighted number of training samples in the left child node

    w_samples_train_right : float
        Weighted number of training samples in the right child node

    w_samples_valid_left : int
        Weighted number of validation samples in the left child node

    w_samples_valid_right : int
        Weighted number of validation samples in the right child node

    impurity_left : float
        Impurity of the left child node

    impurity_right : float
        Impurity of the right child node

    y_sum_left : float
        Weighted sum of training labels in the left child node

    y_sum_right : float
        Weighted sum of training labels in the right child node

    y_sq_sum_left : float
        Weighted sum of squared training labels in the left child node

    y_sq_sum_right : float
        Weighted sum of squared training labels in the riht child node
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
    Since numba's jitclass support for class inheritance is inexistant, we use this
    function both in the constructors of SplitClassifier and SplitRegressor.

    Parameters
    ----------
    split : SplitClassifier or SplitRegressor
        The split to be initialized by this function
    """
    # bins is at most of size 256
    split.bins = np.empty(256, dtype=np.uint8)
    split.bins_count = 0
    split.found_split = False
    split.is_split_categorical = False
    split.gain_proxy = -np.inf
    split.feature = 0
    split.bin_threshold = 0
    # bin_partition is at most of size 128 (since we memorize the smallest of the
    # two elements of the bin partition)
    split.bin_partition = np.empty(128, dtype=np.uint8)
    split.bin_partition_size = 0
    split.w_samples_train_left = 0.0
    split.w_samples_train_right = 0.0
    split.w_samples_valid_left = 0.0
    split.w_samples_valid_right = 0.0
    split.impurity_left = 0.0
    split.impurity_right = 0.0



@jit(
    void(
        TreeClassifierContextType,
        NodeClassifierContextType,
        uintp,
        uintp,
        SplitClassifierType,
        uint8[::1]
    ),
    nopython=True,
    nogil=True,
    locals={
        "n_classes": uintp,
        "w_samples_train": float32,
        "w_samples_valid": float32,
        "w_samples_train_in_bins": float32[::1],
        "w_samples_valid_in_bins": float32[::1],
        "y_sum_in_bins": float32[:, :],
        "is_feature_categorical": boolean,
        "w_samples_train_left": float32,
        "w_samples_train_right": float32,
        "w_samples_valid_left": float32,
        "w_samples_valid_right": float32,
        "y_sum_left": float32[::1],
        "y_sum_right": float32[::1],
        "non_empty_bins_count": uint8,
        "gain_proxy": float32,
        "bin_threshold": uint8,
        "impurity_left": float32,
        "impurity_right": float32,
    },
)
def try_feature_order_for_classifier_split(tree_context, node_context, feature, f, best_split, ordered_bins):

    non_empty_bins_count = node_context.non_empty_bins_count[f]

    is_feature_categorical = tree_context.is_categorical[feature]

    y_sum_in_bins = node_context.y_sum[f]

    n_classes = tree_context.n_classes
    w_samples_train = node_context.w_samples_train
    w_samples_valid = node_context.w_samples_valid
    # Weighed number of training samples in each bin for the feature
    w_samples_train_in_bins = node_context.w_samples_train_in_bins[f]
    # Weighted number of validation samples in each bin for the feature
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins[f]
    # The indices of the non-empty bins for this feature. Note that this array is not
    # sorted in any fashion for now (bin indices appear in the order seen in the data)

    # Counts and sums on the left are zero, since we go from left to right, while
    # counts and sums on the right contain everything
    w_samples_train_left = 0.0
    w_samples_train_right = w_samples_train
    w_samples_valid_left = 0
    w_samples_valid_right = w_samples_valid
    # TODO: we should allocate these vectors in the tree_context, but the benefit
    #  should be negligible
    y_sum_left = np.zeros(n_classes, dtype=np.float32)
    y_sum_right = np.empty(n_classes, dtype=np.float32)
    # TODO: using TODO(*) would lead to a sum only over non-empty bins
    y_sum_right[:] = y_sum_in_bins.sum(axis=0)

    # We go from left to right and compute the information gain proxy of all possible
    #  splits in order to find the best one. Since bin_threshold is included in the
    #  left child, we must stop before the last bin, hence the bins[:-1]
    #  (otherwise w_samples_train_right == 0 for bin_threshold=bins[:-1])
    for bin_threshold in ordered_bins[:-1]:
        # On the left we accumulate the counts
        w_samples_train_left += w_samples_train_in_bins[bin_threshold]
        w_samples_valid_left += w_samples_valid_in_bins[bin_threshold]
        # On the right we remove the counts
        w_samples_train_right -= w_samples_train_in_bins[bin_threshold]
        w_samples_valid_right -= w_samples_valid_in_bins[bin_threshold]
        # Update the label sums on the left and write in the same fashion
        y_sum_left += y_sum_in_bins[bin_threshold]
        y_sum_right -= y_sum_in_bins[bin_threshold]

        # TODO: this test can be removed now since we use non-empty bins
        # If the split would lead to 0 training or 0 validation samples in the left
        # child then we don't consider the split
        if (w_samples_train_left <= eps) or (w_samples_valid_left <= eps):
            continue

        # TODO: this test can be removed now since we use non-empty bins
        # If the split would lead to 0 training or 0 validation samples in the right,
        # and since we go from left to right, no other future bin on the right would
        # lead to an acceptable split, so we break the for loop over bins.
        if (w_samples_train_right <= eps) or (w_samples_valid_right <= eps):
            break

        # TODO: we shall pass the child impurity function as an argument to handle
        #  different impurities
        # Get the impurities of the left and right children
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

        if gain_proxy > best_split.gain_proxy:
            # We've found a split better than the current one, so we save it
            best_split.found_split = True
            best_split.gain_proxy = gain_proxy
            best_split.feature = feature
            best_split.bin_threshold = bin_threshold
            best_split.impurity_left = impurity_left
            best_split.impurity_right = impurity_right
            best_split.w_samples_train_left = w_samples_train_left
            best_split.w_samples_train_right = w_samples_train_right
            best_split.w_samples_valid_left = w_samples_valid_left
            best_split.w_samples_valid_right = w_samples_valid_right
            best_split.y_sum_left[:] = y_sum_left
            best_split.y_sum_right[:] = y_sum_right
            best_split.is_split_categorical = is_feature_categorical
            best_split.bins[:non_empty_bins_count] = ordered_bins
            best_split.bins_count = non_empty_bins_count


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
        "y_sum_in_bins": float32[:, :],
        "y_sum_positives": float32[:, :],
        "is_feature_categorical": boolean,
        "non_empty_bins": uint8[::1],
        "non_empty_bins_count": uint8,
        "bins": uint8[::1],
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
        The best_split found so far
    """
    # TODO(*): get only non-zero bins for y_sum_in_bins as well ?
    # Get the sum (counts) for each (bin, label) of the feature
    y_sum_in_bins = node_context.y_sum[f]

    # The number of non-empty bins for this feature
    non_empty_bins_count = node_context.non_empty_bins_count[f]

    non_empty_bins = node_context.non_empty_bins[f, :non_empty_bins_count]

    # TODO: We don't need to instantiate it here
    bins = np.empty(non_empty_bins_count, dtype=np.uint8)

    is_feature_categorical = tree_context.is_categorical[feature]

    if is_feature_categorical:
        # TODO: Ordering using another strategy, such as CatBoost's target statistics)
        #  for multiclass classification ?
        # TODO: Something faster ?
        # Sort bins according to the proportions of labels equal to 1 in
        # each bin. This leads to the best split partitioning for regression and binary
        # classification
        # TODO: use a placeholder for y_sum_positives in split ?
        y_sum_positives = y_sum_in_bins[non_empty_bins, :].copy()
        y_sum_positives /= np.expand_dims(y_sum_positives.sum(axis=1), axis=1)

        for mod in range(1, y_sum_in_bins.shape[1]):
            idx_sort = np.argsort(y_sum_positives[:, mod])
            bins[:] = non_empty_bins[idx_sort]
            try_feature_order_for_classifier_split(tree_context, node_context, feature, f, best_split, bins)
    else:
        # TODO: we can avoid allocating this array for all non-categorical features
        # TODO: We can use also non_empty_bins.sort() which sorts inplace
        # When the feature is continuous, we simply sort the bins in ascending order
        bins[:] = np.sort(non_empty_bins)
        try_feature_order_for_classifier_split(tree_context, node_context, feature, f, best_split, bins)


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
        "w_samples_train": float32,
        "w_samples_valid": float32,
        "w_samples_train_in_bins": float32[::1],
        "w_samples_valid_in_bins": float32[::1],
        "y_sum_in_bins": float32[::1],
        "is_feature_categorical": boolean,
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
        "non_empty_bins": uint8[::1],
        "non_empty_bins_count": uint8,
        "gain_proxy": float32,
        "bin_threshold": uint8,
        "impurity_left": float32,
        "impurity_right": float32,
        "bins": uint8[::1],
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
    w_samples_train = node_context.w_samples_train
    w_samples_valid = node_context.w_samples_valid
    w_samples_train_in_bins = node_context.w_samples_train_in_bins[f]
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins[f]
    y_sum_in_bins = node_context.y_sum[f]
    # Weighted sum of squared labels in each bin for the feature
    y_sq_sum_in_bins = node_context.y_sq_sum[f]
    non_empty_bins_count = node_context.non_empty_bins_count[f]
    non_empty_bins = node_context.non_empty_bins[f, :non_empty_bins_count]
    w_samples_train_left = 0.0
    w_samples_train_right = w_samples_train
    w_samples_valid_left = 0
    w_samples_valid_right = w_samples_valid
    y_sum_left = 0.0
    y_sum_right = y_sum_in_bins.sum(axis=0)
    y_sq_sum_left = 0.0
    y_sq_sum_right = y_sq_sum_in_bins.sum(axis=0)
    is_feature_categorical = tree_context.is_categorical[feature]
    # TODO: We don't need to instantiate it here
    bins = np.empty(non_empty_bins_count, dtype=np.uint8)

    if is_feature_categorical:
        idx_sort = np.argsort(y_sum_in_bins[non_empty_bins])
        bins[:] = non_empty_bins[idx_sort]
    else:
        bins[:] = np.sort(non_empty_bins)

    for bin_threshold in bins[:-1]:
        w_samples_train_left += w_samples_train_in_bins[bin_threshold]
        w_samples_valid_left += w_samples_valid_in_bins[bin_threshold]
        w_samples_train_right -= w_samples_train_in_bins[bin_threshold]
        w_samples_valid_right -= w_samples_valid_in_bins[bin_threshold]
        y_sum_left += y_sum_in_bins[bin_threshold]
        y_sum_right -= y_sum_in_bins[bin_threshold]
        y_sq_sum_left += y_sq_sum_in_bins[bin_threshold]
        y_sq_sum_right -= y_sq_sum_in_bins[bin_threshold]

        # TODO: this test can be removed now since we use non-empty bins
        if (w_samples_train_left <= 0.0) or (w_samples_valid_left <= 0.0):
            continue

        # TODO: this test can be removed now since we use non-empty bins
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

        if gain_proxy > best_split.gain_proxy:
            best_split.found_split = True
            best_split.gain_proxy = gain_proxy
            best_split.feature = feature
            best_split.bin_threshold = bin_threshold
            best_split.impurity_left = impurity_left
            best_split.impurity_right = impurity_right
            best_split.w_samples_train_left = w_samples_train_left
            best_split.w_samples_train_right = w_samples_train_right
            best_split.w_samples_valid_left = w_samples_valid_left
            best_split.w_samples_valid_right = w_samples_valid_right
            best_split.y_sum_left = y_sum_left
            best_split.y_sum_right = y_sum_right
            best_split.y_sq_sum_left = y_sq_sum_left
            best_split.y_sq_sum_right = y_sq_sum_right
            best_split.is_split_categorical = is_feature_categorical
            best_split.bins[:non_empty_bins_count] = bins
            best_split.bins_count = non_empty_bins_count


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
    tree_context, node_context, find_best_split_along_feature, best_split
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

    best_split : SplitClassifier or SplitRegressor
        The best split found. If there no best split, then best_split.found_split is
        set to False
    """
    # Get the set of features to try out
    features = node_context.features_sampled
    # Loop over the possible features
    init_split(best_split)
    for f, feature in enumerate(features):
        if node_context.non_empty_bins_count[f] > 1:
            # If there is only a single bin value for this feature, it makes no sense
            # to find a split along it, so we skip such cases
            find_best_split_along_feature(
                tree_context, node_context, feature, f, best_split
            )

    # If the split is on a categorical feature, we must store the bin partition
    # correctly
    if best_split.found_split and best_split.is_split_categorical:
        compute_bin_partition(best_split)


@jit(
    [void(SplitClassifierType), void(SplitRegressorType)],
    nopython=True,
    nogil=True,
    locals={
        "bin_threshold": uint8,
        "bins": uint8[::1],
        "bin_partition": uint8[::1],
        "idx_bin": uint8,
        "bin": uint8,
        "left_partition_size": uint8,
        "right_partition_size": uint8,
    },
)
def compute_bin_partition(best_split):
    """Computes the split partition of the best split.

    Parameters
    ----------
    tree_context : TreeClassifierContext or TreeRegressorContext
        The tree context which contains all the data about the tree that is useful to
        find a split

    best_split : SplitClassifier or SplitRegressor
        The best split found for which we want to compute the correct bin_partition
        and bin_partition_size attributes

    Example
    -------
    Let us consider the following examples where we have a categorical features with
    bins [0, 1, 2, 3, 4] and let's say that ordered bins is [3, 0, 4, 2, 1].

    * Example 1. If bin_threshold == 2 then the partition is {[3, 0, 4, 2], [1]}.
      In this case bin_partition = [1] and bin_partition_size = 1

    * Example 2. If bin_threshold == 3 then the partition is {[3], [0, 4, 2, 1]}
      In this case bin_partition = [3] and bin_partition_size = 1

    * Example 3. If bin_threshold == 4 then the partition is {[3, 0, 4], [2, 1]}
      In this case bin_partition = [1, 2] and bin_partition_size = 2
    """
    # Bin threshold used by the best split
    bin_threshold = best_split.bin_threshold
    bins_count = best_split.bins_count
    bins = best_split.bins[:bins_count]
    # The array in which we save the split partition
    bin_partition = best_split.bin_partition
    idx_bin = 0
    bin_ = bins[idx_bin]
    left_partition_size = 1
    right_partition_size = bins_count - 1

    while bin_ != bin_threshold:
        idx_bin += 1
        left_partition_size += 1
        right_partition_size -= 1
        bin_ = bins[idx_bin]

    if left_partition_size < right_partition_size:
        bin_partition[:left_partition_size] = np.sort(bins[:left_partition_size])
        best_split.bin_partition_size = left_partition_size
    else:
        bin_partition[:right_partition_size] = np.sort(
            bins[left_partition_size:bins_count]
        )
        best_split.bin_partition_size = right_partition_size
        # In this case we need to swap impurity_left and impurity_right in the split,
        # since bins in bin_partition go to the left child of the node.
        best_split.impurity_left, best_split.impurity_right = (
            best_split.impurity_right,
            best_split.impurity_left,
        )


@jit(boolean(uint8, uint8[::1]), nopython=True, nogil=True)
def is_bin_in_partition(bin_, bin_partition):
    """Tests if bin is in bin_partition. This assumes that bin_partition is sorted.

    Parameters
    ----------
    bin_ : int
        The bin

    bin_partition : ndarray
        array of shape (bin_partition_size,) and uint8 dtype corresponding to the bins
        that go to the left child

    Returns
    -------
    output : bool
        True if bin_ is in bin_partition, False otherwise

    """
    # TODO: Maybe we'll need to code the searchsorted in numba to improve
    #  speed ?
    # Since bin_partition is (by construction) sorted in ascending order, if bin_
    #  is in bin_partition, then np.searchsorted(bin_partition, bin_) returns bin_.
    #  Otherwise, it returns the insertion index of bin_ so that bin_partition[idx] !=
    #  bin_
    return bin_partition[np.searchsorted(bin_partition, bin_)] == bin_


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
        "bin_partition": uint8[::1],
        "xif_in_partition": boolean,
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
    """This function updates both partition_train, partition_valid, pos_train,
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
    Xf = tree_context.X.T[feature]
    left_buffer = tree_context.left_buffer
    right_buffer = tree_context.right_buffer
    # The current training sample indices in the node
    partition_train = tree_context.partition_train
    # The current validation sample indices in the node
    partition_valid = tree_context.partition_valid

    bin_threshold = split.bin_threshold
    bin_partition = split.bin_partition[: split.bin_partition_size]

    n_samples_train_left = 0
    n_samples_train_right = 0
    if split.is_split_categorical:
        for i in partition_train[start_train:end_train]:
            if is_bin_in_partition(Xf[i], bin_partition):
                left_buffer[n_samples_train_left] = i
                n_samples_train_left += 1
            else:
                right_buffer[n_samples_train_right] = i
                n_samples_train_right += 1
    else:
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
    if split.is_split_categorical:
        for i in partition_valid[start_valid:end_valid]:
            if is_bin_in_partition(Xf[i], bin_partition):
                left_buffer[n_samples_valid_left] = i
                n_samples_valid_left += 1
            else:
                right_buffer[n_samples_valid_right] = i
                n_samples_valid_right += 1
    else:
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
