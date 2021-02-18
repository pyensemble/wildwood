# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains all functions and dataclasses required to find the best split in
a node.
"""

import numpy as np
from numba import (
    jit,
    njit,
    boolean,
    uint8,
    intp,
    uintp,
    float32,
    void,
    optional,
)
from numba.experimental import jitclass

from ._node import NodeContextType
from ._tree_context import TreeContextType
from ._impurity import gini_childs, information_gain_proxy, information_gain

from ._utils import get_type


split_type = [
    # True if we found a split, False otherwise
    ("found_split", boolean),
    # Information gain proxy obtained thanks to this split
    ("gain_proxy", float32),
    # Feature used to split the node
    ("feature", uintp),
    # Index of the bin threshold used to split the node
    ("bin_threshold", uint8),
    # Number of training samples in the left child node
    ("n_samples_train_left", uintp),
    # Number of training samples in the right child node
    ("n_samples_train_right", uintp),
    # Weighted number of training samples in the left child node
    ("w_samples_train_left", float32),
    # Weighted number of training samples in the right child node
    ("w_samples_train_right", float32),
    # Number of validation samples in the left child node
    ("n_samples_valid_left", uintp),
    # Number of validation samples in the right child node
    ("n_samples_valid_right", uintp),
    # Weighted number of validation samples in the left child node
    ("w_samples_valid_left", float32),
    # Weighted number of validation samples in the right child node
    ("w_samples_valid_right", float32),
    # Weighted number of training samples for each label class in the left child node
    ("y_sum_left", float32[::1]),
    # Weighted number of training samples for each label class in the right child node
    ("y_sum_right", float32[::1]),
    # Impurity of the left child node
    ("impurity_left", float32),
    # Impurity of the right child node
    ("impurity_right", float32),
]


@jitclass(split_type)
class Split(object):
    """Pure dataclass to store information about a potential split.

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
        self.found_split = False
        self.gain_proxy = -np.inf
        self.feature = 0
        self.bin_threshold = 0
        self.n_samples_train_left = 0
        self.n_samples_train_right = 0
        self.w_samples_train_left = 0.0
        self.w_samples_train_right = 0.0
        self.n_samples_valid_left = 0
        self.n_samples_valid_right = 0
        self.w_samples_valid_left = 0.0
        self.w_samples_valid_right = 0.0
        self.y_sum_left = np.empty(n_classes, dtype=np.float32)
        self.y_sum_right = np.empty(n_classes, dtype=np.float32)
        self.impurity_left = 0.0
        self.impurity_right = 0.0


SplitType = get_type(Split)


@jit(void(SplitType, SplitType), nopython=True, nogil=True)
def copy_split(from_split, to_split):
    """Copies split data from from_split into to_split

    Parameters
    ----------
    from_split : SplitType
        Copy data from this split

    to_split : SplitType
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
    to_split.y_sum_left[:] = from_split.y_sum_left
    to_split.y_sum_right[:] = from_split.y_sum_right


# TODO: We do not handle missing values yet: it will require a right to left also for
#  features with missing values

# TODO: For now, find_split works only for ordered features... we'll need to
#  code a special sorting for categorical variables


@jit(
    void(TreeContextType, NodeContextType, uintp, SplitType),
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
        # TODO: put back this
        # "y_sum_in_bins": float32[::1],
        "n_samples_train_left": uintp,
        "n_samples_train_right": uintp,
        "w_samples_train_left": float32,
        "w_samples_train_right": float32,
        "w_samples_valid_left": float32,
        "w_samples_valid_right": float32,
        "y_sum_left": float32[::1],
        "y_sum_right": float32[::1],
        "best_gain_proxy": float32,
    },
)
def find_best_split_along_feature(tree_context, node_context, feature, best_split):
    """Finds the best split (best bin_threshold) for a feature. If no split can be
    found (this happens when we can't find a split with a large enough weighted
    number of training and validation samples in the left and right childs) we simply
    set best_split.found_split to False).

    Parameters
    ----------
    tree_context : TreeContextType
        The tree context which contains all the data about the tree that is useful to
        find a split

    node_context : NodeContextType
        The node context which contains all the data about the node required to find
        a best split for it

    feature : uint
        The index of the feature for which we want to find a split

    best_split : SplitType
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
    w_samples_train_in_bins = node_context.w_samples_train_in_bins[feature]
    # Weighted number of validation samples in each bin for the feature
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins[feature]
    # Get the sum of labels (counts) in each bin for the feature
    y_sum_in_bins = node_context.y_sum[feature]

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

    best_gain_proxy = -np.inf

    # TODO: faut aussi rejeter un split qui a weighted_n_samples_valid_left ou
    #  weighted_n_samples_valid_right a 0 verifier
    #  que le
    #  nombre de
    #  noeuds

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

        # TODO: on peut pas mettre ca apres les tests ?
        y_sum_left += y_sum_in_bins[bin]
        y_sum_right -= y_sum_in_bins[bin]

        # If the split would lead to 0 training or 0 validation samples in the left
        # child then we don't consider the split
        if (w_samples_train_left <= 0.0) or (w_samples_valid_left <= 0.0):
            continue

        # If the split would lead to 0 training or 0 validation samples in the right,
        # and since we go from left to right, no other future bin on the right would
        # lead to an acceptable split, so we break the for loop over bins.
        if (w_samples_train_right <= 0.0) or (w_samples_valid_right <= 0.0):
            break

        # Compute the information gain proxy

        # Get the impurities of the left and right childs
        impurity_left, impurity_right = gini_childs(
            n_classes,
            w_samples_train_left,
            w_samples_train_right,
            y_sum_left,
            y_sum_right,
        )

        gain_proxy = information_gain_proxy(
            impurity_left, impurity_right, w_samples_train_left, w_samples_train_right,
        )

        if gain_proxy > best_gain_proxy:
            # We've found a better split
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
        else:
            pass


@jit(SplitType(TreeContextType, NodeContextType), nopython=True, nogil=True)
def find_node_split(tree_context, node_context):
    # print("================ Begin find_node_split ================")
    features = node_context.features
    # Loop over the possible features

    # TODO: a quoi ca sert vraiment de recuperer plein de split_info ?
    # # Pre-allocate the results datastructure to be able to use prange:
    # # numba jitclass do not seem to properly support default values for kwargs.
    # split_infos = [SplitInfo(-1., 0, 0, 0., 0., 0., 0., 0, 0)
    #                for i in range(context.n_features)]

    # best_feature = 0
    # best_bin = 0
    best_gain_proxy = -np.inf

    # Ne les initialiser qu'une seule fois... donc en dehors d'ici ? Dans le
    # local_context ?
    best_split = Split(tree_context.n_classes)
    candidate_split = Split(tree_context.n_classes)

    for feature in features:
        # Compute the best bin and gain proxy obtained for the feature
        find_best_split_along_feature(
            tree_context, node_context, feature, candidate_split
        )

        # print("For feature: ", feature, " found a best split")
        # print("found_split:", candidate_split.found_split)
        # print("bin:", candidate_split.bin)
        # print("gain_proxy:", candidate_split.gain_proxy)

        # If we found a candidate split along the feature
        if candidate_split.found_split:
            # And if it's better than the current one
            if candidate_split.gain_proxy >= best_gain_proxy:
                # Then we replace the best current split
                copy_split(candidate_split, best_split)
                best_gain_proxy = candidate_split.gain_proxy

        # exit(0)

    # TODO: ici faut calculer le vrai gain et le mettre dans le best split ?

    # print("Best split is feature: ", best_split.feature,
    #       ", bin:", best_split.bin, "gain proxy: ", best_split.gain_proxy)
    #
    # print("================ End   find_node_split ================")

    return best_split


@njit
def split_indices(tree_context, split, start_train, end_train, start_valid, end_valid):

    # The feature and bin used for this split
    feature = split.feature
    bin = split.bin_threshold

    # TODO: pourquoi transposition ici ?
    Xf = tree_context.X.T[feature]

    left_buffer = tree_context.left_buffer
    right_buffer = tree_context.right_buffer

    # TODO: faudrait avoir le calcul du nombre de train et valid dans le split

    # start_train = node_record["start_train"]
    # end_train = node_record["end_train"]
    # start_valid = node_record["start_valid"]
    # end_valid = node_record["end_valid"]

    # The current training sample indices in the node
    partition_train = tree_context.partition_train
    # The current validation sample indices in the node
    partition_valid = tree_context.partition_valid

    # We want to update both partition_train, partition_valid, pos_train, pos_valid so
    # that:
    # - partition_train[start_train:pos_train] contains the training sample indices of
    #   the left child
    # - partition_train[pos_train:end_train] contains the training sample indices of
    #   the right child
    # - partition_valid[start_valid:pos_valid] contains the validation sample indices
    #   of the left child
    # - partition_valid[pos_valid:end_valid] contains the validation sample indices of
    #   the right child

    # TODO: c'est bourrin et peut etre pas optimal mais ca suffira pour l'instant
    n_samples_train_left = uintp(0)
    n_samples_train_right = uintp(0)
    for i in partition_train[start_train:end_train]:
        if Xf[i] <= bin:
            left_buffer[n_samples_train_left] = i
            n_samples_train_left += uintp(1)
        else:
            right_buffer[n_samples_train_right] = i
            n_samples_train_right += uintp(1)

    # print("start_train: ", start_train, ", n_samples_train_left: ",
    #       n_samples_train_left, ", n_samples_train_right: ", n_samples_train_right,
    #       ", end_train: ", end_train)

    # print("partition_train[start_train:end_train]: ", partition_train[start_train:end_train])
    # print("left_buffer[:n_samples_train_left]: ", left_buffer[:n_samples_train_left])
    # print("right_buffer[:n_samples_train_right]: ", right_buffer[:n_samples_train_right])

    # start_train + n_samples_train_left + n_samples_train_right

    pos_train = start_train + n_samples_train_left
    partition_train[start_train:pos_train] = left_buffer[:n_samples_train_left]
    partition_train[pos_train:end_train] = right_buffer[:n_samples_train_right]

    # We must have start_train + n_samples_train_left + n_samples_train_right ==
    # end_train

    n_samples_valid_left = uintp(0)
    n_samples_valid_right = uintp(0)
    for i in partition_valid[start_valid:end_valid]:
        if Xf[i] <= bin:
            left_buffer[n_samples_valid_left] = i
            n_samples_valid_left += uintp(1)
        else:
            right_buffer[n_samples_valid_right] = i
            n_samples_valid_right += uintp(1)

    pos_valid = start_valid + n_samples_valid_left
    partition_valid[start_valid:pos_valid] = left_buffer[:n_samples_valid_left]
    partition_valid[pos_valid:end_valid] = right_buffer[:n_samples_valid_right]

    return pos_train, pos_valid
