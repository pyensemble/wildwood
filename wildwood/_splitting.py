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

from ._impurity import gini_childs, information_gain_proxy, information_gain

from ._utils import get_type


# TODO: mettre aussi weighted_samples_left

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


# @njit
# def print_split(split):
#     print("Split(gain_proxy=", split.gain_proxy)
#     print("Split(gain_proxy=)", split.gain_proxy)
#     split.gain_proxy = -infinity
#     split.feature = 0
#     split.bin = 0
#     split.n_samples_left = 0
#     split.n_samples_right = 0
#     split.weighted_samples_left = 0.0
#     split.weighted_samples_right = 0
#     split.y_sum_left = np.empty(n_classes, dtype=np_float32)
#     split.y_sum_right = np.empty(n_classes, dtype=np_float32)

# TODO: a mettre dans _tree
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
    ("partition_train", uintp[::1]),
    ("partition_valid", uintp[::1]),
    ("left_buffer", uintp[::1]),
    ("right_buffer", uintp[::1]),
]

# TODO: a mettre dans _tree
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

        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_samples_train = train_indices.shape[0]
        self.n_samples_valid = valid_indices.shape[0]

        # Two buffers used in the split_indices function
        self.left_buffer = np.empty(n_samples, dtype=uintp)
        self.right_buffer = np.empty(n_samples, dtype=uintp)


@njit
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


TreeContextType = get_type(TreeContext)


@njit
def find_best_split_along_feature(tree_context, node_context, feature, best_split):
    # TODO: n_samples_in_bins c'est n_samples_in_bins[feature] (la ligne concerant
    #  juste la feature)

    # TODO: a terme utiliser ca :
    # n_bins = context.n_bins_per_feature[feature]
    # n_bins = 255

    # print("================ Begin find_best_split_along_feature ================")

    n_classes = tree_context.n_classes
    n_bins = tree_context.max_bins

    # Number of training samples in the node
    n_samples_train = node_context.n_samples_train
    # Weighted number training samples in the node
    w_samples_train = node_context.w_samples_train
    # Number of validation samples in the node
    n_samples_valid = node_context.n_samples_valid
    # Weighted number of validation samples in the node
    w_samples_valid = node_context.w_samples_valid
    # Weighed number of training samples in each bin for the feature
    w_samples_train_in_bins = node_context.w_samples_train_in_bins[feature]
    # Weighted number of validation samples in each bin for the feature
    w_samples_valid_in_bins = node_context.w_samples_valid_in_bins[feature]
    # Get the sum of labels (counts) in each bin for the feature
    y_sum_in_bins = node_context.y_sum[feature]

    # print("n_samples_in_bins: ", n_samples_in_bins)
    # print("y_sum_in_bins: ", y_sum_in_bins)

    # TODO: faudra que les vecteurs left et right soient dans le local_context
    # Counts and sums on left are zero, since we go from left to right, while counts
    # and sums on the right contain everything
    n_samples_train_left = uintp(0)
    n_samples_train_right = n_samples_train
    # n_samples_valid_left = nb_size_t(0)
    # n_samples_valid_right = n_samples_valid

    w_samples_train_left = float32(0)
    w_samples_train_right = w_samples_train
    w_samples_valid_left = float32(0)
    w_samples_valid_right = w_samples_valid

    # print("w_samples_train_left:", w_samples_train_left)
    # print("w_samples_train_right:", w_samples_train_right)
    # print("w_samples_valid_left:", w_samples_valid_left)
    # print("w_samples_valid_right:", w_samples_valid_right)

    y_sum_left = np.zeros(n_classes, dtype=np.float32)
    y_sum_right = np.empty(n_classes, dtype=np.float32)
    y_sum_right[:] = y_sum_in_bins.sum(axis=0)

    best_bin = uint8(0)
    best_gain_proxy = -np.inf

    # TODO: right to left also for features with missing values
    # TODO: this works only for ordered features... special sorting for categorical
    # We go from left to right and compute the information gain proxy of all possible
    # splits

    # print("=" * 64)
    # print("weighted_n_samples_right: ", weighted_n_samples_right)
    # print("weighted_n_samples_train: ", weighted_n_samples_train)

    # TODO: faut aussi rejeter un split qui a weighted_n_samples_valid_left ou
    #  weighted_n_samples_valid_right a 0 verifier
    #  que le
    #  nombre de
    #  noeuds

    # Did we find a split ?
    best_split.found_split = False

    for bin in range(n_bins):
        # On the left we accumulate the counts
        w_samples_train_left += w_samples_train_in_bins[bin]
        w_samples_valid_left += w_samples_valid_in_bins[bin]
        # On the right we remove the counts
        w_samples_train_right -= w_samples_train_in_bins[bin]
        w_samples_valid_right -= w_samples_valid_in_bins[bin]

        # print("w_samples_train_left: ", w_samples_train_left)
        # print("w_samples_train_left: ", w_samples_train_left)
        # print("weighted_n_samples_right: ", weighted_n_samples_right)

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

        # print("weighted_n_samples_left: ", weighted_n_samples_left)
        # print("weighted_n_samples_right: ", weighted_n_samples_right)

        # print("bin: ", bin)
        # print("weighted_n_samples_train[bin]: ", weighted_n_samples_train_in_bins[bin])
        # print("weighted_n_samples_left: ", weighted_n_samples_left, "weighted_n_samples_right: ", weighted_n_samples_right)

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
            # print("================ Found better split with ================:")
            # print("bin: ", bin)
            # print(
            #     "w_samples_train_left: ",
            #     w_samples_train_left,
            #     ", w_samples_train_right: ",
            #     w_samples_train_right,
            # )
            # print(
            #     "w_samples_valid_left: ",
            #     w_samples_valid_left,
            #     ", w_samples_valid_right: ",
            #     w_samples_valid_right,
            # )
            # print("gain_proxy: ", gain_proxy)

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
            # print("bin: ", bin)
            # print(
            #     "w_samples_train_left: ",
            #     w_samples_train_left,
            #     ", w_samples_train_right: ",
            #     w_samples_train_right,
            # )
            # print(
            #     "w_samples_valid_left: ",
            #     w_samples_valid_left,
            #     ", w_samples_valid_right: ",
            #     w_samples_valid_right,
            # )
            # print("gain_proxy: ", gain_proxy)

    # print("Best split is bin:", best_split.bin, "gain proxy: ", gain_proxy)
    # print("================ End   find_best_split_along_feature ================")
    # exit(0)


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
