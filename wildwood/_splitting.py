import numpy as np


from ._utils import (
    njit,
    jitclass,
    np_uint8,
    nb_uint8,
    np_size_t,
    nb_size_t,
    np_uint32,
    nb_uint32,
    np_float32,
    nb_float32,
    np_uint8,
    nb_uint8,
    infinity,
)

from ._impurity import gini_childs, information_gain_proxy, information_gain


# TODO: mettre aussi weighted_samples_left

spec_split_info = [
    ("gain_proxy", nb_float32),
    ("feature", nb_size_t),
    ("bin", nb_uint8),
    ("n_samples_left", nb_size_t),
    ("n_samples_right", nb_size_t),
    ("weighted_samples_left", nb_float32),
    ("weighted_samples_right", nb_float32),
    ("y_sum_left", nb_float32[::1]),
    ("y_sum_right", nb_float32[::1]),
]


@jitclass(spec_split_info)
class SplitInfo(object):
    """Pure data class to store information about a potential split.
    """

    def __init__(
        self, n_classes,
    ):
        self.gain_proxy = -infinity
        self.feature = 0
        self.bin = 0
        self.n_samples_left = 0
        self.n_samples_right = 0
        self.weighted_samples_left = 0.0
        self.weighted_samples_right = 0
        self.y_sum_left = np.empty(n_classes, dtype=np_float32)
        self.y_sum_right = np.empty(n_classes, dtype=np_float32)


@njit
def copy_split(from_split, to_split):
    to_split.gain_proxy = from_split.gain_proxy
    to_split.feature = from_split.feature
    to_split.bin = from_split.bin
    to_split.n_samples_left = from_split.n_samples_left
    to_split.n_samples_right = from_split.n_samples_right
    to_split.weighted_samples_left = from_split.weighted_samples_left
    to_split.weighted_samples_right = from_split.weighted_samples_right
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


# A pure data class which contains global context information, such as the dataset,
# training and validation indices, etc.
spec_tree_context = [
    # The binned matrix of features
    ("X", nb_uint8[::1, :]),
    # The vector of labels
    ("y", nb_float32[::1]),
    # Sample weights
    ("sample_weights", nb_float32[::1]),
    # Training sample indices for tree growth
    ("train_indices", nb_size_t[::1]),
    # Validation sample indices for tree aggregation
    ("valid_indices", nb_size_t[::1]),
    # Total sample size
    ("n_samples", nb_size_t),
    # Training sample size
    ("n_samples_train", nb_size_t),
    # Validation sample size
    ("n_samples_valid", nb_size_t),
    # The total number of features
    ("n_features", nb_size_t),
    # The number of classes
    ("n_classes", nb_size_t),
    # Maximum number of bins
    ("max_bins", nb_uint8),
    # Actual number of bins used for each feature
    ("n_bins_per_feature", nb_uint8[::1]),
    # Maximum number of features to try for splitting
    ("max_features", nb_size_t),
    ("partition_train", nb_size_t[::1]),
    ("partition_valid", nb_size_t[::1]),
    ("left_buffer", nb_size_t[::1]),
    ("right_buffer", nb_size_t[::1]),
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

        self.partition_train = train_indices.copy()
        self.partition_valid = valid_indices.copy()

        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_samples_train = train_indices.shape[0]
        self.n_samples_valid = valid_indices.shape[0]

        # Two buffers used in the split_indices function
        self.left_buffer = np.empty(n_samples, dtype=np_size_t)
        self.right_buffer = np.empty(n_samples, dtype=np_size_t)


# A local context which contains information for the split and local node data
spec_node_context = [
    # Set of candidate features for splitting
    ("features", nb_size_t[::1]),
    # ("min_samples_leaf", uint32),
    # ("min_gain_to_split", float32),
    # Number of training samples in each bin for each feature in the node
    ("n_samples_in_bins", nb_size_t[:, ::1]),
    # Histogram data, number of training samples in each bin for each feature and
    # each label class in the node
    ("y_sum", nb_float32[:, :, ::1]),
    # Sum of the label for each class
    ("y_pred", nb_float32[::1]),
    # Total number of samples in the node
    ("n_samples", nb_size_t),
    # Number of training samples in the node
    ("n_samples_train", nb_size_t),
    # Weighted number of samples in the node
    ("weighted_n_samples_train", nb_float32),
    # Number of validation samples in the node
    ("n_samples_valid", nb_size_t),
    # context.partition_train[start_train:end_train] contains the training sample
    # indices in the node
    ("start_train", nb_size_t),
    ("end_train", nb_size_t),
    # context.partition_valid[start_valid:end_valid] contains the validation sample
    # indices in the node
    ("start_valid", nb_size_t),
    ("end_valid", nb_size_t),
]


@jitclass(spec_node_context)
class NodeContext:
    def __init__(
        self,
        context,
        # TODO: features a tirer au hasard plus tard
        # features
    ):
        self.features = np.arange(0, context.n_features, dtype=np_size_t)
        max_features = context.max_features
        max_bins = context.max_bins
        n_classes = context.n_classes
        self.n_samples_in_bins = np.empty((max_features, max_bins), dtype=np_float32)
        self.y_sum = np.empty((max_features, max_bins, n_classes), dtype=np_float32)
        self.y_pred = np.empty((n_classes,), dtype=np_float32)


# init_node_context ou update_node_context ?


@njit
def init_node_context(tree_context, node_context, node_record):
    """
    Initialize the node_context with the given node_record

    Parameters
    ----------
    node_context
    node_record

    Returns
    -------

    """
    # TODO: initialize node_context.features with Fisher-Yates ?

    start_train = node_record["start_train"]
    end_train = node_record["end_train"]
    start_valid = node_record["start_valid"]
    end_valid = node_record["end_valid"]

    # node_context.start_train = start_train
    # node_context.end_train = end_train
    # node_context.start_valid = start_valid
    # node_context.end_valid = end_valid

    # TODO: c'est utile node_context.n_samples ?

    n_samples_in_bins = node_context.n_samples_in_bins
    n_samples_in_bins[:] = 0
    y_sum = node_context.y_sum
    y_sum[:] = 0
    y_pred = node_context.y_pred
    y_pred[:] = 0

    # Get tree context information
    X = tree_context.X
    y = tree_context.y
    sample_weights = tree_context.sample_weights
    partition_train = tree_context.partition_train
    features = node_context.features

    # Get the node training indices
    # start_train = node_context.start_train
    # end_train = node_context.end_train

    # The indices of the training samples contained in the node
    samples = partition_train[start_train:end_train]

    # For-loop on features and then samples (X is F-major) for cache hits
    # TODO: unrolling this for loop could be faster

    weighted_n_samples_train = nb_float32(0.0)

    # A counter for the features since return arrays are contiguous
    f = nb_size_t(0)

    for feature in features:
        for sample in samples:
            bin = X[sample, feature]
            label = nb_size_t(y[sample])
            sample_weight = sample_weights[sample]
            if f == 0:
                weighted_n_samples_train += sample_weight
                print("label: ", label)
                print("y_pred: ", y_pred)
                y_pred[label] += sample_weight
            # One more sample in this bin for the current feature
            print("f: ", f, ", bin: ", bin)
            print("n_samples_in_bins: ", n_samples_in_bins)
            n_samples_in_bins[f, bin] += sample_weight
            # One more sample in this bin for the current feature with this label
            y_sum[f, bin, label] += sample_weight

        f += nb_size_t(1)

    # Save everything in the node context

    node_context.n_samples_train = end_train - start_train
    node_context.n_samples_valid = end_valid - start_valid
    node_context.weighted_n_samples_train = weighted_n_samples_train


# @njit
# def compute_node_context_statistics(tree_context, node_context):
#     """
#     TODO: la strategie est bien en fait car les histogrammes sont petits comme en
#     principe on sous-echantillone les features
#
#     """
#     # TODO: c'est ici qu'on peut calculer quelles sont les features constantes dans
#     #  le node (si on trouve un bin avec toutes les features. Faudra faire attention
#     #  au cas avec missing values a terme
#     n_samples_in_bins = local_context.n_samples_in_bins
#     n_samples_in_bins[:] = 0
#     y_sum_in_bins = local_context.y_sum_in_bins
#     y_sum_in_bins[:] = 0
#
#     node_context.n_samples_train = end_train - start_train
#     node_context.n_samples_valid = end_valid - start_valid
#
#
#     # Get tree context information
#     X = context.X
#     y = context.y
#     sample_weights = context.sample_weights
#     partition_train = context.partition_train
#     features = local_context.features
#
#     # Get the node training indices
#     train_indices = context.train_indices
#     start_train = local_context.start_train
#     end_train = local_context.end_train
#
#     samples = train_indices[start_train:end_train]
#
#     # A counter for the features since return arrays are contiguous
#     f = nb_uint32(0)
#     # For-loop on features and then samples (X is F-major) for cache hits
#     for feature in features:
#         for sample in samples:
#             bin = X[sample, feature]
#             label = nb_size_t(y[sample])
#             sample_weight = sample_weights[sample]
#             # One more sample in this bin for the current feature
#             n_samples_in_bins[f, bin] += sample_weight
#             # One more sample in this bin for the current feature with this label
#             y_sum_in_bins[f, bin, label] += sample_weight
#         f += 1


@njit
def find_node_split(tree_context, local_context):

    features = local_context.features
    # Loop over the possible features

    # TODO: a quoi ca sert vraiment de recuperer plein de split_info ?
    # # Pre-allocate the results datastructure to be able to use prange:
    # # numba jitclass do not seem to properly support default values for kwargs.
    # split_infos = [SplitInfo(-1., 0, 0, 0., 0., 0., 0., 0, 0)
    #                for i in range(context.n_features)]

    # best_feature = 0
    # best_bin = 0
    best_gain_proxy = -infinity

    # Ne les initialiser qu'une seule fois... donc en dehors d'ici ? Dans le
    # local_context ?
    best_split = SplitInfo(tree_context.n_classes)
    candidate_split = SplitInfo(tree_context.n_classes)

    for feature in features:
        # Compute the best bin and gain proxy obtained for the feature
        find_best_split_along_feature(
            tree_context, local_context, feature, candidate_split
        )
        if candidate_split.gain_proxy > best_gain_proxy:
            # We've found a better split
            copy_split(candidate_split, best_split)
            best_gain_proxy = candidate_split.gain_proxy

    # TODO: ici faut calculer le vrai gain et le mettre dans le best split ?

    return best_split


@njit
def find_best_split_along_feature(tree_context, node_context, feature, best_split):
    """

    Parameters
    ----------
    tree_context
    feature
    n_samples
    n_samples_in_bins

    y_sums_in_bins : ndarray of shape (n_bins, n_classes), dtype=float32

    Returns
    -------
    retourne rien, le resultat est dans candidate split

    """
    # TODO: n_samples_in_bins c'est n_samples_in_bins[feature] (la ligne concerant
    #  juste la feature)

    # TODO: a terme utiliser ca :
    # n_bins = context.n_bins_per_feature[feature]
    # n_bins = 255

    n_classes = tree_context.n_classes
    n_bins = tree_context.max_bins

    # Number of training samples in the node
    n_samples_train = node_context.n_samples_train
    # Get the number of samples in each bin for the feature
    n_samples_in_bins = node_context.n_samples_in_bins[feature]
    # Get the sum of labels (counts) in each bin for the feature
    y_sum_in_bins = node_context.y_sum[feature]

    # TODO: faudra que les vecteurs left et right soient dans le local_context
    # Counts on the left are zero, since we go from left to right
    n_samples_left = 0
    y_sum_left = np.zeros(n_classes, dtype=np_float32)
    # Count on the right contain everything
    n_samples_right = n_samples_train
    y_sum_right = np.empty(n_classes, dtype=np_float32)
    y_sum_right[:] = y_sum_in_bins.sum(axis=0)

    best_bin = nb_uint8(0)
    best_gain_proxy = -infinity

    # TODO: right to left also for features with missing values
    # TODO: this works only for ordered features... special sorting for categorical
    # We go from left to right and compute the information gain proxy of all possible
    # splits
    for bin in range(n_bins):
        # On the left we accumulate the counts
        n_samples_left += n_samples_in_bins[bin]
        y_sum_left += y_sum_in_bins[bin]
        # And we get the counts on the right from the left
        n_samples_right -= n_samples_in_bins[bin]
        y_sum_right -= y_sum_in_bins[bin]

        # Compute the information gain proxy
        gain_proxy = information_gain_proxy(
            gini_childs,
            n_classes,
            n_samples_left,
            n_samples_right,
            y_sum_left,
            y_sum_right,
        )

        if gain_proxy > best_gain_proxy:
            # We've found a better split
            best_gain_proxy = gain_proxy

            best_split.gain_proxy = gain_proxy
            best_split.feature = feature
            best_split.bin = bin
            best_split.n_samples_left = n_samples_left
            best_split.n_samples_right = n_samples_right
            # TODO: faudra calculer les weighted version
            best_split.weighted_samples_left = n_samples_left
            best_split.weighted_samples_right = n_samples_right
            best_split.y_sum_left[:] = y_sum_left
            best_split.y_sum_right[:] = y_sum_right


@njit
def split_indices(tree_context, split, node_record):

    # The feature and bin used for this split
    feature = split.feature
    bin = split.bin

    # TODO: pourquoi transposition ici ?
    Xf = tree_context.X.T[feature]

    left_buffer = tree_context.left_buffer
    right_buffer = tree_context.right_buffer

    # TODO: faudrait avoir le calcul du nombre de train et valid dans le split

    start_train = node_record["start_train"]
    end_train = node_record["end_train"]
    start_valid = node_record["start_valid"]
    end_valid = node_record["end_valid"]

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
    n_samples_train_left = nb_size_t(0)
    n_samples_train_right = nb_size_t(0)
    for i in partition_train:
        if Xf[i] <= bin:
            left_buffer[n_samples_train_left] = i
            n_samples_train_left += nb_size_t(1)
        else:
            right_buffer[n_samples_train_right] = i
            n_samples_train_right += nb_size_t(1)

    print("start_train: ", start_train, ", n_samples_train_left: ",
          n_samples_train_left, ", n_samples_train_right: ", n_samples_train_right,
          ", end_train: ", end_train)


    print("partition_train[start_train:end_train]: ", partition_train[start_train:end_train])
    print("left_buffer[:n_samples_train_left]: ", left_buffer[:n_samples_train_left])
    print("right_buffer[:n_samples_train_right]: ", right_buffer[:n_samples_train_right])

    # start_train + n_samples_train_left + n_samples_train_right

    pos_train = start_train + n_samples_train_left
    partition_train[start_train:pos_train] = left_buffer[:n_samples_train_left]
    partition_train[pos_train:end_train] = right_buffer[:n_samples_train_right]

    # We must have start_train + n_samples_train_left + n_samples_train_right ==
    # end_train

    n_samples_valid_left = nb_size_t(0)
    n_samples_valid_right = nb_size_t(0)
    for i in partition_valid:
        if Xf[i] <= bin:
            left_buffer[n_samples_valid_left] = i
            n_samples_valid_left += nb_size_t(1)
        else:
            right_buffer[n_samples_valid_right] = i
            n_samples_valid_right += nb_size_t(1)

    pos_valid = start_valid + n_samples_valid_left
    partition_valid[start_valid:pos_valid] = left_buffer[:n_samples_valid_left]
    partition_valid[pos_valid:end_valid] = right_buffer[:n_samples_valid_right]

    return pos_train, pos_valid
