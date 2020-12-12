"""

This is for now mostly a translation of scikit-learn's _splitter.pxd and
_splitter.pyx code into numba


"""

#
#
# Translation of the code from _splitter.pxd
#
#

from math import log as ln
import numpy as np

from ._criterion import (
    criterion_reset,
    criterion_update,
    criterion_impurity_improvement,
    classification_criterion_init,
    # classification_criterion_node_value,
)

from ._utils import (
    njit,
    jitclass,
    int32,
    uint32,
    DOUBLE_t,
    NP_DOUBLE_t,
    SIZE_t,
    INT32_t,
    UINT32_t,
    INFINITY,
    NP_SIZE_t,
    DTYPE_t,
    NP_DTYPE_t,
    get_numba_type,
)

from ._criterion import (
    Gini,
    classification_criterion_init,
    gini_children_impurity,
    gini_node_impurity,
    criterion_proxy_impurity_improvement,
)

from collections import namedtuple
from numba import intp, int32, uint32, float64
from numpy import isnan

# from cmath import isnan
# from _utils import get_numba_type

# TODO: correct ?
NULL = np.nan


FEATURE_THRESHOLD = DTYPE_t(1e-7)

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
EXTRACT_NNZ_SWITCH = DTYPE_t(0.1)


# cdef struct SplitRecord:
#     # Data to track sample split
#     SIZE_t feature         # Which feature to split on.
#     SIZE_t pos             # Split samples array at the given position,
#                            # i.e. count of samples below threshold for feature.
#                            # pos is >= end if the node is a leaf.
#     double threshold       # Threshold to split at.
#     double improvement     # Impurity improvement given parent node.
#     double impurity_left   # Impurity of the left split.
#     double impurity_right  # Impurity of the right split.


spec_split_record = [
    ("feature", SIZE_t),
    ("pos", SIZE_t),
    ("threshold", DOUBLE_t),
    ("improvement", DOUBLE_t),
    ("impurity_left", DOUBLE_t),
    ("impurity_right", DOUBLE_t),
]


@jitclass(spec_split_record)
class SplitRecord(object):

    # def __init__(self, feature, pos, threshold, improvement, impurity_left,
    #              impurity_right):
    #     self.feature = feature
    #     self.pos = pos
    #     self.threshold = threshold
    #     self.improvement = improvement
    #     self.impurity_left = impurity_left
    #     self.impurity_right = impurity_right

    def __init__(self):
        self.impurity_left = INFINITY
        self.impurity_right = INFINITY
        self.pos = 0
        self.feature = 0
        self.threshold = 0.0
        self.improvement = -INFINITY


@njit
def split_record_copy(to_record, from_record):
    to_record.impurity_left = from_record.impurity_left
    to_record.impurity_right = from_record.impurity_right
    to_record.pos = from_record.pos
    to_record.feature = from_record.feature
    to_record.threshold = from_record.threshold
    to_record.improvement = from_record.improvement


# cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
#     self.impurity_left = INFINITY
#     self.impurity_right = INFINITY
#     self.pos = start_pos
#     self.feature = 0
#     self.threshold = 0.
#     self.improvement = -INFINITY
@njit
def _init_split(split_record, start_pos):
    split_record.impurity_left = INFINITY
    split_record.impurity_right = INFINITY
    split_record.pos = start_pos
    split_record.feature = 0
    split_record.threshold = 0.0
    split_record.improvement = -INFINITY


# cdef class Splitter:
#     # The splitter searches in the input space for a feature and a threshold
#     # to split the samples samples[start:end].
#     #
#     # The impurity computations are delegated to a criterion object.
#
#     # Internal structures
#     cdef public Criterion criterion      # Impurity criterion
#     cdef public SIZE_t max_features      # Number of features to test
#     cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
#     cdef public double min_weight_leaf   # Minimum weight in a leaf
#
#     cdef object random_state             # Random state
#     cdef UINT32_t rand_r_state           # sklearn_rand_r random number state
#
#     cdef SIZE_t* samples                 # Sample indices in X, y
#     cdef SIZE_t n_samples                # X.shape[0]
#     cdef double weighted_n_samples       # Weighted number of samples
#     cdef SIZE_t* features                # Feature indices in X
#     cdef SIZE_t* constant_features       # Constant features indices
#     cdef SIZE_t n_features               # X.shape[1]
#     cdef DTYPE_t* feature_values         # temp. array holding feature values
#
#     cdef SIZE_t start                    # Start position for the current node
#     cdef SIZE_t end                      # End position for the current node
#
#     cdef const DOUBLE_t[:, ::1] y
#     cdef DOUBLE_t* sample_weight
#
#     # The samples vector `samples` is maintained by the Splitter object such
#     # that the samples contained in a node are contiguous. With this setting,
#     # `node_split` reorganizes the node samples `samples[start:end]` in two
#     # subsets `samples[start:pos]` and `samples[pos:end]`.
#
#     # The 1-d  `features` array of size n_features contains the features
#     # indices and allows fast sampling without replacement of features.
#
#     # The 1-d `constant_features` array of size n_features holds in
#     # `constant_features[:n_constant_features]` the feature ids with
#     # constant values for all the samples that reached a specific node.
#     # The value `n_constant_features` is given by the parent node to its
#     # child nodes.  The content of the range `[n_constant_features:]` is left
#     # undefined, but preallocated for performance reasons
#     # This allows optimization with depth-based tree building.
#
#     # Methods
#     cdef int init(self, object X, const DOUBLE_t[:, ::1] y,
#                   DOUBLE_t* sample_weight) except -1
#
#     cdef int node_reset(self, SIZE_t start, SIZE_t end,
#                         double* weighted_n_node_samples) nogil except -1
#
#     cdef int node_split(self,
#                         double impurity,   # Impurity of the node
#                         SplitRecord* split,
#                         SIZE_t* n_constant_features) nogil except -1
#
#     cdef void node_value(self, double* dest) nogil
#
#     cdef double node_impurity(self) nogil


# spec_criterion = []
# @jitclass(spec_criterion)
# class Criterion(object):
#
#     def __init__(self):
#         pass


# cdef class Splitter:
#     # The splitter searches in the input space for a feature and a threshold
#     # to split the samples samples[start:end].
#     #
#     # The impurity computations are delegated to a criterion object.
#
#     # Internal structures
#     cdef public Criterion criterion      # Impurity criterion
#     cdef public SIZE_t max_features      # Number of features to test
#     cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
#     cdef public double min_weight_leaf   # Minimum weight in a leaf
#
#     cdef object random_state             # Random state
#     cdef UINT32_t rand_r_state           # sklearn_rand_r random number state
#
#     cdef SIZE_t* samples                 # Sample indices in X, y
#     cdef SIZE_t n_samples                # X.shape[0]
#     cdef double weighted_n_samples       # Weighted number of samples
#     cdef SIZE_t* features                # Feature indices in X
#     cdef SIZE_t* constant_features       # Constant features indices
#     cdef SIZE_t n_features               # X.shape[1]
#     cdef DTYPE_t* feature_values         # temp. array holding feature values
#
#     cdef SIZE_t start                    # Start position for the current node
#     cdef SIZE_t end                      # End position for the current node
#
#     cdef const DOUBLE_t[:, ::1] y
#     cdef DOUBLE_t* sample_weight


spec_splitter = [
    ("criterion", get_numba_type(Gini)),
    ("max_features", SIZE_t),
    ("min_samples_leaf", SIZE_t),
    ("min_weight_leaf", SIZE_t),
    ("random_state", UINT32_t),  # TODO: this is wrong
    ("rand_r_state", UINT32_t),
    ("max_features", SIZE_t),
    ("samples", SIZE_t[::1]),  # A numpy array holding the sample indices
    ("n_samples", SIZE_t),  # It's X.shape[0]
    ("weighted_n_samples", DOUBLE_t),
    ("features", SIZE_t[::1]),  # Feature indices in X
    ("constant_features", SIZE_t[::1]),  # Constant feature indices
    ("n_features", SIZE_t),  # It's X.shape[0]
    ("max_features", SIZE_t),
    ("feature_values", DOUBLE_t[::1]),
    ("start", SIZE_t),
    ("end", SIZE_t),
    ("y", DOUBLE_t[:, ::1]),
    ("sample_weight", DOUBLE_t[::1]),
]


@njit
def splitter_init(splitter, X, y, sample_weight):
    #     cdef int init(self,
    #                    object X,
    #                    const DOUBLE_t[:, ::1] y,
    #                    DOUBLE_t* sample_weight) except -1:
    #         """Initialize the splitter.
    #
    #         Take in the input data X, the target Y, and optional sample weights.
    #
    #         Returns -1 in case of failure to allocate memory (and raise MemoryError)
    #         or 0 otherwise.
    #
    #         Parameters
    #         ----------
    #         X : object
    #             This contains the inputs. Usually it is a 2d numpy array.
    #
    #         y : ndarray, dtype=DOUBLE_t
    #             This is the vector of targets, or true labels, for the samples
    #
    #         sample_weight : DOUBLE_t*
    #             The weights of the samples, where higher weighted samples are fit
    #             closer than lower weight samples. If not provided, all samples
    #             are assumed to have uniform weight.
    #         """
    #
    #         self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
    #         cdef SIZE_t n_samples = X.shape[0]
    #
    #         # Create a new array which will be used to store nonzero
    #         # samples from the feature of interest
    #         cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)
    #
    #         cdef SIZE_t i, j
    #         cdef double weighted_n_samples = 0.0
    #         j = 0
    #
    #         for i in range(n_samples):
    #             # Only work with positively weighted samples
    #             if sample_weight == NULL or sample_weight[i] != 0.0:
    #                 samples[j] = i
    #                 j += 1
    #
    #             if sample_weight != NULL:
    #                 weighted_n_samples += sample_weight[i]
    #             else:
    #                 weighted_n_samples += 1.0
    #
    #         # Number of samples is number of positively weighted samples
    #         self.n_samples = j
    #         self.weighted_n_samples = weighted_n_samples
    #
    #         cdef SIZE_t n_features = X.shape[1]
    #         cdef SIZE_t* features = safe_realloc(&self.features, n_features)
    #
    #         for i in range(n_features):
    #             features[i] = i
    #
    #         self.n_features = n_features
    #
    #         safe_realloc(&self.feature_values, n_samples)
    #         safe_realloc(&self.constant_features, n_features)
    #
    #         self.y = y
    #
    #         self.sample_weight = sample_weight
    #         return 0

    # self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
    n_samples = X.shape[0]

    if sample_weight is None:
        sample_weight = np.empty(0, dtype=NP_DOUBLE_t)

    # Create a new array which will be used to store nonzero
    # samples from the feature of interest
    # safe_realloc(&self.samples, n_samples)
    samples = np.empty(n_samples, dtype=NP_SIZE_t)
    splitter.samples = samples

    # cdef SIZE_t i, j
    weighted_n_samples = 0.0

    j = 0
    for i in range(n_samples):
        # Only work with positively weighted samples
        # print(sample_weight)

        if sample_weight.shape[0] == 0 or sample_weight[i] != 0.0:
            samples[j] = i
            j += 1

        if sample_weight.shape[0] != 0:
            weighted_n_samples += sample_weight[i]
        else:
            weighted_n_samples += 1.0

    # print("weighted_n_samples: ", weighted_n_samples)
    # Number of samples is number of positively weighted samples
    splitter.n_samples = j
    splitter.weighted_n_samples = weighted_n_samples

    n_features = X.shape[1]
    # cdef SIZE_t* features = safe_realloc(&self.features, n_features)
    features = np.empty(n_features, dtype=NP_SIZE_t)

    for i in range(n_features):
        features[i] = i

    splitter.features = features

    splitter.n_features = n_features

    # TODO: get correct dtype
    # safe_realloc(&self.feature_values, n_samples)
    splitter.feature_values = np.empty(n_samples, dtype=NP_DOUBLE_t)
    # safe_realloc(&self.constant_features, n_features)
    splitter.constant_features = np.empty(n_features, dtype=NP_SIZE_t)

    # print("coucou")
    # print(y)
    # print(splitter)
    # print(splitter.y)

    splitter.y = y

    splitter.sample_weight = sample_weight


# The samples vector `samples` is maintained by the Splitter object such
# that the samples contained in a node are contiguous. With this setting,
# `node_split` reorganizes the node samples `samples[start:end]` in two
# subsets `samples[start:pos]` and `samples[pos:end]`.

# The 1-d  `features` array of size n_features contains the features
# indices and allows fast sampling without replacement of features.

# The 1-d `constant_features` array of size n_features holds in
# `constant_features[:n_constant_features]` the feature ids with
# constant values for all the samples that reached a specific node.
# The value `n_constant_features` is given by the parent node to its
# child nodes.  The content of the range `[n_constant_features:]` is left
# undefined, but preallocated for performance reasons
# This allows optimization with depth-based tree building.


@njit
def splitter_node_reset(splitter, start, end):
    # cdef int node_reset(self, SIZE_t start, SIZE_t end,
    #                     double* weighted_n_node_samples) nogil except -1:
    #     """Reset splitter on node samples[start:end].
    #
    #     Returns -1 in case of failure to allocate memory (and raise MemoryError)
    #     or 0 otherwise.
    #
    #     Parameters
    #     ----------
    #     start : SIZE_t
    #         The index of the first sample to consider
    #     end : SIZE_t
    #         The index of the last sample to consider
    #     weighted_n_node_samples : ndarray, dtype=double pointer
    #         The total weight of those samples
    #     """
    #
    #     self.start = start
    #     self.end = end
    #
    #     self.criterion.init(self.y,
    #                         self.sample_weight,
    #                         self.weighted_n_samples,
    #                         self.samples,
    #                         start,
    #                         end)
    #
    #     weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
    #     return 0
    splitter.start = start
    splitter.end = end

    # TODO: faut appeler le bon...

    # print(splitter)

    classification_criterion_init(
        splitter.criterion,
        splitter.y,
        splitter.sample_weight,
        splitter.weighted_n_samples,
        splitter.samples,
        start,
        end,
    )

    # weighted_n_node_samples[0] = splitter.criterion.weighted_n_node_samples
    return splitter.criterion.weighted_n_node_samples

#
# @njit
# def splitter_node_value(splitter, values, node_id):
#     # cdef void node_value(self, double* dest) nogil:
#     #     """Copy the value of node samples[start:end] into dest."""
#     #
#     #     self.criterion.node_value(dest)
#     classification_criterion_node_value(splitter.criterion, values, node_id)


@njit
def splitter_node_impurity(splitter):
    # cdef double node_impurity(self) nogil:
    #     """Return the impurity of the current node."""
    #
    #     return self.criterion.node_impurity()
    # TODO: criterion_node_impurity(splitter.criterion, splitter)

    return splitter.criterion.node_impurity()


# cdef class BaseDenseSplitter(Splitter):
#     cdef const DTYPE_t[:, :] X
#
#     cdef SIZE_t n_total_samples


spec_base_dense_splitter = spec_splitter + [
    ("X", DTYPE_t[:, :]),
    ("n_total_samples", SIZE_t),
]


@njit
def base_dense_splitter_init(splitter, X, y, sample_weight):
    #     cdef int init(self,
    #                   object X,
    #                   const DOUBLE_t[:, ::1] y,
    #                   DOUBLE_t* sample_weight) except -1:
    #         """Initialize the splitter
    #
    #         Returns -1 in case of failure to allocate memory (and raise MemoryError)
    #         or 0 otherwise.
    #         """
    #
    #         # Call parent init
    #         Splitter.init(self, X, y, sample_weight)
    #
    #         self.X = X
    #         return 0
    splitter_init(splitter, X, y, sample_weight)
    splitter.X = X


spec_best_splitter = spec_base_dense_splitter


@jitclass(spec_best_splitter)
class BestSplitter(object):
    def __init__(
        self, criterion, max_features, min_samples_leaf, min_weight_leaf, random_state
    ):
        #     def __cinit__(self, Criterion criterion, SIZE_t max_features,
        #                   SIZE_t min_samples_leaf, double min_weight_leaf,
        #                   object random_state):
        #         """
        #         Parameters
        #         ----------
        #         criterion : Criterion
        #             The criterion to measure the quality of a split.
        #
        #         max_features : SIZE_t
        #             The maximal number of randomly selected features which can be
        #             considered for a split.
        #
        #         min_samples_leaf : SIZE_t
        #             The minimal number of samples each leaf can have, where splits
        #             which would result in having less samples in a leaf are not
        #             considered.
        #
        #         min_weight_leaf : double
        #             The minimal weight each leaf can have, where the weight is the sum
        #             of the weights of each sample in it.
        #
        #         random_state : object
        #             The user inputted random state to be used for pseudo-randomness
        #         """
        #
        #         self.criterion = criterion
        #
        #         self.samples = NULL
        #         self.n_samples = 0
        #         self.features = NULL
        #         self.n_features = 0
        #         self.feature_values = NULL
        #
        #         self.sample_weight = NULL
        #
        #         self.max_features = max_features
        #         self.min_samples_leaf = min_samples_leaf
        #         self.min_weight_leaf = min_weight_leaf
        #         self.random_state = random_state
        self.criterion = criterion
        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

        # self.y = np.empty((0, 1), dtype=NP_DOUBLE_t)


@njit
def best_splitter_init(splitter, X, y, sample_weight):
    base_dense_splitter_init(splitter, X, y, sample_weight)


@njit
def best_splitter_node_split(splitter, impurity, n_constant_features, idx_samples_sort):

    # cdef SIZE_t* samples = self.samples
    # cdef SIZE_t start = self.start
    # cdef SIZE_t end = self.end
    samples = splitter.samples
    start = splitter.start
    end = splitter.end

    features = splitter.features
    constant_features = splitter.constant_features
    n_features = splitter.n_features

    Xf = splitter.feature_values

    max_features = splitter.max_features
    min_samples_leaf = splitter.min_samples_leaf
    min_weight_leaf = splitter.min_weight_leaf

    # TODO: random_state me fait chier pour l'instant
    # random_state = splitter.rand_r_state
    random_state = 42

    # cdef SplitRecord best, current
    best = SplitRecord()
    current = SplitRecord()

    current_proxy_improvement = -INFINITY
    best_proxy_improvement = -INFINITY

    f_i = n_features
    # cdef SIZE_t f_j
    # cdef SIZE_t p
    # cdef SIZE_t feature_idx_offset
    # cdef SIZE_t feature_offset
    # cdef SIZE_t i
    # cdef SIZE_t j

    # cdef SIZE_t n_visited_features = 0
    n_visited_features = 0
    # Number of features discovered to be constant during the split search
    # cdef SIZE_t n_found_constants = 0
    n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    # cdef SIZE_t n_drawn_constants = 0
    n_drawn_constants = 0
    # cdef SIZE_t n_known_constants = n_constant_features[0]
    n_known_constants = n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    # cdef SIZE_t n_total_constants = n_known_constants
    n_total_constants = n_known_constants
    # cdef DTYPE_t current_feature_value
    # cdef SIZE_t partition_end

    _init_split(best, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (
        f_i > n_total_constants
        and  # Stop early if remaining features
        # are constant
        (
            n_visited_features < max_features
            or
            # At least one drawn features must be non constant
            n_visited_features <= n_found_constants + n_drawn_constants
        )
    ):

        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        # f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
        #                random_state)
        # TODO: numba only supports the first two arguments... no random_state :(
        f_j = np.random.randint(n_drawn_constants, f_i - n_found_constants)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = (
                features[f_j],
                features[n_drawn_constants],
            )

            n_drawn_constants += 1

        else:
            # f_j in the interval [n_known_constants, f_i - n_found_constants[
            f_j += n_found_constants
            # f_j in the interval [n_total_constants, f_i[
            current.feature = features[f_j]

            # Sort samples along that feature; by
            # copying the values into an array and
            # sorting the array in a manner which utilizes the cache more
            # effectively.

            # for i in range(start, end):
            #     Xf[i] = splitter.X[samples[i], current.feature]

            Xf[start:end] = splitter.X[samples[start:end], current.feature]

            # The index of samples sorted according to the current feature
            # idx_samples = idx_samples_sort[start:end, current.feature]
            # Xf[start:end] = splitter.X[idx_samples, current.feature]
            # samples[start:end] = idx_samples
            # [start:end]
            # print(Xf[start:end])

            sort(Xf + start, samples + start, end - start)
            # print("start: ", start)

            # TODO: c'est le tri qui prend beaucoup de temps
            # sort(Xf[start:], samples[start:], end - start)

            if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                features[f_j], features[n_total_constants] = (
                    features[n_total_constants],
                    features[f_j],
                )

                n_found_constants += 1
                n_total_constants += 1

            else:
                f_i -= 1
                features[f_i], features[f_j] = features[f_j], features[f_i]

                # Evaluate all splits
                # splitter.criterion.reset()
                criterion_reset(splitter.criterion)

                p = start

                while p < end:
                    while p + 1 < end and Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD:
                        p += 1

                    # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                    #                    X[samples[p], current.feature])
                    p += 1
                    # (p >= end) or (X[samples[p], current.feature] >
                    #                X[samples[p - 1], current.feature])

                    if p < end:
                        current.pos = p

                        # Reject if min_samples_leaf is not guaranteed
                        if ((current.pos - start) < min_samples_leaf) or (
                            (end - current.pos) < min_samples_leaf
                        ):
                            continue

                        # splitter.criterion.update(current.pos)
                        criterion_update(splitter.criterion, current.pos)

                        # Reject if min_weight_leaf is not satisfied
                        if (splitter.criterion.weighted_n_left < min_weight_leaf) or (
                            splitter.criterion.weighted_n_right < min_weight_leaf
                        ):
                            continue

                        # current_proxy_improvement =
                        # splitter.criterion.proxy_impurity_improvement()

                        # current_proxy_improvement = criterion_proxy_impurity_improvement(
                        #     splitter.criterion
                        # )

                        criterion = splitter.criterion

                        current_proxy_improvement = criterion_proxy_impurity_improvement(
                            criterion.n_outputs,
                            criterion.n_classes,
                            criterion.sum_left,
                            criterion.sum_right,
                            criterion.weighted_n_left,
                            criterion.weighted_n_right,
                        )

                        # print("best_proxy_improvement: ", best_proxy_improvement)
                        # print("current_proxy_improvement: ", current_proxy_improvement)

                        if current_proxy_improvement > best_proxy_improvement:
                            best_proxy_improvement = current_proxy_improvement
                            # sum of halves is used to avoid infinite value
                            current.threshold = Xf[p - 1] / 2.0 + Xf[p] / 2.0

                            if (
                                (current.threshold == Xf[p])
                                or (current.threshold == INFINITY)
                                or (current.threshold == -INFINITY)
                            ):
                                current.threshold = Xf[p - 1]

                            # TODO: warning this might not do a copy here !
                            # best = current  # copy
                            split_record_copy(best, current)

    # Reorganize into samples[start:best.pos] + samples[best.pos:end]
    if best.pos < end:
        partition_end = end
        p = start

        while p < partition_end:
            if splitter.X[samples[p], best.feature] <= best.threshold:
                p += 1

            else:
                partition_end -= 1

                samples[p], samples[partition_end] = samples[partition_end], samples[p]

        # splitter.criterion.reset()
        criterion_reset(splitter.criterion)

        # splitter.criterion.update(best.pos)
        criterion_update(splitter.criterion, best.pos)

        # splitter.criterion.children_impurity( & best.impurity_left,
        #                                 &best.impurity_right)

        # impurity_left, impurity_right = gini_children_impurity(splitter.criterion)

        criterion = splitter.criterion

        impurity_left, impurity_right = gini_children_impurity(
            criterion.n_outputs,
            criterion.n_classes,
            criterion.sum_left,
            criterion.sum_right,
            criterion.weighted_n_left,
            criterion.weighted_n_right,
        )

        best.impurity_left = impurity_left
        best.impurity_right = impurity_right

        # best.improvement = splitter.criterion.impurity_improvement(
        #     impurity, best.impurity_left, best.impurity_right)

        # print("best.impurity_left, best.impurity_right")
        # print(best.impurity_left, best.impurity_right)

        # best.improvement = criterion_impurity_improvement(
        #     splitter.criterion, impurity, best.impurity_left, best.impurity_right
        # )

        best.improvement = criterion_impurity_improvement(
            criterion.weighted_n_samples,
            criterion.weighted_n_node_samples,
            criterion.weighted_n_left,
            criterion.weighted_n_right,
            impurity,
            best.impurity_left,
            best.impurity_right,
        )




    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes

    #   memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)
    features[:n_known_constants] = constant_features[:n_known_constants]

    # Copy newly found constant features
    # memcpy(constant_features + n_known_constants,
    #        features + n_known_constants,
    #        sizeof(SIZE_t) * n_found_constants)

    constant_features[
        n_known_constants : (n_known_constants + n_found_constants)
    ] = features[n_known_constants : (n_known_constants + n_found_constants)]

    # Return values
    # split[0] = best
    # n_constant_features[0] = n_total_constants

    return best, n_total_constants


@njit
def sort(Xf, samples, n):
    # # Sort n-element arrays pointed to by Xf and samples, simultaneously,
    # # by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
    # cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    #     if n == 0:
    #       return
    #     cdef int maxd = 2 * <int>log(n)
    #     introsort(Xf, samples, n, maxd)

    if n == 0:
        return

    maxd = 2 * int(ln(n) / ln(2))
    introsort(Xf, samples, n, maxd)


@njit
def swap(Xf, samples, i, j):
    # cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
    #         SIZE_t i, SIZE_t j) nogil:
    #     # Helper for sort
    #     Xf[i], Xf[j] = Xf[j], Xf[i]
    #     samples[i], samples[j] = samples[j], samples[i]
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


@njit
def median3(Xf, n):
    # cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    #     # Median of three pivot selection, after Bentley and McIlroy (1993).
    #     # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    #     cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    #     if a < b:
    #         if b < c:
    #             return b
    #         elif a < c:
    #             return c
    #         else:
    #             return a
    #     elif b < c:
    #         if a < c:
    #             return a
    #         else:
    #             return c
    #     else:
    #         return b
    a = Xf[0]
    b = Xf[n // 2]
    c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# # Introsort with median of 3 pivot selection and 3-way partition function
# # (robust to repeated elements, e.g. lots of zero features).
@njit
def introsort(Xf, samples, n, maxd):
    # cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
    #                     SIZE_t n, int maxd) nogil:
    #     cdef DTYPE_t pivot
    #     cdef SIZE_t i, l, r
    #
    #     while n > 1:
    #         if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
    #             heapsort(Xf, samples, n)
    #             return
    #         maxd -= 1
    #
    #         pivot = median3(Xf, n)
    #
    #         # Three-way partition.
    #         i = l = 0
    #         r = n
    #         while i < r:
    #             if Xf[i] < pivot:
    #                 swap(Xf, samples, i, l)
    #                 i += 1
    #                 l += 1
    #             elif Xf[i] > pivot:
    #                 r -= 1
    #                 swap(Xf, samples, i, r)
    #             else:
    #                 i += 1
    #
    #         introsort(Xf, samples, l, maxd)
    #         Xf += r
    #         samples += r
    #         n -= r

    # offset = 0
    while n > 1:
        if maxd <= 0:  # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = 0
        l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)

        # TODO: Original implementation uses pointer arithmetic. Use indexing instead ?
        # Xf += r
        Xf = Xf[r:]
        # samples += r
        samples = samples[r:]
        # offset += r
        n -= r


#
@njit
def sift_down(Xf, samples, start, end):
    # cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
    #                            SIZE_t start, SIZE_t end) nogil:
    #     # Restore heap order in Xf[start:end] by moving the max element to start.
    #     cdef SIZE_t child, maxind, root
    #
    #     root = start
    #     while True:
    #         child = root * 2 + 1
    #
    #         # find max of root, left child, right child
    #         maxind = root
    #         if child < end and Xf[maxind] < Xf[child]:
    #             maxind = child
    #         if child + 1 < end and Xf[maxind] < Xf[child + 1]:
    #             maxind = child + 1
    #
    #         if maxind == root:
    #             break
    #         else:
    #             swap(Xf, samples, root, maxind)
    #             root = maxind

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


@njit
def heapsort(Xf, samples, n):
    # cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    #     cdef SIZE_t start, end
    #
    #     # heapify
    #     start = (n - 2) / 2
    #     end = n
    #     while True:
    #         sift_down(Xf, samples, start, end)
    #         if start == 0:
    #             break
    #         start -= 1
    #
    #     # sort by shrinking the heap, putting the max element immediately after it
    #     end = n - 1
    #     while end > 0:
    #         swap(Xf, samples, 0, end)
    #         sift_down(Xf, samples, 0, end)
    #         end = end - 1

    # heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


# @njit
# def main():
#     split = SplitRecord(
#         0,          # feature
#         0,          # pos
#         0.0,        # threshold
#         -INFINITY,  # improvement
#         np.inf,     # impurity_left
#         np.inf      # impurity_right
#     )
#     _init_split(split, 123)
#
#     splitter = Splitter(Criterion())
#
#     print(split)

# x = np.array([1, 3, np.nan, 2], dtype=NP_SIZE_t)


# @njit
# def main():
#
#     n = 10
#
#     Xf = np.random.randn(10)
#     print(Xf)
#     samples = np.arange(0, 10)
#     print(samples)
#
#     sort(Xf, samples, n)
#
#     print(Xf)
#     print(samples)
#
#     # max_features = 2
#     # min_samples_leaf = 2
#     # min_weight_leaf = 1.0
#     # random_state = 123
#     # sample_weight = np.ones(X.shape[0], dtype=DOUBLE_t)
#     # sample_weight[123] = 0.0
#     # # sample_weight = np.empty(0, dtype=DOUBLE_t)
#     #
#     # n_outputs = 1
#     # n_classes = np.array([3], dtype=NP_SIZE_t)
#     # criterion = Gini(n_outputs, n_classes)
#     #
#     # # splitter = Splitter(criterion, max_features, min_samples_leaf,
#     # #                     min_weight_leaf, random_state)
#     #
#     # splitter_init(splitter, X, y, sample_weight)
#     #
#     # # x = NULL
#     # # print(isnan(x))
#
#
# main()
