"""

This is for now mostly a translation of scikit-learn's _splitter.pxd and
_splitter.pyx code into numba


"""

#
#
# Translation of the code from _splitter.pxd
#
#

import numpy as np
from numba import njit
from numba.experimental import jitclass

from _utils import int32, uint32, DOUBLE_t, NP_DOUBLE_t, SIZE_t, INT32_t, UINT32_t, \
    INFINITY, NP_SIZE_t

from _criterion import Gini

from collections import namedtuple
from numba import intp, int32, uint32, float64
from numpy import isnan

# from cmath import isnan
# from _utils import get_numba_type

# TODO: correct ?
NULL = np.nan


@njit
def _init_split(split, start_pos):
    split[0] = 0
    split.pos = start_pos
    split.threshold = 0.0
    split.improvement = -np.inf
    split.impurity_left = np.inf
    split.impurity_right = np.inf

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
    ("impurity_right", DOUBLE_t)
]


@jitclass(spec_split_record)
class SplitRecord(object):

    def __init__(self, feature, pos, threshold, improvement, impurity_left,
                 impurity_right):
        self.feature = feature
        self.pos = pos
        self.threshold = threshold
        self.improvement = improvement
        self.impurity_left = impurity_left
        self.impurity_right = impurity_right


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
    split_record.threshold = 0.
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


spec_splitter = [
    ("criterion", Gini.class_type.instance_type),
    ("max_features", SIZE_t),
    ("min_samples_leaf", SIZE_t),
    ("min_weight_leaf", SIZE_t),
    ("random_state", UINT32_t), # TODO: this is wrong
    ("rand_r_state", UINT32_t),
    ("max_features", SIZE_t),
    ("samples", SIZE_t[::1]),  # A numpy array holding the sample indices
    ("n_samples", SIZE_t),        # It's X.shape[0]
    ("weighted_n_samples", DOUBLE_t),
    ("features", SIZE_t[::1]),      # Feature indices in X
    ("constant_features", SIZE_t[::1]), # Constant feature indices
    ("n_features", SIZE_t),   # It's X.shape[0]
    ("max_features", SIZE_t),
    ("feature_values", DOUBLE_t[::1]),
    ("start", SIZE_t),
    ("end", SIZE_t),
    ("y", DOUBLE_t[::1]),
    ("sample_weight", DOUBLE_t[::1]),
]

@jitclass(spec_splitter)
class Splitter(object):

    def __init__(self, criterion, max_features, min_samples_leaf, min_weight_leaf,
                 random_state):

        self.criterion = criterion
        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

@njit
def splitter_init(splitter, X, y, sample_weight):
        # self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        # safe_realloc(&self.samples, n_samples)
        samples = np.empty(n_samples, dtype=NP_SIZE_t)

        # cdef SIZE_t i, j
        weighted_n_samples = 0.0

        j = 0
        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight.shape[0] == 0 or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight.shape[0] != 0:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        print("weighted_n_samples: ", weighted_n_samples)
        # Number of samples is number of positively weighted samples
        splitter.n_samples = j
        splitter.weighted_n_samples = weighted_n_samples

        n_features = X.shape[1]
        # cdef SIZE_t* features = safe_realloc(&self.features, n_features)
        features = np.empty(n_features, dtype=NP_SIZE_t)

        for i in range(n_features):
            features[i] = i

        splitter.n_features = n_features

        # TODO: get correct dtype
        # safe_realloc(&self.feature_values, n_samples)
        splitter.feature_values = np.empty(n_samples, dtype=DOUBLE_t)
        # safe_realloc(&self.constant_features, n_features)
        splitter.constant_features = np.empty(n_features, dtype=SIZE_t)

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
def splitter_node_reset(splitter, start, end, weighted_n_node_samples):
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
    splitter.criterion.init(splitter.y,
                            splitter.sample_weight,
                            splitter.weighted_n_samples,
                            splitter.samples,
                            start,
                            end)

    weighted_n_node_samples[0] = splitter.criterion.weighted_n_node_samples


@njit
def splitter_node_split(splitter, impurity, split, n_constant_features):
    # cdef int node_split(self, double impurity, SplitRecord* split,
    #                     SIZE_t* n_constant_features) nogil except -1:
    #     """Find the best split on node samples[start:end].
    #
    #     This is a placeholder method. The majority of computation will be done
    #     here.
    #
    #     It should return -1 upon errors.
    #     """
    #
    #     pass
    pass


@njit
def splitter_node_value(splitter, dest):
    # cdef void node_value(self, double* dest) nogil:
    #     """Copy the value of node samples[start:end] into dest."""
    #
    #     self.criterion.node_value(dest)
    splitter.criterion.node_value(dest)

@njit
def node_impurity(splitter):
    # cdef double node_impurity(self) nogil:
    #     """Return the impurity of the current node."""
    #
    #     return self.criterion.node_impurity()
    # TODO: criterion_node_impurity(splitter.criterion, splitter)
    return splitter.criterion.node_impurity()


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


from sklearn.datasets import make_circles

n_samples = 150
random_state = 42


X, y = make_circles(n_samples=n_samples, noise=0.2, random_state=random_state)

y = y.astype(NP_DOUBLE_t)

# print(y[:10])

@njit
def main():

    max_features = 2
    min_samples_leaf = 2
    min_weight_leaf = 1.0
    random_state = 123
    sample_weight = np.ones(X.shape[0], dtype=DOUBLE_t)
    sample_weight[123] = 0.0
    # sample_weight = np.empty(0, dtype=DOUBLE_t)

    n_outputs = 1
    n_classes = np.array([3], dtype=NP_SIZE_t)
    criterion = Gini(n_outputs, n_classes)

    splitter = Splitter(criterion, max_features, min_samples_leaf,
                        min_weight_leaf, random_state)

    splitter_init(splitter, X, y, sample_weight)

    # x = NULL
    # print(isnan(x))


main()
