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
    UINT8_t,
    NP_UINT8_t,
    NP_BOOL_t,
    DTYPE_t,
    NP_DTYPE_t,
    get_numba_type,
)

from ._random import RAND_R_MAX, rand_int


from ._criterion import (
    Gini,
    classification_criterion_init,
    gini_children_impurity,
    gini_node_impurity,
    criterion_proxy_impurity_improvement,
)

# TODO: correct ?
NULL = np.nan


FEATURE_THRESHOLD = DTYPE_t(1e-7)

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
EXTRACT_NNZ_SWITCH = DTYPE_t(0.1)


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


@njit
def _init_split(split_record, start_pos):
    split_record.impurity_left = INFINITY
    split_record.impurity_right = INFINITY
    split_record.pos = start_pos
    split_record.feature = 0
    split_record.threshold = 0.0
    split_record.improvement = -INFINITY


spec_splitter = [
    ("criterion", get_numba_type(Gini)),
    ("max_features", SIZE_t),
    ("min_samples_leaf", SIZE_t),
    ("min_weight_leaf", SIZE_t),
    ("random_state", UINT32_t),
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

    # np.random.RandomState(seed)

    # self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)

    # TODO: Ca va etre le seed qui va evoluer le long des samplings
    splitter.rand_r_state = np.random.randint(0, RAND_R_MAX)

    n_samples = X.shape[0]

    if sample_weight is None:
        sample_weight = np.empty(0, dtype=NP_DOUBLE_t)

    # Create a new array which will be used to store nonzero
    # samples from the feature of interest
    # safe_realloc(&self.samples, n_samples)
    samples = np.empty(n_samples, dtype=NP_SIZE_t)
    splitter.samples = samples
    weighted_n_samples = 0.0

    j = 0
    has_weights = sample_weight.size > 0
    if has_weights:
        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight[i] > 0.0:
                samples[j] = i
                j += 1
            weighted_n_samples += sample_weight[i]
    else:
        for i in range(n_samples):
            samples[j] = i
            j += 1
            weighted_n_samples += 1.0

    # print("weighted_n_samples: ", weighted_n_samples)
    # Number of samples is number of positively weighted samples
    splitter.n_samples = j
    splitter.weighted_n_samples = weighted_n_samples

    n_features = X.shape[1]
    # cdef SIZE_t* features = safe_realloc(&self.features, n_features)

    # features = np.empty(n_features, dtype=NP_SIZE_t)
    features = np.arange(n_features, dtype=NP_SIZE_t)
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
    splitter.start = start
    splitter.end = end

    # TODO: faut appeler le bon...
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


@njit
def splitter_node_impurity(splitter):
    return splitter.criterion.node_impurity()


spec_base_dense_splitter = spec_splitter + [
    ("X", DTYPE_t[:, :]),
    ("n_total_samples", SIZE_t),
    ("node_mask", UINT8_t[::1])
]


@njit
def base_dense_splitter_init(splitter, X, y, sample_weight):
    splitter_init(splitter, X, y, sample_weight)
    splitter.X = X
    n_total_samples = X.shape[0]
    splitter.n_total_samples = n_total_samples
    splitter.node_mask = np.zeros(n_total_samples, dtype=NP_UINT8_t)


spec_best_splitter = spec_base_dense_splitter


@jitclass(spec_best_splitter)
class BestSplitter(object):
    def __init__(
        self, criterion, max_features, min_samples_leaf, min_weight_leaf, random_state
    ):
        self.criterion = criterion
        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

        # print("random_state: ", random_state)


@njit
def best_splitter_init(splitter, X, y, sample_weight):
    base_dense_splitter_init(splitter, X, y, sample_weight)


@njit
def best_splitter_node_split(splitter, impurity, n_constant_features, X_idx_sort):
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
    # random_state = splitter.rand_r_state

    # cdef SplitRecord best, current
    best = SplitRecord()
    current = SplitRecord()

    current_proxy_improvement = -INFINITY
    best_proxy_improvement = -INFINITY

    f_i = n_features
    n_visited_features = 0
    # Number of features discovered to be constant during the split search
    n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    n_drawn_constants = 0
    n_known_constants = n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    n_total_constants = n_known_constants

    _init_split(best, end)

    node_mask = splitter.node_mask

    for p in range(start, end):
        node_mask[samples[p]] = 1

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

        # f_j = np.random.randint(n_drawn_constants, f_i - n_found_constants)

        f_j, splitter.rand_r_state = rand_int(
            n_drawn_constants, f_i - n_found_constants, splitter.rand_r_state
        )

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

            # TODO: remove sorting

            # Imaginons que dans le noeud on a comme samples
            # [5 8 3 0 6 2 1 7 4 9] = samples
            # start=3 et end=7
            # [0 6 2 1] = samples[start:end]

            # Et que l'ordre qui trie selon la feature c'est
            # [8 5 1 2 7 6 3 2 4 0] = X_idx_sort[:, current.feature]
            # On veut mettre dans Xf et dans samples les trucs dans le bon ordre

            # 0 -> 8 , 6 -> 3, 2 -> 1, 1 -> 5

            # [0 1 2 3 4 5 6 7 8 9]
            # [8 5 1 2 7 6 3 2 4 0]

            # [1 2 6 0] c'est le truc qu'on veut recuperer du coup

            # On veut recuperer  -> c'est ce qui trie les samples du noeud
            # dans le bon ordre

            # if self.presort == 1:
            #     p = start
            #     feature_idx_offset = self.X_idx_sorted_stride * current.feature
            #
            #     for i in range(self.n_total_samples):
            #         j = X_idx_sorted[i + feature_idx_offset]
            #         if sample_mask[j] == 1:
            #             samples[p] = j
            #             Xf[p] = self.X[j, current.feature]
            #             p += 1

            # idx_feature_sort contains the full sample indices that sort feature
            idx_feature_sort = X_idx_sort[:, current.feature]
            p = start
            for i in range(splitter.n_total_samples):
                j = idx_feature_sort[i]
                if node_mask[j] == 1:
                    samples[p] = j
                    Xf[p] = splitter.X[j, current.feature]
                    p += 1

                if p == end:
                    break
                    # TODO: if p == end on arrete ?

            # samples[start:end] contains the set of i such that x_i is in the node

            # Xf[start:end] = splitter.X[samples[start:end], current.feature]

            # print('------------')
            # print("current.feature: ", current.feature, ", start: ", start, " end: ",
            #       end)
            # print("== before sort")
            # print("Xf: ", Xf)
            # print("Xf[start:end]: ", Xf[start:end])
            # print("samples: ", samples)
            # print("samples[start:end]: ", samples[start:end])

            # sort(Xf[start:], samples[start:], end - start)

            # Xf[start:end] = splitter.X[idx_feature_sort, current.feature]
            # samples[start:end] = samples[idx_feature_sort]

            # print("== after sort")
            # print("Xf: ", Xf)
            # print("Xf[start:end]: ", Xf[start:end])
            # print("samples: ", samples)
            # print("samples[start:end]: ", samples[start:end])

            # print("start: ", start)

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

        criterion_reset(splitter.criterion)
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

    # Put back the mask in this original form
    # TODO: useless only if we remove the creation of the node_mask ecah time
    for p in range(start, end):
        node_mask[samples[p]] = 0

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
    while n > 1:
        if maxd <= 0:  # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
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
