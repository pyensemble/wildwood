# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause


import numpy as np
from ._utils import (
    SIZE_t,
    NP_SIZE_t,
    UINT32_t,
    NP_UINT32_t,
    DOUBLE_t,
    NP_DOUBLE_t,
    jitclass,
    njit,
)


@njit
def criterion_proxy_impurity_improvement(
    n_outputs, n_classes, sum_left, sum_right, weighted_n_left, weighted_n_right
):
    #     cdef double proxy_impurity_improvement(self) nogil:
    #         """Compute a proxy of the impurity reduction.
    #
    #         This method is used to speed up the search for the best split.
    #         It is a proxy quantity such that the split that maximizes this value
    #         also maximizes the impurity improvement. It neglects all constant terms
    #         of the impurity decrease for a given split.
    #
    #         The absolute impurity improvement is only computed by the
    #         impurity_improvement method once the best split has been found.
    #         """

    impurity_left, impurity_right = gini_children_impurity(
        n_outputs, n_classes, sum_left, sum_right, weighted_n_left, weighted_n_right
    )

    # impurity_left, impurity_right = gini_children_impurity(criterion)
    return -weighted_n_right * impurity_right - weighted_n_left * impurity_left


@njit
def criterion_impurity_improvement(
    weighted_n_samples,
    weighted_n_node_samples,
    weighted_n_left,
    weighted_n_right,
    impurity_parent,
    impurity_left,
    impurity_right,
):
    #         """Compute the improvement in impurity.
    #
    #         This method computes the improvement in impurity when a split occurs.
    #         The weighted impurity improvement equation is the following:
    #
    #             N_t / N * (impurity - N_t_R / N_t * right_impurity
    #                                 - N_t_L / N_t * left_impurity)
    #
    #         where N is the total number of samples, N_t is the number of samples
    #         at the current node, N_t_L is the number of samples in the left child,
    #         and N_t_R is the number of samples in the right child,
    #
    #         Parameters
    #         ----------
    #         impurity_parent : double
    #             The initial impurity of the parent node before the split
    #
    #         impurity_left : double
    #             The impurity of the left child
    #
    #         impurity_right : double
    #             The impurity of the right child
    #
    #         Return
    #         ------
    #         double : improvement in impurity after the split occurs
    #         """

    return (weighted_n_node_samples / weighted_n_samples) * (
        impurity_parent
        - (weighted_n_right / weighted_n_node_samples * impurity_right)
        - (weighted_n_left / weighted_n_node_samples * impurity_left)
    )


spec_criterion = [
    ("y", DOUBLE_t[:, ::1]),
    ("sample_weight", DOUBLE_t[::1]),
    ("samples", SIZE_t[::1]),  # A numpy array holding the sample indices
    ("start", SIZE_t),
    ("pos", SIZE_t),
    ("end", SIZE_t),
    ("n_samples", SIZE_t),  # It's X.shape[0]
    ("n_node_samples", SIZE_t),  # It's X.shape[0]
    ("n_outputs", SIZE_t),
    ("weighted_n_samples", DOUBLE_t),
    ("weighted_n_node_samples", DOUBLE_t),
    ("weighted_n_left", DOUBLE_t),
    ("weighted_n_right", DOUBLE_t),
    ("max_n_classes", SIZE_t),
    ("sum_total", DOUBLE_t[:, ::1]),
    ("sum_left", DOUBLE_t[:, ::1]),
    ("sum_right", DOUBLE_t[:, ::1]),
]

spec_classification_criterion = spec_criterion + [("n_classes", SIZE_t[::1])]


@jitclass(spec_classification_criterion)
class ClassificationCriterion(object):
    def __init__(self, n_outputs, n_classes):
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.n_classes = np.empty(n_outputs, dtype=NP_SIZE_t)

        self.n_classes[:] = n_classes

        self.max_n_classes = np.max(self.n_classes)

        shape = (self.n_outputs, self.max_n_classes)
        self.sum_total = np.empty(shape, dtype=NP_DOUBLE_t)
        self.sum_left = np.empty(shape, dtype=NP_DOUBLE_t)
        self.sum_right = np.empty(shape, dtype=NP_DOUBLE_t)


@njit
def classification_criterion_init(
    criterion, y, sample_weight, weighted_n_samples, samples, start, end
):

    criterion.y = y
    criterion.sample_weight = sample_weight
    criterion.samples = samples
    criterion.start = start
    criterion.end = end
    criterion.n_node_samples = end - start
    criterion.weighted_n_samples = weighted_n_samples
    criterion.weighted_n_node_samples = 0.0

    n_classes = criterion.n_classes
    sum_total = criterion.sum_total

    w = 1.0

    sum_total[:] = 0.0

    for p in range(start, end):
        i = samples[p]

        # w is originally set to be 1.0, meaning that if no sample weights
        # are given, the default weight of each sample is 1.0
        # if sample_weight != NULL:
        # TODO: faire ce test en dehors
        if sample_weight.size > 0:
            w = sample_weight[i]

        # Count weighted class frequency for each target
        for n_output in range(criterion.n_outputs):
            c = SIZE_t(criterion.y[i, n_output])
            # sum_total[n_output * criterion.sum_stride + c] += w
            sum_total[n_output, c] += w

        criterion.weighted_n_node_samples += w

    # Reset to pos=start
    # criterion.reset()
    criterion_reset(criterion)
    return 0



@njit
def criterion_reset(criterion):

    criterion.pos = criterion.start
    criterion.weighted_n_left = 0.0
    criterion.weighted_n_right = criterion.weighted_n_node_samples

    sum_total = criterion.sum_total
    sum_left = criterion.sum_left
    sum_right = criterion.sum_right

    sum_left[:, :] = 0
    sum_right[:, :] = sum_total


@njit
def criterion_reverse_reset(criterion):
    criterion.pos = criterion.end
    criterion.weighted_n_left = criterion.weighted_n_node_samples
    criterion.weighted_n_right = 0.0

    sum_total = criterion.sum_total
    sum_left = criterion.sum_left
    sum_right = criterion.sum_right

    sum_right[:, :] = 0
    sum_left[:, :] = sum_total


@njit
def criterion_update(criterion, new_pos):
    pos = criterion.pos
    end = criterion.end

    sum_left = criterion.sum_left
    sum_right = criterion.sum_right
    sum_total = criterion.sum_total

    n_classes = criterion.n_classes
    samples = criterion.samples
    sample_weight = criterion.sample_weight

    w = 1.0

    # Update statistics up to new_pos
    #
    # Given that
    #   sum_left[x] +  sum_right[x] = sum_total[x]
    # and that sum_total is known, we are going to update
    # sum_left from the direction that require the least amount
    # of computations, i.e. from pos to new_pos or from end to new_po.

    has_sample_weight = sample_weight.size > 0

    if (new_pos - pos) <= (end - new_pos):
        for p in range(pos, new_pos):
            i = samples[p]

            if has_sample_weight:
                w = sample_weight[i]

            for k in range(criterion.n_outputs):
                c = SIZE_t(criterion.y[i, k])
                sum_left[k, c] += w

            criterion.weighted_n_left += w

    else:
        criterion_reverse_reset(criterion)
        # criterion.reverse_reset()

        for p in range(end - 1, new_pos - 1, -1):
            i = samples[p]

            if has_sample_weight:
                w = sample_weight[i]

            for k in range(criterion.n_outputs):
                # label_index = k * criterion.sum_stride + SIZE_t(criterion.y[i, k])
                c = SIZE_t(criterion.y[i, k])
                sum_left[k, c] -= w

            criterion.weighted_n_left -= w

    # Update right part statistics
    criterion.weighted_n_right = (
        criterion.weighted_n_node_samples - criterion.weighted_n_left
    )

    # idx_sum_total = 0
    # idx_sum_left = 0
    # idx_sum_right = 0
    #
    # for k in range(criterion.n_outputs):
    #     # TODO : c'est pas terrible ca
    #     for c in range(n_classes[k]):
    #
    #         # sum_right[c] = sum_total[c] - sum_left[c]
    #         sum_right[idx_sum_right + c] = sum_total[idx_sum_total + c] - sum_left[
    #             idx_sum_left + c]
    #
    #     idx_sum_total += criterion.sum_stride
    #     idx_sum_left += criterion.sum_stride
    #     idx_sum_right += criterion.sum_stride

    # c'est ok ?
    sum_right[:] = sum_total - sum_left

    criterion.pos = new_pos
    return 0


@jitclass(spec_classification_criterion)
class Gini(object):
    # cdef class Gini(ClassificationCriterion):
    #     r"""Gini Index impurity criterion.
    #
    #     This handles cases where the target is a classification taking values
    #     0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    #     then let
    #
    #         count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)
    #
    #     be the proportion of class k observations in node m.
    #
    #     The Gini Index is then defined as:
    #
    #         index = \sum_{k=0}^{K-1} count_k (1 - count_k)
    #               = 1 - \sum_{k=0}^{K-1} count_k ** 2
    #     """

    def __init__(self, n_outputs, n_classes):
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.n_classes = np.empty(n_outputs, dtype=NP_SIZE_t)
        self.n_classes[:] = n_classes
        self.max_n_classes = np.max(self.n_classes)

        shape = (self.n_outputs, self.max_n_classes)
        self.sum_total = np.empty(shape, dtype=NP_DOUBLE_t)
        self.sum_left = np.empty(shape, dtype=NP_DOUBLE_t)
        self.sum_right = np.empty(shape, dtype=NP_DOUBLE_t)


@njit
def gini_node_impurity(n_outputs, n_classes, sum_total, weighted_n_node_samples):
    gini = 0.0
    # TODO: For loop since a label might have more classes than others... du coup on
    #  ajouter plein de zeros mais ptet moins bon en terme de cache

    weighted_n_node_samples_sq = weighted_n_node_samples * weighted_n_node_samples
    for k in range(n_outputs):
        sq_count = 0.0
        for c in range(n_classes[k]):
            count_k = sum_total[k, c]
            sq_count += count_k * count_k

        gini += 1.0 - sq_count / weighted_n_node_samples_sq

    return gini / n_outputs


@njit
def gini_children_impurity(
    n_outputs, n_classes, sum_left, sum_right, weighted_n_left, weighted_n_right
):
    gini_left = 0.0
    gini_right = 0.0

    weighted_n_left_sq = weighted_n_left * weighted_n_left
    weighted_n_right_sq = weighted_n_right * weighted_n_right

    for k in range(n_outputs):
        sq_count_left = 0.0
        sq_count_right = 0.0

        for c in range(n_classes[k]):
            count_k = sum_left[k, c]
            sq_count_left += count_k * count_k

            count_k = sum_right[k, c]
            sq_count_right += count_k * count_k

        gini_left += 1.0 - sq_count_left / weighted_n_left_sq
        gini_right += 1.0 - sq_count_right / weighted_n_right_sq

    return gini_left / n_outputs, gini_right / n_outputs
