
import numpy as np


from ._utils import (
    njit,
    jitclass,
    int32,
    float32,
    uint8,
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


@jitclass([
    ('gain', float32),
    ('feature_idx', uint32),
    ('bin_idx', uint8),
    ('n_samples_left', float32),   # float32 whenever samples_weights is used
    ('n_samples_right', float32),  # float32 whenever samples_weights is used
    # TODO: attention c'est que pour la clasiication
    ('y_count_left', float32[::1]),
    ('y_count_right', float32[::1])
])
class SplitInfo:
    """Pure data class to store information about a potential split.

    Parameters
    ----------
    gain : float32
        The gain of the split
    feature_idx : int
        The index of the feature to be split
    bin_idx : int
        The index of the bin on which the split is made
    n_samples_left : int
        The number of samples in the left child
    n_samples_right : int
        The number of samples in the right child
    y_count_left : array of int
        The number of samples for each label class in the left child
    y_count_right : array of int
        The number of samples for each label class in the right child
    """
    def __init__(self, gain=-1., feature_idx=0, bin_idx=0,
                 n_samples_left=0., n_samples_right=0., y_count_left=None,
                 y_count_right=None):
        self.gain = gain
        self.feature_idx = feature_idx
        self.bin_idx = bin_idx
        self.n_samples_left = n_samples_left
        self.n_samples_right = n_samples_right
        if y_count_left is not None:
            self.n_samples_left = n_samples_left
        if y_count_right is not None:
            self.n_samples_right = n_samples_right



spec_splitting_context = [
    ('n_features', uint32),
    ('X_binned', uint8[::1, :]),
    ('max_bins', uint32),
    ('n_bins_per_feature', uint32[::1]),
    ('min_samples_leaf', uint32),
    ('min_gain_to_split', float32),
    ('partition', uint32[::1]),
    ('left_indices_buffer', uint32[::1]),
    ('right_indices_buffer', uint32[::1]),
]


@jitclass(spec_splitting_context)
class SplittingContext:
    """A data class containing useful context information for splitting. This is
    instanciated by ....

    Parameters
    ----------
    X_binned : array of int
        The binned input samples. Must be Fortran-aligned.
    max_bins : int, optional(default=256)
        The maximum number of bins. Used to define the shape of the
        histograms.
    n_bins_per_feature : array-like of int
        The actual number of bins needed for each feature, which is lower or
        equal to max_bins.
    min_samples_leaf : int
        The minimum number of samples per leaf.
    min_gain_to_split : float, optional(default=0.)
        The minimum gain needed to split a node. Splits with lower gain will
        be ignored.
    """
    def __init__(self, X_binned, max_bins, n_bins_per_feature,
                 gradients, hessians, l2_regularization,
                 min_hessian_to_split=1e-3, min_samples_leaf=20,
                 min_gain_to_split=0.):

        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        # Note: all histograms will have <max_bins> bins, but some of the
        # last bins may be unused if n_bins_per_feature[f] < max_bins
        self.max_bins = max_bins
        self.n_bins_per_feature = n_bins_per_feature
        self.gradients = gradients
        self.hessians = hessians
        # for root node, gradients and hessians are already ordered
        self.ordered_gradients = gradients.copy()
        self.ordered_hessians = hessians.copy()
        self.sum_gradients = self.gradients.sum()
        self.sum_hessians = self.hessians.sum()
        self.constant_hessian = hessians.shape[0] == 1
        self.l2_regularization = l2_regularization
        self.min_hessian_to_split = min_hessian_to_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split
        if self.constant_hessian:
            self.constant_hessian_value = self.hessians[0]  # 1 scalar
        else:
            self.constant_hessian_value = float32(1.)  # won't be used anyway

        # The partition array maps each sample index into the leaves of the
        # tree (a leaf in this context is a node that isn't splitted yet, not
        # necessarily a 'finalized' leaf). Initially, the root contains all
        # the indices, e.g.:
        # partition = [abcdefghijkl]
        # After a call to split_indices, it may look e.g. like this:
        # partition = [cef|abdghijkl]
        # we have 2 leaves, the left one is at position 0 and the second one at
        # position 3. The order of the samples is irrelevant.
        self.partition = np.arange(0, X_binned.shape[0], 1, np.uint32)
        # buffers used in split_indices to support parallel splitting.
        self.left_indices_buffer = np.empty_like(self.partition)
        self.right_indices_buffer = np.empty_like(self.partition)


@njit(locals={'gradient_left': float32, 'hessian_left': float32,
              'n_samples_left': uint32},
      fastmath=True)
def _find_best_bin_to_split_helper(context, feature_idx, histogram, n_samples):
    """Find best bin to split on, and return the corresponding SplitInfo.

    Splits that do not satisfy the splitting constraints (min_gain_to_split,
    etc.) are discarded here. If no split can satisfy the constraints, a
    SplitInfo with a gain of -1 is returned. If for a given node the best
    SplitInfo has a gain of -1, it is finalized into a leaf.
    """
    # Allocate the structure for the best split information. It can be
    # returned as such (with a negative gain) if the min_hessian_to_split
    # condition is not satisfied. Such invalid splits are later discarded by
    # the TreeGrower.
    best_split = SplitInfo(-1., 0, 0, 0., 0., None, None)
    gradient_left, hessian_left = 0., 0.
    n_samples_left = 0

    for bin_idx in range(context.n_bins_per_feature[feature_idx]):
        n_samples_left += histogram[bin_idx]['count']
        n_samples_right = n_samples - n_samples_left

        if context.constant_hessian:
            hessian_left += (histogram[bin_idx]['count']
                             * context.constant_hessian_value)
        else:
            hessian_left += histogram[bin_idx]['sum_hessians']
        hessian_right = context.sum_hessians - hessian_left

        gradient_left += histogram[bin_idx]['sum_gradients']
        gradient_right = context.sum_gradients - gradient_left

        if n_samples_left < context.min_samples_leaf:
            continue
        if n_samples_right < context.min_samples_leaf:
            # won't get any better
            break

        if hessian_left < context.min_hessian_to_split:
            continue
        if hessian_right < context.min_hessian_to_split:
            # won't get any better (hessians are > 0 since loss is convex)
            break

        gain = _split_gain(gradient_left, hessian_left,
                           gradient_right, hessian_right,
                           context.sum_gradients, context.sum_hessians,
                           context.l2_regularization)

        if gain > best_split.gain and gain > context.min_gain_to_split:
            best_split.gain = gain
            best_split.feature_idx = feature_idx
            best_split.bin_idx = bin_idx
            best_split.gradient_left = gradient_left
            best_split.hessian_left = hessian_left
            best_split.n_samples_left = n_samples_left
            best_split.gradient_right = gradient_right
            best_split.hessian_right = hessian_right
            best_split.n_samples_right = n_samples_right

    return best_split, histogram



@njit(fastmath=False)
def _split_gain(gradient_left, hessian_left, gradient_right, hessian_right,
                sum_gradients, sum_hessians, l2_regularization):
    """Loss reduction

    Compute the reduction in loss after taking a split, compared to keeping
    the node a leaf of the tree.

    See Equation 7 of:
    XGBoost: A Scalable Tree Boosting System, T. Chen, C. Guestrin, 2016
    https://arxiv.org/abs/1603.02754
    """
    def negative_loss(gradient, hessian):
        return (gradient ** 2) / (hessian + l2_regularization)

    gain = negative_loss(gradient_left, hessian_left)
    gain += negative_loss(gradient_right, hessian_right)
    gain -= negative_loss(sum_gradients, sum_hessians)
    return gain
