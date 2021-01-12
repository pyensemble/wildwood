"""This module contains njitted functions to build histogram.

For a classification problem with n_classes, with n_bins bins and max_features, an
"histogram" consist of a (max_features, n_bins, n_classes) numpy array which contains
the number of features to test, the maximum number of bins and the number of classes

TODO: reprendre ce texte
"""
import numpy as np
from numba import njit


from ._utils import np_float32, nb_float32, np_uint32, nb_uint32


# HISTOGRAM_DTYPE = np.dtype([
#     ('sum_gradients', np_float32),
#     ('sum_hessians', np_float32),
#     ('count', np_uint32),
#
# ])


@njit
def compute_bin_statistics(X, y, n_classes, features, n_bins, train_indices,
                           sample_weight_train):
    """
    TODO: la strategie est bien en fait car les histogrammes sont petits comme en
    principe on sous-echantillone les features

    Parameters
    ----------
    X
    y
    n_classes
    features
    n_bins
    train_indices
    sample_weight_train

    Returns
    -------

    """
    max_features = features.size
    # The dtype is float32 since it's actually weighted sample counts
    # TODO: faudra que ces tableaux soient crees en dehors et appartiennent au
    #  context et C major pour iterer vite sur n_bins

    # TODO: c'est ici qu'on peut calculer quelles sont les features constantes dans
    #  le node (si on trouve un bin avec toutes les features. Faudra faire attention
    #  au cas avec missing values a terme
    n_samples_in_bins = np.zeros((max_features, n_bins), dtype=np_float32)
    y_counts_in_bins = np.zeros((max_features, n_bins, n_classes), dtype=np_float32)
    # A counter for the features since return arrays are contiguous
    f = 0
    # For-loop on features and then samples (X is F-major) for cache hits
    for feature in features:
        for sample in train_indices:
            bin = X[sample, feature]
            label = y[sample]
            # sample_weight = sample_weight_train[sample]
            # TODO: c'est galere faudrait plutot avec un seul vecteur de
            #  sample_weight pour train et valid, car on utilise train_indices
            sample_weight = nb_float32(1.0)
            # One more sample in this bin for the current feature
            n_samples_in_bins[f, bin] += sample_weight
            # One more sample in this bin for the current feature with this label
            y_counts_in_bins[f, bin, label] += sample_weight

        f += 1

    return n_samples_in_bins, y_counts_in_bins


# TODO: later, unroll the loops as below for histogram computations, for faster
#  computations


# @njit
# def _build_histogram_naive(n_bins, sample_indices, binned_feature,
#                            ordered_gradients, ordered_hessians):
#     """Build histogram in a naive way, without optimizing for cache hit."""
#     histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
#     for i, sample_idx in enumerate(sample_indices):
#         bin_idx = binned_feature[sample_idx]
#         histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
#         histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
#         histogram[bin_idx]['count'] += 1
#     return histogram
#
#
# @njit
# def _subtract_histograms(n_bins, hist_a, hist_b):
#     """Return hist_a - hist_b"""
#
#     histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
#
#     sg = 'sum_gradients'
#     sh = 'sum_hessians'
#     c = 'count'
#
#     for i in range(n_bins):
#         histogram[i][sg] = hist_a[i][sg] - hist_b[i][sg]
#         histogram[i][sh] = hist_a[i][sh] - hist_b[i][sh]
#         histogram[i][c] = hist_a[i][c] - hist_b[i][c]
#
#     return histogram
#
#
# @njit
# def _build_histogram(n_bins, sample_indices, binned_feature, ordered_gradients,
#                      ordered_hessians):
#     """Return histogram for a given feature."""
#     histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
#     n_node_samples = sample_indices.shape[0]
#     unrolled_upper = (n_node_samples // 4) * 4
#
#     for i in range(0, unrolled_upper, 4):
#         bin_0 = binned_feature[sample_indices[i]]
#         bin_1 = binned_feature[sample_indices[i + 1]]
#         bin_2 = binned_feature[sample_indices[i + 2]]
#         bin_3 = binned_feature[sample_indices[i + 3]]
#
#         histogram[bin_0]['sum_gradients'] += ordered_gradients[i]
#         histogram[bin_1]['sum_gradients'] += ordered_gradients[i + 1]
#         histogram[bin_2]['sum_gradients'] += ordered_gradients[i + 2]
#         histogram[bin_3]['sum_gradients'] += ordered_gradients[i + 3]
#
#         histogram[bin_0]['sum_hessians'] += ordered_hessians[i]
#         histogram[bin_1]['sum_hessians'] += ordered_hessians[i + 1]
#         histogram[bin_2]['sum_hessians'] += ordered_hessians[i + 2]
#         histogram[bin_3]['sum_hessians'] += ordered_hessians[i + 3]
#
#         histogram[bin_0]['count'] += 1
#         histogram[bin_1]['count'] += 1
#         histogram[bin_2]['count'] += 1
#         histogram[bin_3]['count'] += 1
#
#     for i in range(unrolled_upper, n_node_samples):
#         bin_idx = binned_feature[sample_indices[i]]
#         histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
#         histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
#         histogram[bin_idx]['count'] += 1
#
#     return histogram
#
#
# @njit
# def _build_histogram_no_hessian(n_bins, sample_indices, binned_feature,
#                                 ordered_gradients):
#     """Return histogram for a given feature.
#
#     Hessians are not updated (used when hessians are constant).
#     """
#     histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
#     n_node_samples = sample_indices.shape[0]
#     unrolled_upper = (n_node_samples // 4) * 4
#
#     for i in range(0, unrolled_upper, 4):
#         bin_0 = binned_feature[sample_indices[i]]
#         bin_1 = binned_feature[sample_indices[i + 1]]
#         bin_2 = binned_feature[sample_indices[i + 2]]
#         bin_3 = binned_feature[sample_indices[i + 3]]
#
#         histogram[bin_0]['sum_gradients'] += ordered_gradients[i]
#         histogram[bin_1]['sum_gradients'] += ordered_gradients[i + 1]
#         histogram[bin_2]['sum_gradients'] += ordered_gradients[i + 2]
#         histogram[bin_3]['sum_gradients'] += ordered_gradients[i + 3]
#
#         histogram[bin_0]['count'] += 1
#         histogram[bin_1]['count'] += 1
#         histogram[bin_2]['count'] += 1
#         histogram[bin_3]['count'] += 1
#
#     for i in range(unrolled_upper, n_node_samples):
#         bin_idx = binned_feature[sample_indices[i]]
#         histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
#         histogram[bin_idx]['count'] += 1
#
#     return histogram
#
#
# @njit
# def _build_histogram_root_no_hessian(n_bins, binned_feature, all_gradients):
#     """Special case for the root node
#
#     The root node has to find the split among all the samples from the
#     training set. binned_feature and all_gradients already have a consistent
#     ordering.
#
#     Hessians are not updated (used when hessians are constant)
#     """
#     histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
#     n_node_samples = binned_feature.shape[0]
#     unrolled_upper = (n_node_samples // 4) * 4
#
#     for i in range(0, unrolled_upper, 4):
#         bin_0 = binned_feature[i]
#         bin_1 = binned_feature[i + 1]
#         bin_2 = binned_feature[i + 2]
#         bin_3 = binned_feature[i + 3]
#
#         histogram[bin_0]['sum_gradients'] += all_gradients[i]
#         histogram[bin_1]['sum_gradients'] += all_gradients[i + 1]
#         histogram[bin_2]['sum_gradients'] += all_gradients[i + 2]
#         histogram[bin_3]['sum_gradients'] += all_gradients[i + 3]
#
#         histogram[bin_0]['count'] += 1
#         histogram[bin_1]['count'] += 1
#         histogram[bin_2]['count'] += 1
#         histogram[bin_3]['count'] += 1
#
#     for i in range(unrolled_upper, n_node_samples):
#         bin_idx = binned_feature[i]
#         histogram[bin_idx]['sum_gradients'] += all_gradients[i]
#         histogram[bin_idx]['count'] += 1
#
#     return histogram
#
#
# @njit
# def _build_histogram_root(n_bins, binned_feature, all_gradients,
#                           all_hessians):
#     """Special case for the root node
#
#     The root node has to find the split among all the samples from the
#     training set. binned_feature and all_gradients and all_hessians already
#     have a consistent ordering.
#     """
#     histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
#     n_node_samples = binned_feature.shape[0]
#     unrolled_upper = (n_node_samples // 4) * 4
#
#     for i in range(0, unrolled_upper, 4):
#         bin_0 = binned_feature[i]
#         bin_1 = binned_feature[i + 1]
#         bin_2 = binned_feature[i + 2]
#         bin_3 = binned_feature[i + 3]
#
#         histogram[bin_0]['sum_gradients'] += all_gradients[i]
#         histogram[bin_1]['sum_gradients'] += all_gradients[i + 1]
#         histogram[bin_2]['sum_gradients'] += all_gradients[i + 2]
#         histogram[bin_3]['sum_gradients'] += all_gradients[i + 3]
#
#         histogram[bin_0]['sum_hessians'] += all_hessians[i]
#         histogram[bin_1]['sum_hessians'] += all_hessians[i + 1]
#         histogram[bin_2]['sum_hessians'] += all_hessians[i + 2]
#         histogram[bin_3]['sum_hessians'] += all_hessians[i + 3]
#
#         histogram[bin_0]['count'] += 1
#         histogram[bin_1]['count'] += 1
#         histogram[bin_2]['count'] += 1
#         histogram[bin_3]['count'] += 1
#
#     for i in range(unrolled_upper, n_node_samples):
#         bin_idx = binned_feature[i]
#         histogram[bin_idx]['sum_gradients'] += all_gradients[i]
#         histogram[bin_idx]['sum_hessians'] += all_hessians[i]
#         histogram[bin_idx]['count'] += 1
#
#     return histogram
