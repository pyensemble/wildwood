# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause


"""This modules contains private jit-compiled utilities for binning
continuous features.
"""

import numpy as np
from math import isnan
from numba import jit, uint64


# Global jit decorator options
NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False
CACHE = True

ALMOST_INF = 1e300


# TODO: put back signatures everywhere

def _find_binning_thresholds(col, max_bins, col_is_pandas_series=False):
    """Extract quantiles from a continuous feature.

    Missing values are ignored for finding the thresholds.

    Parameters
    ----------
    col : array-like
        A numpy ndarray of shape (n_samples,) or a pandas Series for which we
        compute binning
        thresholds.

    max_bins: int
        The maximum number of bins to use for non-missing values. If a column
        contains less than `max_bins` values, these values will be used to compute
        the bin thresholds, instead of the quantiles.

    col_is_pandas_series

    Return
    ------
    binning_thresholds : ndarray of shape(min(max_bins, n_unique_values) - 1,)
        The increasing numeric values that can be used to separate the bins.
        A given value x will be mapped into bin value i iff
        binning_thresholds[i - 1] < x <= binning_thresholds[i]

    Notes
    -----
    This function is a minor modification of scikit-learn's function by Nicolas Hug, see
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_hist_gradient_boosting/binning.py
    """
    if col_is_pandas_series:
        if col.hasnans:
            col = col.dropna()
        col = col.values
    else:
        missing_mask = np.isnan(col)
        if missing_mask.any():
            col = col[~missing_mask]

    # TODO: it's not usefull IMO
    col = np.ascontiguousarray(col, dtype=np.float64)
    uniques = np.unique(col)

    if uniques.size <= max_bins:
        middles = uniques[:-1] + uniques[1:]
        middles *= 0.5
    else:
        # TODO: create percentiles only once once
        percentiles = np.linspace(0, 100, num=max_bins + 1)
        percentiles = percentiles[1:-1]
        # TODO: is there a faster approach ?
        middles = np.percentile(col, percentiles, interpolation="midpoint")
        assert middles.shape[0] == max_bins - 1

    # We avoid having +inf thresholds: +inf thresholds are only allowed in
    # a "split on nan" situation.
    np.clip(middles, a_min=None, a_max=ALMOST_INF, out=middles)
    return middles


@jit(
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    cache=CACHE,
)
def _bin_continuous_column(col, binning_thresholds, missing_values_bin, binned_col):
    """This performs binary search to find the bin index of each value in the column

    Parameters
    ----------
    col
    binning_thresholds
    missing_values_bin
    binned_col

    Returns
    -------

    """
    for i, x in enumerate(col):
        if isnan(x):
            binned_col[i] = missing_values_bin
        else:
            left, right = 0, binning_thresholds.shape[0]
            while left < right:
                # Equal to (right + left - 1) // 2 but avoids overflow
                middle = left + (right - left - 1) // 2
                if col[i] <= binning_thresholds[middle]:
                    right = middle
                else:
                    left = middle + 1

            binned_col[i] = left
