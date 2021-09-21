# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""This modules contains the Encoder class. The Encoder performs the transformation
of an input pandas dataframe or numpy array into a Dataset class. It performs ordinal
encoding of categorical features and binning using quantile intervals for continuous
features.

TODO: also, it deals with missing values by using the last bin when a column has
missing values

"""
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state

from wildwood.preprocessing import Dataset, dataset_fill_column, dataset_to_array

from ._binning import _find_binning_thresholds, _bin_continuous_column

# TODO: fit and transform in parallel over columns
# TODO: use bitsets for known categories ?


def check_X(X, n_features=None):
    """Checks if input is a 2d numpy array or a pandas dataframe.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to fit and transform. It can be either a pandas dataframe or a 2D
        numpy array.

    Returns
    -------
    output : bool
        True if input is a pandas dataframe, False if it is a numpy array
    """
    if hasattr(X, "ndim") and hasattr(X, "shape"):
        # It can be either a pandas dataframe or a numpy array
        if hasattr(X, "dtypes") and hasattr(X, "columns") and hasattr(X, "index"):
            # It really looks like a pandas dataframe
            is_dataframe = True
        elif hasattr(X, "dtype"):
            # It really looks like a numpy array
            if X.ndim == 2:
                is_dataframe = False
            else:
                raise ValueError("X is must be a pandas dataframe or a 2d numpy array")
        else:
            raise ValueError("X is must be a pandas dataframe or a 2d numpy array")
    else:
        raise ValueError("X is not a pandas dataframe or a numpy array")

    if n_features is not None:
        if X.shape[1] != n_features:
            msg = (
                "The number of features in X is different to the number of "
                "features of the fitted data. The fitted data had {0} features "
                "and the X has {1} features.".format(n_features, X.shape[1])
            )
            raise ValueError(msg)

    return is_dataframe


class Encoder(TransformerMixin, BaseEstimator):
    """A class that transforms an input pandas dataframe or numpy array into a
    Dataset class, corresponding to a column-wise binning of the original
    columns.

    Categorical columns are simply ordinal-encoded using contiguous non-negative
    integers, while continuous columns are binned using inter-quantiles intervals,
    so that each bin contains approximately the same number of samples.

    Both mappings from categories to integers (for categorical columns) and from
    inter-quantile intervals to integers (for continuous columns) are computed using
    the `.fit()` method.

    The `.transform()` method will create the binned dataset. Its default behaviour is
    to raise an error whenever an unknown category is met, but this can be changed
    using the `handle_unknown` option.

    If a column contains missing values, the last bin (last integer) is used to
    encode them.

    Parameters
    ----------
    max_bins : int
        The maximum number of bins to use, including the bin for missing values.
        This should be at least 4, and this impacts both categorical and continuous
        columns as follows. For a continuous column, we will use `max_bins` bins when
        the column has no missing values, and use `max_bins - 1` bins if it does. The
        last bin (`max_bins - 1`) is used to encode missing values. When a continuous
        column has less than `max_bins` different inter-quantile intervals , we use
        than `max_bins` bins for it. For categorical columns with `M` categories
        (including missing values), we use `M` bins when `M <= max_bins` and use
        `max_bins` otherwise, by grouping together all the sparsest categories
        corresponding to non-missing values.

    subsample : int or None, default=2e5
        If `n_samples > subsample`, then `subsample` samples are chosen at random to
        compute the quantiles. If `None`, then the whole data is used.

    is_categorical : ndarray
        blabla

    handle_unknown : {"error", "consider_missing"}, default="error"
        If set to "error", an error will be raised at transform whenever a category was
        not seen during fit. If set to "consider_missing", we will consider it as a
        missing value (it will end up in the same bin as missing values).

    n_jobs : 1
        blabla

    random_state : None
        blabla

    verbose : bool
        Blabla

    Attributes
    ----------
    n_samples_in_ : int
        The number of samples passed to fit

    n_features_in_ : int
        The number of features (columns) passed to fit

    categories_ : dict of ndarrays
        A dictionary that maps the index of a categorical column to an array
        containing its raw categories. For instance, categories_[2][4] is the raw
        category corresponding to the bin index 4 for column index 2

    binning_thresholds_ : dict of ndarrays
        A dictionary that maps the index of a continuous column to an array
        containing its binning thresholds. It is usually of length max_bins - 1,
        unless the column has less unique values than that.

    is_categorical_ : ???

        self.has_missing_values_ = None
        self.n_bins_no_missing_values_ = None

    """

    def __init__(
        self,
        max_bins=256,
        subsample=int(2e5),
        is_categorical=None,
        handle_unknown="error",
        n_jobs=-1,
        random_state=None,
        verbose=False,
    ):
        self.max_bins = max_bins
        self.subsample = subsample
        self.is_categorical = is_categorical
        self.handle_unknown = handle_unknown
        self.random_state = random_state
        # And initialize the attributes
        self.n_samples_in_ = None
        self.n_features_in_ = None
        self.categories_ = None
        self.binning_thresholds_ = None
        self.is_categorical_ = None
        self.n_bins_no_missing_values_ = None

    def fit(self, X, y=None):
        """This computes, for each column of X, a mapping from raw values to bins:
        categories to integers for categorical columns and inter-quantile intervals
        for continuous columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform. It can be either a pandas dataframe or a 2d
            numpy array.

        y: None
            This is ignored.

        Returns
        -------
        self : object
            The current Encoder instance
        """
        # TODO: test for categories will too low modalities
        is_dataframe = check_X(X)
        n_samples, n_features = X.shape

        is_categorical = np.zeros(n_features, dtype=np.bool)
        categories = {}
        binning_thresholds = {}
        # TODO: check is_categorical, size, dtype, etc.

        max_bins = self.max_bins

        # An array containing the number of bins for each column, without the bin for
        # missing values or unknown categories. Note that this corresponds as well to
        # the bin used for possible missing values or unknown categories.
        n_bins_no_missing_values = np.empty((n_features,), dtype=np.uint64)

        rng = check_random_state(self.random_state)
        if self.subsample is not None and n_samples > self.subsample:
            subset = rng.choice(n_samples, self.subsample, replace=False)
        else:
            subset = None

        if self.is_categorical is None:
            if is_dataframe:
                for col_idx, (col_name, col) in enumerate(X.items()):
                    col_dtype = col.dtype

                    if col_dtype.name == "category":
                        is_categorical[col_idx] = True
                        # Save the categories
                        col_categories = col.cat.categories.values
                        categories[col_idx] = col_categories
                        # Number of bins is the number of categories (and do not
                        # include missing values)
                        n_bins_no_missing_values[col_idx] = col_categories.size
                    else:
                        if subset is not None:
                            col = col.take(subset)
                        col_binning_thresholds = _find_binning_thresholds(
                            col, max_bins=max_bins, col_is_pandas_series=True
                        )
                        binning_thresholds[col_idx] = col_binning_thresholds
                        # The number of bins used is the number of binning thresholds
                        # plus one
                        n_bins_no_missing_values[col_idx] = (
                            col_binning_thresholds.size + 1
                        )
            else:
                raise ValueError("Only pandas DataFrame objects are supported for now")

        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features
        self.categories_ = categories
        self.binning_thresholds_ = binning_thresholds
        self.is_categorical_ = is_categorical
        self.n_bins_no_missing_values_ = n_bins_no_missing_values
        return self

    def transform(self, X, y=None):
        """Bins the columns in X. Both continuous and categorical columns are mapped
        to a contiguous range of non-negative integers. The resulting binned data is
        stored in a memory-efficient `Dataset` class, which uses internally a bitarray.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform using binning. It can be either a pandas dataframe or
            a 2d numpy array.

        y: None
            This is ignored.

        Returns
        -------
        output : Dataset
            A WildWood `Dataset` class corresponding to the binned data.
        """
        is_dataframe = check_X(X, self.n_features_in_)

        # TODO: check that "categories" are for the same columns at the ones passed
        #  to fit
        n_samples, n_features = X.shape
        # TODO: many tests are missing here, such as for n_features...
        n_features = self.n_features_in_

        is_categorical = self.is_categorical_
        categories = self.categories_
        binning_thresholds = self.binning_thresholds_
        n_bins_no_missing_values = self.n_bins_no_missing_values_

        # This array will contain the maximum value in each binned column. This
        # corresponds to the number of bins used minus one {0, 1, 2, ..., n_bins - 1}.
        max_values = np.empty((n_features,), dtype=np.uint64)

        # First, we need to know about the actual number of bins needed for each
        # column in X.
        if is_dataframe:
            # TODO: check that X has the correct dtype
            # First, we need to know if some columns from X have unknown categories,
            # since this might increase by one the number of bins required.

            # We perform a first loop over the columns of X to find out about the
            # required number of bins
            for col_idx, (col_name, col) in enumerate(X.items()):
                # Does the column have missing values ?
                col_has_missing_values = col.hasnans
                # has_missing_values[col_idx] = col_has_missing_values
                # Number of bins required for this column
                col_n_bins = n_bins_no_missing_values[col_idx]

                if is_categorical[col_idx]:
                    col_categories_at_fit = set(categories[col_idx])
                    col_categories_at_transform = set(col.cat.categories)
                    # TODO: is this the best approach for this test ?
                    col_has_unknown_categories = not col_categories_at_transform.issubset(
                        col_categories_at_fit
                    )

                    if col_has_unknown_categories and self.handle_unknown == "error":
                        # We don't allow unknown categories, so we raise an error
                        unknowns = col_categories_at_transform.difference(
                            col_categories_at_fit
                        )
                        msg = (
                            "Found unknown categories {0} in column {1} during "
                            "transform".format(unknowns, col_idx)
                        )
                        raise ValueError(msg)

                    # We need an extra bin if there are missing values, or an unknown
                    # category (and we did not raised an error above).
                    requires_extra_bin = (
                        col_has_missing_values or col_has_unknown_categories
                    )
                else:
                    requires_extra_bin = col_has_missing_values

                if requires_extra_bin:
                    col_n_bins += 1

                # Once again, max_value is the number of bins minus one
                max_values[col_idx] = col_n_bins - 1
        else:
            raise NotImplementedError("We only support pandas dataframes for now")

        # It's time now to instantiate the dataset
        dataset = Dataset(n_samples, max_values)

        # And to loop again over the columns to fill it
        if is_dataframe:
            for col_idx, (col_name, col) in enumerate(X.items()):
                if is_categorical[col_idx]:
                    col_categories_at_fit = categories[col_idx]
                    # Use the same categories as the ones used in fit and replace
                    # unknown categories by missing values
                    col = col.cat.set_categories(
                        col_categories_at_fit, inplace=False
                    )
                    if col.hasnans:
                        # Missing values are now either actual missing values or unknown
                        # categories. Internally, the code used by pandas for missing
                        # values is always -1, while we want to use instead the last bin
                        # index to stick to non-negative integers.
                        # TODO: Direct numba code for this by testing the -1 directly ?
                        binned_col = col.cat.codes.values.copy()
                        binned_col[col.isna()] = col_categories_at_fit.size
                    else:
                        # No copy is required
                        binned_col = col.cat.codes.values
                else:
                    missing_value_bin = max_values[col_idx]
                    binned_col = np.empty((n_samples,), dtype=np.uint64)
                    _bin_continuous_column(
                        col.values,
                        binning_thresholds[col_idx],
                        missing_value_bin,
                        binned_col,
                    )

                # Time to add the binned column in the dataset
                dataset_fill_column(dataset, col_idx, binned_col)
        else:
            raise NotImplementedError("We only support pandas dataframes for now")

        return dataset

    # TODO: the inverse_transform is still somewhat brittle... values in columns will
    #  be ok, but dtypes, index and columns will be ok in standard cases, but not
    #  with particular / weird cases
    def inverse_transform(
        self, dataset, return_dataframe=True, columns=None, index=None
    ):
        """

        Parameters
        ----------
        dataset
        return_dataframe
        columns
        index

        Returns
        -------

        """
        # Get back the original binned columns from the dataset
        X_binned = dataset_to_array(dataset)

        # TODO: keep also column and index information to rebuild the exact same
        #  dataframe or pass those to inverse_transform() ?

        df = pd.DataFrame()

        is_categorical = self.is_categorical_
        categories = self.categories_
        binning_thresholds = self.binning_thresholds_
        n_bins_no_missing_values = self.n_bins_no_missing_values_

        for col_idx in range(self.n_features_in_):
            binned_col = X_binned[:, col_idx]
            # The bin used for missing values equals the number of bins used for
            # non-missing values
            missing_values_bin = n_bins_no_missing_values[col_idx]
            missing_mask = binned_col == missing_values_bin
            has_missing_values = missing_mask.any()

            if is_categorical[col_idx]:
                col_categories = categories[col_idx]
                # Find out where the missing values are: they use the bin
                # corresponding to missing values
                # If there are missing values, we need to replace their bins by
                # something smaller to get back the raw categories
                if has_missing_values:
                    binned_col[missing_mask] = 0

                # Get back the categories from the codes
                col = pd.Categorical(col_categories[binned_col])

                if has_missing_values:
                    # If it has missing values we need to put then back
                    if col.dtype.kind == "i":
                        # Integers to not support missing values, we need to change
                        # to float
                        col = col.astype(np.float)

                    col[missing_mask] = np.nan
            else:
                # For continuous features, we replace bins by the corresponding
                # inter-quantiles intervals
                col_binning_thresholds = np.concatenate(
                    ([-np.inf], binning_thresholds[col_idx], [np.inf])
                )

                intervals = pd.arrays.IntervalArray.from_arrays(
                    col_binning_thresholds[:-1], col_binning_thresholds[1:],
                    closed="right"
                )
                if has_missing_values:
                    binned_col[missing_mask] = 0

                col = intervals[binned_col]
                if has_missing_values:
                    col[missing_mask] = np.nan

            df[col_idx] = col

        return df

    @property
    def max_bins(self):
        return self._max_bins

    @max_bins.setter
    def max_bins(self, val):
        if not isinstance(val, int):
            raise ValueError("max_bins must be an integer number")
        else:
            if val < 4:
                raise ValueError("max_bins must be >= 4")
            else:
                self._max_bins = val

    @property
    def handle_unknown(self):
        return self._handle_unknown

    @handle_unknown.setter
    def handle_unknown(self, val):
        if val in {"error", "consider_missing"}:
            self._handle_unknown = val
        else:
            msg = (
                "handle_unknown must be 'error' or 'consider_missing' but "
                "got {0}".format(val)
            )
            raise ValueError(msg)

    @property
    def subsample(self):
        return self._subsample

    @subsample.setter
    def subsample(self, val):
        if val is None:
            self._subsample = val
        elif isinstance(val, (int, float)) and val >= 50_000:
            self._subsample = int(val)
        else:
            msg = "subsample should be None or a number >= 50000"
            raise ValueError(msg)
