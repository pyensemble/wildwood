# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""This module contains the Encoder class. The Encoder performs the transformation
of an input ``pandas.DataFrame`` or ``numpy.ndarray`` into a ``FeaturesBitArray``
class.
"""

from math import floor, log, sqrt
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state

from wildwood.preprocessing import (
    FeaturesBitArray,
    features_bitarray_fill_column,
    features_bitarray_to_array,
)
from ._binning import _find_binning_thresholds, _bin_continuous_column
from ._checks import check_X, is_series_categorical, get_array_dtypes


# TODO: what if for a numerical column, there is no missing values at fit time,
#  and missing values at training time ? This should be tested against.


class Encoder(TransformerMixin, BaseEstimator):
    """A class that transforms an input ``pandas.DataFrame`` or ``numpy.ndarray`` into
    a ``wildwood.FeaturesBitArray`` class, corresponding to a column-wise binning of the
    original columns.

    Categorical columns are simply ordinal-encoded using contiguous non-negative
    integers, while continuous columns are binned using inter-quantiles intervals,
    so that each bin contains approximately the same number of samples.

    Both mappings from categories to integers (for categorical columns) and from
    inter-quantile intervals to integers (for continuous columns) are computed using
    the ``.fit()`` method.

    The ``.transform()`` method will bin the features and create the features' bitarray.
    Its default behavior is to raise an error whenever an unknown category is met,
    but this can be changed using the ``handle_unknown`` option.

    When the input is a ``pandas.DataFrame``, we support the encoding of missing
    values both for categorical and numerical columns. However, when the input is a
    ``numpy.ndarray``, missing values are supported only for a numerical data type.
    Other situations might raise unexpected errors.
    If a column contains missing values, the last bin (last integer) is used to
    encode them.

    Parameters
    ----------
    max_bins : int
        The maximum number of bins for numerical columns, not including the bin used
        for missing values, if any. Should be at least 3.
        We will use ``max_bins`` bins when the column has no missing values, and
        ``max_bins + 1`` bins if it does.
        The last bin (at index ``max_bins``) is used to encode missing values.
        If a column has less than ``max_bins`` different inter-quantile or
        categories, we use less than ``max_bins`` bins for it.

    is_categorical : None or numpy.ndarray
        If not ``None``, it is a ``numpy.ndarray`` of shape (n_features,) with boolean
        dtype, which corresponds to a categorical indicator for each column as
        specified by the user.

    handle_unknown : {"error", "consider_missing"}, default="error"
        If set to "error", an error will be raised at transform whenever a category was
        not seen during fit. If set to "consider_missing", we will consider it as a
        missing value (it will end up in the same bin as missing values).

    cat_min_categories : int or {"log", "sqrt"}, default="log"
        When a column is numerical and ``is_categorical`` is None, WildWood decides
        that it is categorical whenever its number of unique values is smaller or
        equal to ``cat_min_categories``. Otherwise, it is considered numerical.
        If an int larger than 3 is given, we use it as ``cat_min_categories``.
        If "log", we set ``cat_min_categories=max(2, floor(log(n_samples)))``.
        If "sqrt", we set ``cat_min_categories=max(2, floor(sqrt(n_samples)))``.
        Default is "log".

    subsample : int or None, default=200000
        If ``n_samples > subsample``, then ``subsample`` samples are chosen at random
        to compute the quantiles. If ``None``, the whole dataset is used.

    random_state: int, default=None
        Allows to seed the random number generator used to generate a subsample for
        quantiles computation.

    verbose : bool, default=False
        If True, display warnings concerning columns typing.

    Attributes
    ----------
    n_samples_in_ : int
        The number of samples passed to fit.

    n_features_in_ : int
        The number of features (columns) passed to fit.

    categories_ : dict
        A dictionary that maps the index of a categorical column to an array
        containing its raw categories. For instance, categories_[2][4] is the raw
        category corresponding to the bin index 4 for column index 2.

    binning_thresholds_ : dict
        A dictionary that maps the index of a continuous column to an array
        containing its binning thresholds. It is usually of length ``max_bins - 1``,
        unless the column has less unique values than that.

    is_categorical_ : numpy.ndarray
        A numpy array of shape (n_features,) with boolean dtype, which indicates if
        each feature is considered as categorical by WildWood or not. This might
        differ from the ``is_categorical`` given by the user. See the ``_checks.py``
        module for details.
    """

    def __init__(
        self,
        max_bins=256,
        subsample=int(2e5),
        is_categorical=None,
        cat_min_categories="log",
        handle_unknown="error",
        random_state=None,
        verbose=False,
    ):
        self.max_bins = max_bins
        self.subsample = subsample
        self.cat_min_categories = cat_min_categories
        self.is_categorical = is_categorical
        self.handle_unknown = handle_unknown
        self.random_state = random_state
        self.verbose = verbose

        self._fitted = False
        self._n_samples_in_ = None
        self._n_features_in_ = None
        self._categories_ = None
        self._binning_thresholds_ = None
        self._is_categorical_ = None
        self._has_missing_values_ = None
        self._n_bins_no_missing_values_ = None
        self._cat_min_categories_ = None

    def fit(self, X, y=None):
        """This computes, for each column of X, a mapping from raw values to bins:
        categories to integers for categorical columns and inter-quantile intervals
        for continuous columns. It also infers if columns are either categorical or
        continuous using the ``is_categorical`` attribute, and tries to guess it if not
        provided by the user.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform. It can be either a ``pandas.DataFrame``
            or a 2D ``numpy.ndarray``.

        y: None
            This is ignored.

        Returns
        -------
        self : Encoder
            The current Encoder instance
        """
        # TODO: a faster fit transform where we avoid doing the same thing twice
        # TODO: test for constant columns
        is_dataframe = check_X(X, is_categorical=self.is_categorical)
        n_samples, n_features = X.shape
        max_bins = self.max_bins

        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features

        categories = {}
        binning_thresholds = {}

        # An array containing the number of bins for each column, without the bin for
        # missing values or unknown categories. Note that this corresponds as well to
        # the index of the bin used for possible missing values or unknown categories.
        n_bins_no_missing_values = np.empty((n_features,), dtype=np.uint64)

        if self.subsample is not None and n_samples > self.subsample:
            rng = check_random_state(self.random_state)
            subset = rng.choice(n_samples, self.subsample, replace=False)
        else:
            subset = None

        if isinstance(self.cat_min_categories, int):
            cat_min_categories_ = self.cat_min_categories
        elif self.cat_min_categories == "log":
            cat_min_categories_ = int(max(2, floor(log(n_samples))))
        else:
            cat_min_categories_ = int(max(2, floor(sqrt(n_samples))))

        self.cat_min_categories_ = cat_min_categories_

        if is_dataframe:
            # This array will contain the actual boolean mask corresponding to the
            # categorical features, which might differ from self.is_categorical
            is_categorical_ = np.zeros(n_features, dtype=np.bool_)

            for col_idx, (col_name, col) in enumerate(X.items()):

                if self.is_categorical is None:
                    is_col_categorical = is_series_categorical(
                        col,
                        is_categorical=None,
                        cat_min_categories=cat_min_categories_,
                        verbose=self.verbose,
                    )
                else:
                    is_col_categorical = is_series_categorical(
                        col,
                        is_categorical=self.is_categorical[col_idx],
                        cat_min_categories=cat_min_categories_,
                        verbose=self.verbose,
                    )

                if is_col_categorical == "is_category":
                    # The column is already categorical. We save it as actually
                    # categorical and save its categories
                    is_categorical_[col_idx] = True
                    col_categories = col.cat.categories.values
                    categories[col_idx] = col_categories
                    # Number of bins is the number of categories (not including missing
                    #  values, if any)
                    n_bins_no_missing_values[col_idx] = col_categories.size

                elif is_col_categorical == "to_category":
                    # The column must be converted to a categorical one
                    is_categorical_[col_idx] = True
                    # TODO: actually creating the column might be avoided ?
                    col_as_category = col.astype("category")
                    col_categories = col_as_category.cat.categories.values
                    categories[col_idx] = col_categories
                    n_bins_no_missing_values[col_idx] = col_categories.size

                else:
                    # In this case is_col_categorical == "numerical":
                    if subset is not None:
                        col = col.take(subset)
                    col_binning_thresholds = _find_binning_thresholds(
                        col, max_bins=max_bins, col_is_pandas_series=True
                    )
                    binning_thresholds[col_idx] = col_binning_thresholds
                    # The actual number of bins used is the number of binning
                    # thresholds (which can be smaller than max_bins, if there is
                    # less unique values than that) + 1
                    n_bins_no_missing_values[col_idx] = col_binning_thresholds.size + 1

        else:
            is_categorical_ = get_array_dtypes(
                X, self.is_categorical, cat_min_categories_, self.verbose
            )
            # TODO: attention ici on part du principe aussi que X ne peut contenir de
            #  donnees manquantes que si son dtype est numerique
            for col_idx in range(n_features):
                col = X[:, col_idx]
                is_col_categorical = is_categorical_[col_idx]
                if is_col_categorical:
                    col = pd.Series(col, dtype="category")
                    col_categories = col.cat.categories.values
                    categories[col_idx] = col_categories
                    n_bins_no_missing_values[col_idx] = col_categories.size
                else:
                    if subset is not None:
                        col = col.take(subset)

                    # TODO: keep is as numpy array
                    col = pd.Series(col)
                    col_binning_thresholds = _find_binning_thresholds(
                        col, max_bins=max_bins, col_is_pandas_series=True
                    )
                    binning_thresholds[col_idx] = col_binning_thresholds
                    # The number of bins used is the number of binning thresholds
                    # plus one
                    n_bins_no_missing_values[col_idx] = col_binning_thresholds.size + 1

        self.categories_ = categories
        self.binning_thresholds_ = binning_thresholds
        self.is_categorical_ = is_categorical_
        self.n_bins_no_missing_values_ = n_bins_no_missing_values
        self._fitted = True
        return self

    def transform(self, X, y=None):
        """Bins the columns in X. Both continuous and categorical columns are mapped
        to a contiguous range of non-negative integers. The resulting binned data is
        stored in a memory-efficient `FeaturesBitArray` class, which uses internally a
        bitarray.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform using binning. It can be either a pandas dataframe or
            a 2d numpy array.

        y: None
            This is ignored.

        Returns
        -------
        output : FeaturesBitArray
            A WildWood `FeaturesBitArray` class corresponding to the binned data.
        """
        is_dataframe = check_X(
            X, is_categorical=self.is_categorical, n_features=self.n_features_in_
        )
        n_samples, n_features = X.shape
        n_features = self.n_features_in_

        is_categorical_ = self.is_categorical_
        categories = self.categories_
        binning_thresholds = self.binning_thresholds_
        n_bins_no_missing_values = self.n_bins_no_missing_values_

        # This array will contain the maximum value in each binned column. This
        # corresponds to the number of bins used minus one {0, 1, 2, ..., n_bins - 1}.
        max_values = np.empty((n_features,), dtype=np.uint64)

        X_dict = {}

        # First, we need to know about the actual number of bins needed for each
        # column in X and we need to convert columns to category dtype is necessary
        if is_dataframe:
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

                if is_categorical_[col_idx]:
                    col_categories_at_fit = set(categories[col_idx])

                    if col.dtype.name != "category":
                        col = col.astype("category")

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

                # Save the eventually transformed column (if categorical)
                X_dict[col_name] = col
        else:
            # First, we need to know if some columns from X have unknown categories,
            # since this might increase by one the number of bins required. Note that
            # if X has object dtype (or similar), this code assumes that it cannot
            # contain missing values. We might obtain missing values if unknown
            # categories occur.
            if X.dtype.kind in "uif":
                # If X has a numerical dtype, we can look for missing values
                has_missing_values = np.isnan(X).any(axis=0)
            else:
                # Otherwise, there is no missing value
                has_missing_values = np.zeros(n_features, dtype=np.bool_)

            for col_idx in range(n_features):
                col = X[:, col_idx]
                # Does the column have missing values ?
                col_has_missing_values = has_missing_values[col_idx]
                # Number of bins required for this column
                col_n_bins = n_bins_no_missing_values[col_idx]

                if is_categorical_[col_idx]:
                    col_categories_at_fit = set(categories[col_idx])
                    # First, we convert the column to a categorical pandas Series
                    col = pd.Series(col, dtype="category")
                    col_categories_at_transform = set(col.cat.categories)
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
                    # category (and we did not raised an error above)
                    requires_extra_bin = (
                        col_has_missing_values or col_has_unknown_categories
                    )
                else:
                    requires_extra_bin = col_has_missing_values

                if requires_extra_bin:
                    col_n_bins += 1

                # Once again, max_value is the number of bins minus one
                max_values[col_idx] = col_n_bins - 1
                # Save the eventually transformed column (if categorical)
                X_dict[col_idx] = col

        # It's time now to instantiate the FeaturesBitArray
        features_bitarray = FeaturesBitArray(n_samples, max_values)

        # And to loop again over the columns to fill it
        if is_dataframe:
            for col_idx, (col_name, col) in enumerate(X_dict.items()):
                if is_categorical_[col_idx]:
                    col_categories_at_fit = categories[col_idx]
                    # Use the same categories as the ones used in fit and replace
                    # unknown categories by missing values.
                    col = col.cat.set_categories(col_categories_at_fit)#, inplace=False)

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

                # Time to add the binned column in the bitarray
                features_bitarray_fill_column(features_bitarray, col_idx, binned_col)
        else:
            is_dtype_numerical = X.dtype.kind in "uif"
            for col_idx in range(n_features):
                col = X_dict[col_idx]
                if is_categorical_[col_idx]:
                    col_categories_at_fit = categories[col_idx]
                    col = col.cat.set_categories(col_categories_at_fit)#, inplace=False)
                    if col.hasnans:
                        # Here, missing values can only come from unknown
                        # categories. Internally, the code used by pandas for missing
                        # values is always -1, while we want to use instead the last bin
                        # index to stick to non-negative integers.
                        binned_col = col.cat.codes.values.copy()
                        binned_col[col.isna()] = col_categories_at_fit.size
                    else:
                        # No copy is required
                        binned_col = col.cat.codes.values
                else:
                    if not is_dtype_numerical:
                        col = col.astype(np.float32)
                    missing_value_bin = max_values[col_idx]
                    binned_col = np.empty((n_samples,), dtype=np.uint64)
                    _bin_continuous_column(
                        col, binning_thresholds[col_idx], missing_value_bin, binned_col,
                    )

                # Time to add the binned column in the bitarray
                features_bitarray_fill_column(features_bitarray, col_idx, binned_col)

        return features_bitarray

    # TODO: the inverse_transform is still somewhat brittle... values in columns will
    #  be ok, but dtypes, index and columns will be ok in standard cases, but not
    #  with particular / weird cases
    def inverse_transform(
        self, features_bitarray, return_dataframe=True, columns=None, index=None
    ):
        """

        Parameters
        ----------
        features_bitarray
        return_dataframe
        columns
        index

        Returns
        -------

        """
        # Get back the original binned columns from the bitarray
        X_binned = features_bitarray_to_array(features_bitarray)

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
                    col_binning_thresholds[:-1],
                    col_binning_thresholds[1:],
                    closed="right",
                )
                if has_missing_values:
                    binned_col[missing_mask] = 0

                col = intervals[binned_col]
                if has_missing_values:
                    col[missing_mask] = np.nan

            df[col_idx] = col

        return df

    @property
    def n_samples_in_(self):
        if self._fitted:
            return self._n_samples_in_
        else:
            raise ValueError("You must call fit before asking for n_samples_in_")

    @n_samples_in_.setter
    def n_samples_in_(self, val):
        if isinstance(val, int) and val >= 1:
            self._n_samples_in_ = val
        else:
            raise ValueError("n_samples_in_ must be an int >= 1")

    @property
    def n_features_in_(self):
        if self._fitted:
            return self._n_features_in_
        else:
            raise ValueError("You must call fit before asking for n_features_in_")

    @n_features_in_.setter
    def n_features_in_(self, val):
        if isinstance(val, int) and val >= 1:
            self._n_features_in_ = val
        else:
            raise ValueError("n_features_int_ must be an int >= 1")

    @property
    def max_bins(self):
        return self._max_bins

    @max_bins.setter
    def max_bins(self, val):
        if not isinstance(val, int):
            raise ValueError("max_bins must be an integer number")
        else:
            if val < 3:
                raise ValueError("max_bins must be >= 3")
            else:
                self._max_bins = val

    @property
    def is_categorical(self):
        return self._is_categorical

    @is_categorical.setter
    def is_categorical(self, val):
        if val is None:
            self._is_categorical = val
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                if val.dtype.name == "bool":
                    self._is_categorical = val
                else:
                    self._is_categorical = val.astype(np.bool_)
            else:
                raise ValueError(
                    "is_categorical must be either None or a "
                    "one-dimensional array-like with bool-like dtype"
                )
        else:
            try:
                val = np.array(val, dtype=np.bool_)
            except ValueError:
                raise ValueError(
                    "is_categorical must be either None or a "
                    "one-dimensional array-like with bool-like dtype"
                )
            if val.ndim == 1:
                self._is_categorical = val
            else:
                raise ValueError(
                    "is_categorical must be either None or a "
                    "one-dimensional array-like with bool-like dtype"
                )

    @property
    def is_categorical_(self):
        return self._is_categorical_

    @is_categorical_.setter
    def is_categorical_(self, val):
        self._is_categorical_ = val

    @property
    def categories_(self):
        return self._categories_

    @categories_.setter
    def categories_(self, val):
        self._categories_ = val

    @property
    def binning_thresholds_(self):
        return self._binning_thresholds_

    @binning_thresholds_.setter
    def binning_thresholds_(self, val):
        self._binning_thresholds_ = val

    @property
    def has_missing_values_(self):
        return self._has_missing_values_

    @has_missing_values_.setter
    def has_missing_values_(self, val):
        self._has_missing_values_ = val

    @property
    def n_bins_no_missing_values_(self):
        return self._n_bins_no_missing_values_

    @n_bins_no_missing_values_.setter
    def n_bins_no_missing_values_(self, val):
        self._n_bins_no_missing_values_ = val

    @property
    def cat_min_categories_(self):
        return self._cat_min_categories_

    @cat_min_categories_.setter
    def cat_min_categories_(self, val):
        self._cat_min_categories_ = val

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

    @property
    def cat_min_categories(self):
        return self._cat_min_categories

    @cat_min_categories.setter
    def cat_min_categories(self, val):
        if isinstance(val, int):
            if val <= 3:
                raise ValueError(
                    f"cat_min_categories should be an int > 3 or either "
                    f"'log' or 'sqrt'; {val} was given"
                )
            else:
                self._cat_min_categories = val
        elif isinstance(val, str):
            if val not in {"log", "sqrt"}:
                raise ValueError(
                    f"cat_min_categories should be an int > 3 or either "
                    f"'log' or 'sqrt'; {val} was given"
                )
            else:
                self._cat_min_categories = val
