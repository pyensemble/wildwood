# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""This modules contains the Encoder class. The Encoder performs the transformation
of an input pandas dataframe or numpy array into a Dataset class. It performs ordinal
encoding of categorical features and binning using quantile intervals for continuous
features.

TODO: also, it deals with missing values by using the last bin when a column has
missing values

"""
import numbers
from math import floor, log, sqrt
import warnings
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state

from wildwood.preprocessing import Dataset, dataset_fill_column, dataset_to_array

from ._binning import _find_binning_thresholds, _bin_continuous_column

# TODO: fit and transform in parallel over columns
# TODO: use bitsets for known categories ?


def _is_series_categorical(col, is_categorical, max_modalities):
    """Decides if col contains categorical data or not and if it must be converted
    to the "category" data type. The following rules are applied:

    - If ``col`` data type is ``object``, ``str``, ``unicode`` or ``bool``
        - If ``is_categorical`` is ``None``
            - If ``col`` data type is already categorical: we do nothing
            - Else: we convert col to categorical and raise a warning
        - If ``is_categorical == True``
            - If ``col`` data type is already categorical: we do nothing
            - Else: we convert col to categorical
        - If ``is_categorical == False``
            - If ``col`` data type is already categorical: we do nothing but raise a
              warning
            - Else: we convert col to categorical and raise a warning
    - If ``col`` data type is ``uint``, ``int`` or any ``float``
        - If ``is_categorical`` is ``None``
            - If the number of unique values in ``col`` is <= ``max_modalities`` we
              convert ``col`` to categorical and raise a warning
            - Else: ``col`` is considered as numerical
        - If ``is_categorical == True``: we convert ``col`` to categorical
        - If ``is_categorical == False``: we consider ``col`` as numerical
    - Else: we raise an error

    Parameters
    ----------
    col : pandas.Series
        The column

    is_categorical : bool or None
        True if the user specified the column as categorical

    max_modalities : int
        When data type is numerical and the user gives no information
        (``is_categorical=None``), WildWood decides that the column is categorical
        whenever the number of unique values in col is <= max_modalities. Otherwise,
        it is considered numerical.

    Returns
    -------
    output : {"is_category", "to_category", "numerical"}
        The inferred type of column. If "is_category" the column is already
        categorical and does not need to be converted. If "to_category", the column
        is categorical but needs to be transformed as a pandas.Series with "category"
        data type. If "numerical", the column is numerical and won't be transformed
        into a categorical column.
    """
    assert isinstance(col, pd.Series)
    assert is_categorical in {True, False, None}
    dtype = col.dtype

    if dtype.kind in "bOSU":
        # The column's data type is either boolean, object, string or unicode,
        # which corresponds to a categorical column
        if is_categorical is None:
            # is_categorical is not specified by the user
            if dtype.name == "category":
                # It is already a categorical data type. Nothing to do here
                return "is_category"
            else:
                # It is not already categorical, but the data type correspond to
                # a categorical one, so we must convert it and inform the user
                warnings.warn(f"I will consider column {col.name} as categorical.")
                return "to_category"
        elif is_categorical:
            # is_categorical is True, the column is declared by the user as
            # categorical
            if dtype.name == "category":
                # It is already a categorical data type. Nothing to do here
                return "is_category"
            else:
                # It is not, but is declared categorical by the user, so we must
                # convert it
                return "to_category"
        else:
            # is_categorical is False. This means that the column is declared as
            # non-categorical by the user
            if dtype.name == "category":
                # It has categorical data type but was declared as
                # non-categorical by the user. We need to raise a warning to the
                # user even if the column is unchanged
                warnings.warn(
                    f"I will consider column {col.name} as categorical: it was "
                    f"declared non-categorical but has a categorical data type."
                )
                return "is_category"
            else:
                # The data type corresponds to a categorical one, but the user
                # declared it as non-categorical. We must convert it to
                # categorical and raise a warning to the user
                warnings.warn(
                    f"I will consider column {col.name} as categorical: it was "
                    f"declared non-categorical but has a categorical data type."
                )
                return "to_category"

    elif dtype.kind in "uif":
        # The column's data type is either unsigned int, int or float. The column
        # might be categorical or continuous, depending on what the user says and
        # the number of unique values in it.
        if is_categorical is None:
            # The user does not specifies if the column is categorical or not. We
            # check the number of unique values in the column and decide
            # automatically.
            n_modalities = col.nunique()
            if n_modalities <= max_modalities:
                warnings.warn(
                    f"I will consider column {col.name} as categorical: it has "
                    f"{n_modalities} unique values."
                )
                return "to_category"
            else:
                warnings.warn(
                    f"I will consider column {col.name} as numerical: it has "
                    f"{n_modalities} unique values."
                )
                return "numerical"
        elif is_categorical:
            # The user declared the column as categorical. We convert it to
            # categorical
            return "to_category"
        else:
            # The user declared the column as non-categorical. We will
            # consider it numerical
            return "numerical"

    else:
        # The column's data type is neither boolean, object, string, unicode,
        # unsigned int, int or float. It can be a date object or something else that
        # we do not support
        raise ValueError(
            f"Column {col.name} has dtype {col.dtype} which is not supported by "
            f"WildWood."
        )


def _get_array_dtypes(X, is_categorical, max_modalities):
    """Decides if each column of X contains categorical or numerical data. The
    following rules are applied:

    - If ``X`` data type is ``object``, ``str``, ``unicode`` or ``bool``
        - If ``is_categorical`` is ``None``
            - If the number of unique values in a column is <= ``max_modalities`` we
              consider it categorical and raise a warning
            - Else: the column is considered numerical and we raise a warning
        - If `is_categorical == True`
            - If `col` data type is already categorical: we do nothing
            - Else: we convert col to categorical
        - If `is_categorical == False`
            - If `col` data type is already categorical: we do nothing but raise a
              warning
            - Else: we convert col to categorical and raise a warning
    - If `col` data type is `uint`, `int` or any `float`
        - If `is_categorical` is `None`
            - If the number of unique values in `col` is <= `max_modalities` we
              convert `col` to categorical and raise a warning
            - Else: `col` is considered as numerical
        - If `is_categorical == True`: we convert `col` to categorical
        - If `is_categorical == False`: we consider `col` as numerical
    - Else: we raise an error

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix of features of shape (n_samples, n_features)

    is_categorical : None or ndarray
        If not None, it is a numpy array of shape (n_features,) with boolean dtype,
        which corresponds to a categorical indicator for each column.

    Returns
    -------
    output : numpy.ndarray
        A numpy array of shape (n_features,) with bool dtype indicating if each
        column is indeed categorical or not.
    """
    assert isinstance(X, np.ndarray)
    dtype = X.dtype
    n_samples, n_features = X.shape
    assert is_categorical is None or (
        isinstance(is_categorical, np.ndarray)
        and is_categorical.shape == (n_features,)
        and is_categorical.dtype.name == "bool"
    )
    is_categorical_out = np.zeros(n_features, dtype=np.bool_)
    if dtype.kind in "bOSU":
        # The input X as a dtype that can can correspond to a mixture of both
        # categorical and numerical columns.
        if is_categorical is None:
            # is_categorical is not specified by the user, so we need to "guess" if
            # each column is categorical or not
            for col_idx in range(n_features):
                col = X[:, col_idx]
                col_unique = np.unique(col)
                nunique = col_unique.size
                if nunique <= max_modalities:
                    # The column looks categorical
                    is_categorical_out[col_idx] = True
                    warnings.warn(f"I will consider column {col_idx} as categorical.")
                else:
                    # It looks numerical. Let's see if it can be transformed to float
                    success = True
                    try:
                        col.astype(np.float32)
                    except ValueError:
                        success = False
                    if success:
                        is_categorical_out[col_idx] = False
                        warnings.warn(f"I will consider column {col_idx} as numerical.")
                    else:
                        is_categorical_out[col_idx] = True
                        warnings.warn(
                            f"I will consider column {col_idx} as " f"categorical."
                        )
        else:
            # The user specified is_categorical, so we must follow what it says
            for col_idx in range(n_features):
                col = X[:, col_idx]
                is_col_categorical = is_categorical[col_idx]
                if is_col_categorical:
                    is_categorical_out[col_idx] = True
                else:
                    try:
                        col.astype(np.float32)
                    except ValueError:
                        # The column is declared as numerical, but it cannot be
                        # casted to float, so we raise an error
                        raise ValueError(
                            f"Column {col_idx} is declared as numerical, but it cannot "
                            f"be converted to float"
                        )
                    is_categorical_out[col_idx] = False

        return is_categorical_out
    elif dtype.kind in "uif":
        # The dtype of input X in integer or float
        if is_categorical is None:
            # is_categorical is not specified by the user, so we need to "guess" if
            # each column is categorical or not
            for col_idx in range(n_features):
                col = X[:, col_idx]
                col_unique = np.unique(col)
                nunique = col_unique.size
                if nunique <= max_modalities:
                    # The column looks categorical
                    is_categorical_out[col_idx] = True
                    warnings.warn(f"I will consider column {col_idx} as categorical.")
                else:
                    # It looks numerical. So need to check if it can be converted,
                    # since it is "uif"
                    is_categorical_out[col_idx] = False
                    warnings.warn(f"I will consider column {col_idx} as numerical.")
        else:
            # The user specified is_categorical, so we must follow what it says
            for col_idx in range(n_features):
                is_col_categorical = is_categorical[col_idx]
                if is_col_categorical:
                    is_categorical_out[col_idx] = True
                else:
                    is_categorical_out[col_idx] = False

        return is_categorical_out
    else:
        # X's data type is neither boolean, object, string, unicode, unsigned int,
        # int or float. It can be a date object or something else that we do not support
        raise ValueError(f"X has dtype {X.dtype} which is not supported by WildWood.")


def check_X(X, is_categorical=None, n_features=None):
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
    if isinstance(X, pd.DataFrame):
        # X is a pd.DataFrame
        is_dataframe = True
    elif isinstance(X, np.ndarray):
        if X.ndim == 2:
            is_dataframe = False
        else:
            raise ValueError(
                "X is must be a `pandas.DataFrame` or a "
                "two-dimensional `numpy.ndarray`."
            )
        if X.dtype.kind not in "bOSUuif":
            raise ValueError(f"The dtype of X {X.dtype} is not supported by WildWood")

    else:
        msg = (
            f"X is must be a `pandas.DataFrame` or a `numpy.ndarray`, a {type(X)} "
            f"was received."
        )
        raise ValueError(msg)

    if n_features is not None:
        if X.shape[1] != n_features:
            msg = (
                "The number of features in X is different from the number of "
                "features of the fitted data. The fitted data had {0} features "
                "and the X has {1} features.".format(n_features, X.shape[1])
            )
            raise ValueError(msg)

    if is_categorical is not None:
        if X.shape[1] != is_categorical.size:
            msg = (
                f"The number of features in X differs from the size of is_categorical. "
                f"X has shape {X.shape} while is_categorical has "
                f"shape {is_categorical.shape}"
            )
            raise ValueError(msg)

    return is_dataframe


# TODO: finish encoder docstrings


class Encoder(TransformerMixin, BaseEstimator):
    """A class that transforms an input ``pandas.DataFrame`` or ``numpy.ndarray`` into a
    Dataset class, corresponding to a column-wise binning of the original columns.

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

    When the input is a ``pandas.DataFrame``, we support the encoding of missing
    values both for categorical and numerical columns. However, when the input is a
    ``numpy.ndarray``, missing values are supported only for a numerical data type.
    Other situations might raise unexpected errors.

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

    is_categorical : None or ndarray
        blabla

    max_modalities : {"log", "sqrt"} or int, default="log"
        When a column's data type is numerical and ``is_categorical`` is None, WildWood
        decides that a column is categorical whenever the number of unique values it
        contains is smaller or equal to ``max_modalities``. Otherwise, it is considered
        numerical.

        - If "log", we set ``max_modalities=max(2, floor(log(n_samples)))``
        - If "sqrt", we set ``max_modalities=max(2, floor(sqrt(n_samples)))``

        Default is "log".

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
        max_modalities="log",
        handle_unknown="error",
        n_jobs=-1,
        random_state=None,
        verbose=False,
    ):
        # TODO: once fitted, most of these attributes must be readonly
        self.max_bins = max_bins
        self.subsample = subsample
        self.max_modalities = max_modalities

        # NB: this setattr calls a property that can change is_categorical
        self.is_categorical = is_categorical

        self.handle_unknown = handle_unknown
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # And initialize the attributes
        self.n_samples_in_ = None
        self.n_features_in_ = None
        self.categories_ = None
        self.binning_thresholds_ = None
        self.is_categorical_ = None
        self.n_bins_no_missing_values_ = None
        self.max_modalities_ = None

    # TODO: a faster fit transform where we avoid doing the same thing twice

    def fit(self, X, y=None):
        """This computes, for each column of X, a mapping from raw values to bins:
        categories to integers for categorical columns and inter-quantile intervals
        for continuous columns. It also infers if columns are categorical or
        continuous using the ``is_categorical`` attribute, and tries to guess it if not
        provided by the user.

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
        # TODO: test for constant columns
        # TODO: check that size of X matches that of is_categorical if not None here
        #  and in transform
        is_dataframe = check_X(X, is_categorical=self.is_categorical)

        n_samples, n_features = X.shape
        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features

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

        if isinstance(self.max_modalities, int):
            max_modalities_ = self.max_modalities
        elif self.max_modalities == "log":
            max_modalities_ = int(max(2, floor(log(n_samples))))
        else:
            max_modalities_ = int(max(2, floor(sqrt(n_samples))))

        self.max_modalities_ = max_modalities_

        if is_dataframe:
            # This array will contain the actual boolean mask corresponding to the
            # categorical features, which might differ from self.is_categorical
            is_categorical_ = np.zeros(n_features, dtype=np.bool_)

            for col_idx, (col_name, col) in enumerate(X.items()):
                if self.is_categorical is None:
                    is_col_categorical = _is_series_categorical(
                        col, is_categorical=None, max_modalities=max_modalities_
                    )

                else:
                    is_col_categorical = _is_series_categorical(
                        col,
                        is_categorical=self.is_categorical[col_idx],
                        max_modalities=max_modalities_,
                    )

                if is_col_categorical == "is_category":
                    # The column is already categorical. We save it as actually
                    # categorical and save its categories
                    is_categorical_[col_idx] = True
                    col_categories = col.cat.categories.values
                    categories[col_idx] = col_categories
                    # Number of bins is the number of categories (and do not
                    # include missing values)
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
                    # The number of bins used is the number of binning thresholds
                    # plus one
                    n_bins_no_missing_values[col_idx] = col_binning_thresholds.size + 1

        else:
            is_categorical_ = _get_array_dtypes(X, self.is_categorical, max_modalities_)
            # TODO: attention is on part du principe aussi que X ne peut contenir de
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

        # It's time now to instantiate the dataset
        dataset = Dataset(n_samples, max_values)

        # And to loop again over the columns to fill it
        if is_dataframe:
            for col_idx, (col_name, col) in enumerate(X_dict.items()):
                if is_categorical_[col_idx]:
                    col_categories_at_fit = categories[col_idx]
                    # Use the same categories as the ones used in fit and replace
                    # unknown categories by missing values.
                    col = col.cat.set_categories(col_categories_at_fit, inplace=False)

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
            is_dtype_numerical = X.dtype.kind in "uif"
            for col_idx in range(n_features):
                col = X_dict[col_idx]
                if is_categorical_[col_idx]:
                    col_categories_at_fit = categories[col_idx]
                    col = col.cat.set_categories(col_categories_at_fit, inplace=False)
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

                # Time to add the binned column in the dataset
                dataset_fill_column(dataset, col_idx, binned_col)

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
    def max_modalities(self):
        return self._max_modalities

    @max_modalities.setter
    def max_modalities(self, val):
        if isinstance(val, int):
            if val <= 3:
                raise ValueError(
                    f"max_modalities should be an int >= 3 or either "
                    f"'log' or 'sqrt'; {val} was given"
                )
            else:
                self._max_modalities = val
        elif isinstance(val, str):
            if val not in {"log", "sqrt"}:
                raise ValueError(
                    f"max_modalities should be an int >= 3 or either "
                    f"'log' or 'sqrt'; {val} was given"
                )
            else:
                self._max_modalities = val
