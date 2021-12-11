# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""This modules contains functions that helps to decide if columns from the feature
matrix ``X`` (either a ``pandas.DataFrame`` or ``numpy.ndarray``) are numerical or
categorical, and if a conversion is required, and a function that performs several
check on ``X``.
"""


import warnings
import numpy as np
import pandas as pd


def is_series_categorical(col, is_categorical, cat_min_categories, verbose):
    """Decides if a column (pandas.Series) contains categorical data or not and if it
    must be converted to the "category" data type. No actual data processing is applied
    in this function, it only helps to decides what will be done in the Encoder.
    The following set of rules is applied:

    - If ``col`` data type is ``object``, ``str``, ``unicode`` or ``bool``
        - If ``is_categorical`` is ``None``
            - If ``col`` data type is already ``category``: we do nothing
            - Else: we convert col to ``category`` and raise a warning
        - If ``is_categorical == True``
            - If ``col`` data type is already ``category``: we do nothing
            - Else: we convert col to ``category``
        - If ``is_categorical == False``
            - If ``col`` data type is already ``category``: we do nothing but raise a
              warning
            - Else: we convert col to categorical and raise a warning
    - If ``col`` data type is ``uint``, ``int`` or any ``float``
        - If ``is_categorical`` is ``None``
            - If the number of unique values in ``col`` is <= ``cat_min_categories``
              we convert ``col`` to ``category`` and raise a warning
            - Else: ``col`` is considered as numerical
        - If ``is_categorical == True``: we convert ``col`` to ``category``
        - If ``is_categorical == False``: we consider ``col`` as numerical
    - Else: we raise an error

    Parameters
    ----------
    col : pandas.Series
        The input column.

    is_categorical : None or bool
        True if the user specified the column as categorical.

    cat_min_categories : int
        When data type is numerical and the user gives no information (namely when
        ``is_categorical=None``), WildWood decides that the column is ``category``
        whenever the number of unique values in ``col`` is <= ``cat_min_categories``.
        Otherwise, it is considered numerical.

    verbose : bool, default=False
        If True, display warnings concerning columns typing.

    Returns
    -------
    output : {"is_category", "to_category", "numerical"}
        The inferred type of column. If "is_category" the column is already
        categorical and does not need to be converted. If "to_category", the column
        is categorical but needs to be transformed as a pandas.Series with "category"
        data type. If "numerical", the column is numerical and won't be transformed.
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
                if verbose:
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
                if verbose:
                    warnings.warn(
                        f"I will consider column {col.name} as categorical: it was "
                        f"declared non-categorical but has a categorical data type."
                    )
                return "is_category"
            else:
                # The data type corresponds to a categorical one, but the user
                # declared it as non-categorical. We must convert it to
                # categorical and raise a warning to the user
                if verbose:
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
            if n_modalities <= cat_min_categories:
                if verbose:
                    warnings.warn(
                        f"I will consider column {col.name} as categorical: it has "
                        f"{n_modalities} unique values."
                    )
                return "to_category"
            else:
                if verbose:
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


def get_array_dtypes(X, is_categorical, cat_min_categories, verbose):
    """Decides if each column of X contains categorical or numerical data. The
    following rules are applied:

    - If ``X`` data type is ``object``, ``str``, ``unicode`` or ``bool``
        - If ``is_categorical`` is ``None``
            - If the number of unique values in a column is <= ``cat_min_categories``
              we consider it categorical and raise a warning
            - Else: the column is considered numerical and we raise a warning
        - If ``is_categorical == True``
            - If column data type is already categorical: we do nothing
            - Else: we convert col to categorical
        - If ``is_categorical == False``
            - If column data type is already categorical: we do nothing but raise a
              warning
            - Else: we convert col to categorical and raise a warning
    - If ``X`` data type is ``uint``, ``int`` or any ``float``
        - If ``is_categorical`` is ``None``
            - If the number of unique values in a column is <= ``cat_min_categories``
              we convert the column to "category"" and raise a warning
            - Else: the column is considered as numerical
        - If ``is_categorical == True``: we convert the column to categorical
        - If ``is_categorical == False``: we consider the column as numerical
    - Else: we raise an error

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix of features of shape (n_samples, n_features).

    is_categorical : None or numpy.ndarray
        If not None, it is a numpy array of shape (n_features,) with boolean dtype,
        which corresponds to a categorical indicator for each column.

    cat_min_categories : int
        When data type is numerical and the user gives no information (namely when
        ``is_categorical=None``), WildWood decides that a column has dtype "category"
        whenever its number of unique values is <= ``cat_min_categories``.
        Otherwise, it is considered numerical.

    verbose : bool, default=False
        If True, display warnings concerning columns typing.

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
                if nunique <= cat_min_categories:
                    # The column looks categorical
                    is_categorical_out[col_idx] = True
                    if verbose:
                        warnings.warn(
                            f"I will consider column {col_idx} as categorical."
                        )
                else:
                    # It looks numerical. Let's see if it can be transformed to float
                    success = True
                    try:
                        col.astype(np.float32)
                    except ValueError:
                        success = False
                    if success:
                        is_categorical_out[col_idx] = False
                        if verbose:
                            warnings.warn(
                                f"I will consider column {col_idx} as numerical."
                            )
                    else:
                        is_categorical_out[col_idx] = True
                        if verbose:
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
                if nunique <= cat_min_categories:
                    # The column looks categorical
                    is_categorical_out[col_idx] = True
                    if verbose:
                        warnings.warn(
                            f"I will consider column {col_idx} as categorical."
                        )
                else:
                    # It looks numerical. So need to check if it can be converted,
                    # since it is "uif"
                    is_categorical_out[col_idx] = False
                    if verbose:
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
    """Checks if input is a 2D ``numpy.ndarray`` or a ``pandas.DataFrame`` and performs
    several sanity checks about it.

    - If X is a ``numpy.ndarray``, it checks that it is 2-dimensional and that its
      dtype is supported
    - If ``is_categorical`` is provided, it checks its shape with that of ``X``
    - If ``n_features`` is provided, it checks that ``X`` has ``n_features`` columns

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to fit and transform. It can be either a pandas dataframe or a 2D
        numpy array.

    is_categorical : None or numpy.ndarray
        If not None, it is a ``numpy.ndarray`` of shape (n_features,) with boolean
        dtype, which corresponds to a categorical indicator for each column.

    n_features : None or int
        The expected number of features. Allows to check that the number of features
        passed for prediction matches the one used for training.

    Returns
    -------
    output : bool
        True if input is a ``pandas.DataFrame``, False if it is a ``numpy.ndarray``.
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


def get_is_categorical(categorical_features, n_features):
    """Performs checks on categorical_features and converts it to a boolean
    ``numpy.ndarray``.

    Parameters
    ----------
    categorical_features : None or array-like
        Array-like containing boolean or integer values or shape (n_features,) or
        (n_categorical_features,) indicating the categorical features.
        - If **None** : no feature will be considered categorical.
        - If **boolean array-like** : boolean mask indicating categorical features.
        - If **integer array-like** : integer indices indicating categorical features.

    n_features : int
        Number of features (columns in X)

    Returns
    -------
    output : None or numpy.ndarray
        None whenever no feature is categorical or a numpy.ndarray of shape (
        n_features,) with dtype=bool that indicates whether each feature is categorical
        or not.
    """
    if categorical_features is None:
        return None
    else:
        categorical_features = np.asarray(categorical_features)
        if categorical_features.size == 0:
            return None
        else:
            kind = categorical_features.dtype.kind
            if kind == "i":
                if (
                    np.max(categorical_features) >= n_features
                    or np.min(categorical_features) < 0
                ):
                    raise ValueError(
                        "categorical_features set as integer "
                        "indices must be in [0, n_features - 1]"
                    )
                else:
                    is_categorical = np.zeros(n_features, dtype=bool)
                    is_categorical[categorical_features] = True
            elif kind == "b":
                if categorical_features.shape[0] != n_features:
                    raise ValueError(
                        "categorical_features set as a boolean mask "
                        "must have shape (n_features,), got: "
                        f"{categorical_features.shape}"
                    )
                else:
                    is_categorical = categorical_features.copy()
            else:
                raise ValueError(
                    "categorical_features must be an array-like of "
                    "bools or array-like of ints."
                )

            # We comment this since we raise when something weird is passed by
            # the user in the encoder
            # if not np.any(is_categorical):
            #     return None

            return is_categorical
