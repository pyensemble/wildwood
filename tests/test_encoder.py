# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module performs unittests for the Encoder class
"""
from math import floor, log, sqrt
import pandas as pd
import pytest
import numpy as np

from wildwood.preprocessing import Encoder, features_bitarray_to_array
from wildwood.preprocessing.encoder import is_series_categorical, get_array_dtypes

np.random.seed(42)


@pytest.mark.parametrize(
    "col",
    (
        pd.Series([0, 1, 2, 3], name="col"),
        pd.Series(["a", "b", "a", "c"], name="col"),
        pd.Series(["a", "b", "a", "c"], dtype="category", name="col"),
        pd.Series(np.random.randn(100), name="col"),
        pd.Series(["é", "à", "a", "ï", "c"], name="col"),
        pd.to_datetime(pd.Series(["2009-10-01", "2011-08-17"], name="col")),
    ),
)
@pytest.mark.parametrize("is_categorical", (True, False, None))
@pytest.mark.parametrize("cat_min_categories", (3,))
def test_is_series_categorical(col, is_categorical, cat_min_categories):
    dtype = col.dtype
    if dtype.kind in "bOSU":
        if is_categorical is None:
            if dtype.name == "category":
                output = "is_category"
                raises = None
                msg = None
            else:
                output = "to_category"
                raises = "warning"
                msg = f"I will consider column {col.name} as categorical."
        elif is_categorical:
            if dtype.name == "category":
                output = "is_category"
                raises = None
                msg = None
            else:
                output = "to_category"
                raises = None
                msg = None
        else:
            if dtype.name == "category":
                output = "is_category"
                raises = "warning"
                msg = (
                    f"I will consider column {col.name} as categorical: it was "
                    f"declared non-categorical but has a categorical data type."
                )
            else:
                output = "to_category"
                raises = "warning"
                msg = (
                    f"I will consider column {col.name} as categorical: it was "
                    f"declared non-categorical but has a categorical data type."
                )
    elif dtype.kind in "uif":
        if is_categorical is None:
            n_modalities = col.nunique()
            if n_modalities <= cat_min_categories:
                output = "to_category"
                raises = "warning"
                msg = (
                    f"I will consider column {col.name} as categorical: it has "
                    f"{n_modalities} unique values."
                )
            else:
                output = "numerical"
                raises = "warning"
                msg = (
                    f"I will consider column {col.name} as numerical: it has "
                    f"{n_modalities} unique values."
                )
        elif is_categorical:
            output = "to_category"
            raises = None
            msg = None
        else:
            output = "numerical"
            raises = None
            msg = None
    else:
        output = None
        raises = "error"
        msg = (
            f"Column {col.name} has dtype {col.dtype} which is not supported by "
            f"WildWood."
        )
    if raises is None:
        assert output == is_series_categorical(
            col, is_categorical, cat_min_categories, verbose=True
        )
    elif raises == "warning":
        with pytest.warns(UserWarning, match=msg):
            assert output == is_series_categorical(
                col, is_categorical, cat_min_categories, verbose=True
            )
    elif raises == "error":
        with pytest.raises(ValueError) as exc_info:
            assert output == is_series_categorical(
                col, is_categorical, cat_min_categories, verbose=True
            )
        assert exc_info.type is ValueError
        assert exc_info.value.args[0] == msg
    else:
        raise ValueError("Something is weird in this test")


@pytest.mark.parametrize(
    "X, is_categorical, cat_min_categories, out, error_type, msg",
    [
        (
            np.array([[0.1, "a"], [0.2, "b"], [0.4, "c"]]),
            np.array([False, True]),
            2,
            np.array([False, True]),
            None,
            None,
        ),
        (
            np.array([[0.1, "a"], [0.2, "b"], [0.4, "c"]]),
            np.array([False, True]),
            4,
            np.array([False, True]),
            None,
            None,
        ),
        (
            np.array([[0.1, "a"], [0.2, "b"], [0.4, "c"]]),
            None,
            2,
            np.array([False, True]),
            [UserWarning, UserWarning],
            [
                "I will consider column 0 as numerical.",
                "I will consider column 1 as categorical.",
            ],
        ),
        (
            np.array([[0.1, "a"], [0.2, "b"], [0.4, "c"]]),
            None,
            4,
            np.array([True, True]),
            [UserWarning, UserWarning],
            [
                "I will consider column 0 as categorical.",
                "I will consider column 1 as categorical.",
            ],
        ),
        (
            np.array([[0.1, "a"], [0.2, "b"], [0.4, "c"]]),
            np.array([False, False]),
            2,
            np.array([False, True]),
            ValueError,
            "could not convert string to float: 'a'",
        ),
        # (
        #         np.array([[0.1, "a"], [0.2, "b"], [0.4, "c"]]),
        #         np.array([False, True]),
        #         2,
        #         np.array([False, False]),
        # ),
    ],
)
def test_get_array_dtypes(X, is_categorical, cat_min_categories, out, error_type, msg):
    if error_type is None:
        output = get_array_dtypes(X, is_categorical, cat_min_categories, verbose=True)
        assert np.all(out == output)
    elif isinstance(error_type, ValueError):
        with pytest.raises(ValueError, match=msg):
            output = get_array_dtypes(
                X, is_categorical, cat_min_categories, verbose=True
            )
    elif isinstance(error_type, list):
        # In this case, it's the list of the warnings raised
        with pytest.warns(None) as records:
            output = get_array_dtypes(
                X, is_categorical, cat_min_categories, verbose=True
            )

        assert len(error_type) == len(records)
        for warning, warning_message in zip(records, msg):
            assert issubclass(warning.category, UserWarning)
            assert str(warning.message) == warning_message


@pytest.mark.filterwarnings("ignore:I will consider column")
def test_encoder_max_bins():
    encoder = Encoder()
    assert encoder.max_bins == 256
    encoder = Encoder(max_bins=127)
    assert encoder.max_bins == 127
    encoder.max_bins = 42
    assert encoder.max_bins == 42
    with pytest.raises(ValueError, match="max_bins must be an integer number"):
        encoder = Encoder(max_bins=3.14)
    with pytest.raises(ValueError, match="max_bins must be >= 3"):
        encoder.max_bins = 2


def test_encoder_handle_unknown():
    encoder = Encoder()
    assert encoder.handle_unknown == "error"
    encoder = Encoder(handle_unknown="consider_missing")
    assert encoder.handle_unknown == "consider_missing"
    encoder.handle_unknown = "error"
    assert encoder.handle_unknown == "error"

    msg = "handle_unknown must be 'error' or 'consider_missing' but " "got {0}".format(
        "truc"
    )
    with pytest.raises(ValueError, match=msg):
        encoder = Encoder(handle_unknown="truc")


def test_encoder_subsample():
    encoder = Encoder()
    assert encoder.subsample == int(2e5)
    encoder = Encoder(subsample=None)
    assert encoder.subsample is None
    encoder.subsample = 100_000.0
    assert encoder.subsample == 100_000

    msg = "subsample should be None or a number >= 50000"
    with pytest.raises(ValueError, match=msg):
        encoder = Encoder(subsample="truc")
    with pytest.raises(ValueError, match=msg):
        encoder = Encoder(subsample=10000)
    with pytest.raises(ValueError, match=msg):
        encoder = Encoder(subsample=-1)


@pytest.mark.filterwarnings("ignore:I will consider column")
@pytest.mark.parametrize(
    "cat_min_categories, msg, test_cat_min_categories_",
    [
        (None, None, False),
        ("log", None, True),
        ("sqrt", None, True),
        (17, None, True),
        (
            1,
            "cat_min_categories should be an int > 3 or either 'log' or 'sqrt'; 1 was given",
            False,
        ),
        (
            "oops",
            "cat_min_categories should be an int > 3 or either 'log' or 'sqrt'; oops was given",
            False,
        ),
    ],
)
def test_encoder_cat_min_categories(cat_min_categories, msg, test_cat_min_categories_):
    if cat_min_categories is None:
        encoder = Encoder()
        assert encoder.cat_min_categories == "log"
    else:
        if msg is None:
            encoder = Encoder(cat_min_categories=cat_min_categories)
            assert encoder.cat_min_categories == cat_min_categories
            assert encoder._cat_min_categories == cat_min_categories
            encoder = Encoder()
            encoder.cat_min_categories = cat_min_categories
            assert encoder.cat_min_categories == cat_min_categories
            assert encoder._cat_min_categories == cat_min_categories
        else:
            with pytest.raises(ValueError, match=msg):
                _ = Encoder(cat_min_categories=cat_min_categories)
            encoder = Encoder()
            with pytest.raises(ValueError, match=msg):
                encoder.cat_min_categories = cat_min_categories

    if test_cat_min_categories_:
        n_samples = 13
        df = pd.DataFrame({"col": np.random.randn(n_samples)})
        encoder = Encoder(cat_min_categories=cat_min_categories)
        encoder.fit(df)
        if cat_min_categories == "log":
            assert encoder.cat_min_categories_ == floor(log(n_samples))
        elif cat_min_categories == "sqrt":
            assert encoder.cat_min_categories_ == floor(sqrt(n_samples))
        elif isinstance(cat_min_categories, int):
            assert encoder.cat_min_categories_ == cat_min_categories

        n_samples = 2
        df = pd.DataFrame({"col": np.random.randn(n_samples)})
        encoder = Encoder(cat_min_categories=cat_min_categories)
        encoder.fit(df)
        if cat_min_categories == "log":
            assert encoder.cat_min_categories_ == 2
        elif cat_min_categories == "sqrt":
            assert encoder.cat_min_categories_ == 2
        elif isinstance(cat_min_categories, int):
            assert encoder.cat_min_categories_ == cat_min_categories


@pytest.mark.parametrize(
    "is_categorical, msg",
    [
        (None, None),
        (np.zeros(3, dtype=np.bool_), None),
        ([False, True, False], None),
        ((0, 1, 0), None),
        (np.zeros((2,), dtype=np.float64), None),
        (
            np.zeros((2, 2), dtype=np.bool_),
            "is_categorical must be either None or a "
            "one-dimensional array-like with bool-like dtype",
        ),
    ],
)
def test_is_categorical(is_categorical, msg):
    if is_categorical is None:
        encoder = Encoder()
        assert encoder.is_categorical is None
        assert encoder._is_categorical is None
        encoder = Encoder()
        encoder.is_categorical = None
        assert encoder.is_categorical is None
        assert encoder._is_categorical is None
    else:
        if msg is None:
            encoder = Encoder(is_categorical=is_categorical)
            assert np.all(encoder.is_categorical == is_categorical)
            assert np.all(encoder._is_categorical == is_categorical)
            assert isinstance(encoder._is_categorical, np.ndarray)
            assert encoder._is_categorical.dtype.name == "bool"

            encoder.is_categorical = is_categorical
            assert np.all(encoder.is_categorical == is_categorical)
            assert np.all(encoder._is_categorical == is_categorical)
            assert isinstance(encoder._is_categorical, np.ndarray)
            assert encoder._is_categorical.dtype.name == "bool"

        else:
            with pytest.raises(ValueError, match=msg):
                _ = Encoder(is_categorical=is_categorical)
            with pytest.raises(ValueError, match=msg):
                encoder = Encoder()
                encoder.is_categorical = is_categorical


def test_check_is_category_matches_X():
    encoder = Encoder(is_categorical=np.zeros(3, dtype=np.bool_))


def assert_frames_equal(df1, df2, **kwargs):
    # pandas does not support interval dtype comparison at a specified tolerance
    for ((colname1, col1), (colname2, col2)) in zip(df1.items(), df2.items()):
        assert colname1 == colname2
        if col1.dtype.name == "interval":
            col1_left = pd.Series([val if pd.isna(val) else val.left for val in col1])
            col2_left = pd.Series([val if pd.isna(val) else val.left for val in col2])
            pd.testing.assert_series_equal(col1_left, col2_left, **kwargs)
            col1_right = pd.Series([val if pd.isna(val) else val.right for val in col1])
            col2_right = pd.Series([val if pd.isna(val) else val.right for val in col2])
            pd.testing.assert_series_equal(col1_right, col2_right)
        else:
            pd.testing.assert_series_equal(col1, col2, **kwargs)


# TODO: we will need also to test against what is passed to is_categorical


def get_example1_pandas():
    """First example with no missing values, A and D are declared categoricals,
    while B is also categorical but not declared as such, one which is
    obviously categorical and two continuous
    """
    df = pd.DataFrame(
        {
            "A": [0, 1, 1, 1, 0, 0, 0, 0, 1],
            "B": ["b", "a", "b", "c", "a", "a", "b", "c", "b"],
            "C": [3, 3, 0, -1, 42, 7, 1, 17, 8],
            "D": ["b", "a", "b", "c", "a", "a", "d", "c", "a"],
            "E": [-4, 1, 2, 1, -3, 17, 2, 3.0, -1],
        },
    )
    for colname in ["A", "D"]:
        df[colname] = df[colname].astype("category")

    max_bins = 5
    n_bins_no_missing_values_ = np.array([2, 3, 5, 4, 5])
    categories_ = {
        0: np.array([0, 1]),
        1: np.array(["a", "b", "c"]),
        3: np.array(["a", "b", "c", "d"]),
    }
    binning_thresholds_ = {
        2: np.array([0.5, 3.0, 5.0, 12.5]),
        4: np.array([-2.0, 1.0, 1.5, 2.5]),
    }

    X_binned = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 3],
            [1, 2, 0, 2, 1],
            [0, 0, 4, 0, 0],
            [0, 0, 3, 0, 4],
            [0, 1, 1, 3, 3],
            [0, 2, 4, 2, 4],
            [1, 1, 3, 0, 1],
        ]
    )

    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.Categorical([0, 1, 1, 1, 0, 0, 0, 0, 1])
    df_inverse_transform[1] = pd.Categorical(
        ["b", "a", "b", "c", "a", "a", "b", "c", "b"]
    )
    df_inverse_transform[2] = pd.arrays.IntervalArray.from_tuples(
        [
            (0.5, 3.0),
            (0.5, 3.0),
            (-np.inf, 0.5),
            (-np.inf, 0.5),
            (12.5, np.inf),
            (5.0, 12.5),
            (0.5, 3.0),
            (12.5, np.inf),
            (5.0, 12.5),
        ]
    )
    df_inverse_transform[3] = pd.Categorical(
        ["b", "a", "b", "c", "a", "a", "d", "c", "a"]
    )
    df_inverse_transform[4] = pd.arrays.IntervalArray.from_tuples(
        [
            (-np.inf, -2.0),
            (-2.0, 1.0),
            (1.5, 2.5),
            (-2.0, 1.0),
            (-np.inf, -2.0),
            (2.5, np.inf),
            (1.5, 2.5),
            (2.5, np.inf),
            (-2.0, 1.0),
        ]
    )

    warnings = [
        "I will consider column B as categorical.",
        "I will consider column C as numerical: it has 8 unique values.",
        "I will consider column E as numerical: it has 7 unique values.",
    ]

    return (
        df,
        max_bins,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
        warnings,
    )


def get_example2_pandas():
    """Example with missing values both for continuous and categoricals. A is not
    specified as categorical"""
    df = pd.DataFrame(
        {
            "A": [None, "a", "b", None, "a", "a", "d", "c"],
            "B": [3, None, 0, -1, 42, 7, 1, None],
            "C": ["b", "a", "b", "c", "a", "a", "d", "c"],
            "D": [-4, 1, 2, 1, -3, None, 2, 3.0],
        },
    )
    for colname in ["B", "C"]:
        df[colname] = df[colname].astype("category")

    max_bins = 4
    n_bins_no_missing_values_ = np.array([4, 6, 4, 4])
    categories_ = {
        0: np.array(["a", "b", "c", "d"]),
        1: np.array([-1.0, 0.0, 1.0, 3.0, 7.0, 42.0]),
        2: np.array(["a", "b", "c", "d"]),
    }
    binning_thresholds_ = {3: np.array([-1.0, 1.0, 2.0])}

    X_binned = np.array(
        [
            [4, 3, 1, 0],
            [0, 6, 0, 1],
            [1, 1, 1, 2],
            [4, 0, 2, 1],
            [0, 5, 0, 0],
            [0, 4, 0, 4],
            [3, 2, 3, 2],
            [2, 6, 2, 3],
        ]
    )

    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.Categorical(
        [np.nan, "a", "b", np.nan, "a", "a", "d", "c"]
    )
    df_inverse_transform[1] = pd.Categorical(
        [3.0, np.nan, 0.0, -1.0, 42.0, 7.0, 1.0, np.nan,]
    )
    df_inverse_transform[2] = pd.Categorical(["b", "a", "b", "c", "a", "a", "d", "c"])
    df_inverse_transform[3] = pd.arrays.IntervalArray.from_tuples(
        [
            (-np.inf, -1.0),
            (-1.0, 1.0),
            (1.0, 2.0),
            (-1.0, 1.0),
            (-np.inf, -1.0),
            np.nan,
            (1.0, 2.0),
            (2.0, np.inf),
        ]
    )
    warnings = [
        "I will consider column A as categorical.",
        "I will consider column D as numerical: it has 5 unique values.",
    ]
    return (
        df,
        max_bins,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
        warnings,
    )


def get_example3_pandas():
    """Same as first example but with default (256) bins"""
    df = pd.DataFrame(
        {
            "A": [0, 1, 1, 1, 0, 0, 0, 0, 1],
            "B": ["b", "a", "b", "c", "a", "a", "b", "c", "b"],
            "C": [3, 3, 0, -1, 42, 7, 1, 17, 8],
            "D": ["b", "a", "b", "c", "a", "a", "d", "c", "a"],
            "E": [-4, 1, 2, 1, -3, 17, 2, 3.0, -1],
        },
    )

    for colname in ["A", "B", "D"]:
        df[colname] = df[colname].astype("category")

    max_bins = 256
    n_bins_no_missing_values_ = np.array([2, 3, 8, 4, 7])
    categories_ = {
        0: np.array([0, 1]),
        1: np.array(["a", "b", "c"]),
        3: np.array(["a", "b", "c", "d"]),
    }
    binning_thresholds_ = {
        2: np.array([-0.5, 0.5, 2.0, 5.0, 7.5, 12.5, 29.5]),
        4: np.array([-3.5, -2.0, 0.0, 1.5, 2.5, 10.0]),
    }

    X_binned = np.array(
        [
            [0, 1, 3, 1, 0],
            [1, 0, 3, 0, 3],
            [1, 1, 1, 1, 4],
            [1, 2, 0, 2, 3],
            [0, 0, 7, 0, 1],
            [0, 0, 4, 0, 6],
            [0, 1, 2, 3, 4],
            [0, 2, 6, 2, 5],
            [1, 1, 5, 0, 2],
        ]
    )

    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.Categorical([0, 1, 1, 1, 0, 0, 0, 0, 1])
    df_inverse_transform[1] = pd.Categorical(
        ["b", "a", "b", "c", "a", "a", "b", "c", "b"]
    )
    df_inverse_transform[2] = pd.arrays.IntervalArray.from_tuples(
        [
            (2.0, 5.0),
            (2.0, 5.0),
            (-0.5, 0.5),
            (-np.inf, -0.5),
            (29.5, np.inf),
            (5.0, 7.5),
            (0.5, 2.0),
            (12.5, 29.5),
            (7.5, 12.5),
        ]
    )
    df_inverse_transform[3] = pd.Categorical(
        ["b", "a", "b", "c", "a", "a", "d", "c", "a"]
    )
    df_inverse_transform[4] = pd.arrays.IntervalArray.from_tuples(
        [
            (-np.inf, -3.5),
            (0.0, 1.5),
            (1.5, 2.5),
            (0.0, 1.5),
            (-3.5, -2.0),
            (10.0, np.inf),
            (1.5, 2.5),
            (2.5, 10.0),
            (-2.0, 0.0),
        ]
    )
    warnings = [
        "I will consider column C as numerical: it has 8 unique values.",
        "I will consider column E as numerical: it has 7 unique values.",
    ]
    return (
        df,
        max_bins,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
        warnings,
    )


def get_example4_pandas():
    """The same with example 2 but with the default max_bins (256)"""
    df = pd.DataFrame(
        {
            "A": [None, "a", "b", None, "a", "a", "d", "c"],
            "B": [3, None, 0, -1, 42, 7, 1, None],
            "C": ["b", "a", "b", "c", "a", "a", "d", "c"],
            "D": [-4, 1, 2, 1, -3, None, 2, 3.0],
        },
    )
    for colname in ["A", "B", "C"]:
        df[colname] = df[colname].astype("category")

    max_bins = 256
    n_bins_no_missing_values_ = np.array([4, 6, 4, 5])
    categories_ = {
        0: np.array(["a", "b", "c", "d"]),
        1: np.array([-1.0, 0.0, 1.0, 3.0, 7.0, 42.0]),
        2: np.array(["a", "b", "c", "d"]),
    }
    binning_thresholds_ = {3: np.array([-3.5, -1, 1.5, 2.5])}

    X_binned = np.array(
        [
            [4, 3, 1, 0],
            [0, 6, 0, 2],
            [1, 1, 1, 3],
            [4, 0, 2, 2],
            [0, 5, 0, 1],
            [0, 4, 0, 5],
            [3, 2, 3, 3],
            [2, 6, 2, 4],
        ]
    )

    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.Categorical(
        [np.nan, "a", "b", np.nan, "a", "a", "d", "c"]
    )
    df_inverse_transform[1] = pd.Categorical(
        [3.0, np.nan, 0.0, -1.0, 42.0, 7.0, 1.0, np.nan]
    )
    df_inverse_transform[2] = pd.Categorical(["b", "a", "b", "c", "a", "a", "d", "c"])
    df_inverse_transform[3] = pd.arrays.IntervalArray.from_tuples(
        [
            (-np.inf, -3.5),
            (-1.0, 1.5),
            (1.5, 2.5),
            (-1.0, 1.5),
            (-3.5, -1.0),
            np.nan,
            (1.5, 2.5),
            (2.5, np.inf),
        ]
    )
    warnings = ["I will consider column D as numerical: it has 5 unique values."]
    return (
        df,
        max_bins,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
        warnings,
    )


def get_example5_pandas():
    # TODO: a purely numerical array with no missing values but is_categorical and a
    #  dependence on max_modalities
    pass


def get_example1_ndarray():
    """An example with a purely numerical ndarray"""
    random_state = 42
    n_samples = 17
    n_features = 3
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    max_bins = 5
    is_categorical = None
    n_bins_no_missing_values_ = np.array([5, 5, 5])
    categories_ = {}
    binning_thresholds_ = {
        0: np.array([-0.58199707, -0.17071229, 0.22541293, 0.43620609]),
        1: np.array([-1.21256741, -0.46202823, -0.18620884, 0.42469458]),
        2: np.array([-1.27451485, -0.46760207, -0.12381709, 0.31916565]),
    }
    X_binned = np.array(
        [
            [4, 3, 4],
            [4, 2, 2],
            [4, 4, 1],
            [4, 1, 2],
            [3, 0, 0],
            [1, 1, 3],
            [0, 0, 4],
            [1, 3, 0],
            [1, 3, 1],
            [3, 1, 2],
            [0, 4, 3],
            [0, 4, 1],
            [2, 0, 0],
            [2, 4, 3],
            [2, 2, 0],
            [0, 2, 4],
            [3, 0, 4],
        ]
    )
    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.arrays.IntervalArray.from_tuples(
        [
            (0.4362060856784523, np.inf),
            (0.4362060856784523, np.inf),
            (0.4362060856784523, np.inf),
            (0.4362060856784523, np.inf),
            (0.22541293328539475, 0.4362060856784523),
            (-0.5819970707351848, -0.1707122914373881),
            (-np.inf, -0.5819970707351848),
            (-0.5819970707351848, -0.1707122914373881),
            (-0.5819970707351848, -0.1707122914373881),
            (0.22541293328539475, 0.4362060856784523),
            (-np.inf, -0.5819970707351848),
            (-np.inf, -0.5819970707351848),
            (-0.1707122914373881, 0.22541293328539475),
            (-0.1707122914373881, 0.22541293328539475),
            (-0.1707122914373881, 0.22541293328539475),
            (-np.inf, -0.5819970707351848),
            (0.22541293328539475, 0.4362060856784523),
        ]
    )
    df_inverse_transform[1] = pd.arrays.IntervalArray.from_tuples(
        [
            (-0.18620883794726031, 0.4246945848526383),
            (-0.46202823188612485, -0.18620883794726031),
            (0.4246945848526383, np.inf),
            (-1.2125674108348576, -0.46202823188612485),
            (-np.inf, -1.2125674108348576),
            (-1.2125674108348576, -0.46202823188612485),
            (-np.inf, -1.2125674108348576),
            (-0.18620883794726031, 0.4246945848526383),
            (-0.18620883794726031, 0.4246945848526383),
            (-1.2125674108348576, -0.46202823188612485),
            (0.4246945848526383, np.inf),
            (0.4246945848526383, np.inf),
            (-np.inf, -1.2125674108348576),
            (0.4246945848526383, np.inf),
            (-0.46202823188612485, -0.18620883794726031),
            (-0.46202823188612485, -0.18620883794726031),
            (-np.inf, -1.2125674108348576),
        ]
    )
    df_inverse_transform[2] = pd.arrays.IntervalArray.from_tuples(
        [
            (0.31916565099503447, np.inf),
            (-0.4676020697526045, -0.12381709084355724),
            (-1.2745148494347265, -0.4676020697526045),
            (-0.4676020697526045, -0.12381709084355724),
            (-np.inf, -1.2745148494347265),
            (-0.12381709084355724, 0.31916565099503447),
            (0.31916565099503447, np.inf),
            (-np.inf, -1.2745148494347265),
            (-1.2745148494347265, -0.4676020697526045),
            (-0.4676020697526045, -0.12381709084355724),
            (-0.12381709084355724, 0.31916565099503447),
            (-1.2745148494347265, -0.4676020697526045),
            (-np.inf, -1.2745148494347265),
            (-0.12381709084355724, 0.31916565099503447),
            (-np.inf, -1.2745148494347265),
            (0.31916565099503447, np.inf),
            (0.31916565099503447, np.inf),
        ]
    )
    warnings = [
        "I will consider column 0 as numerical.",
        "I will consider column 1 as numerical.",
        "I will consider column 2 as numerical.",
    ]

    return (
        X,
        max_bins,
        is_categorical,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
        warnings,
    )


@pytest.mark.filterwarnings("ignore:I will consider column")
@pytest.mark.parametrize(
    "df, max_bins, n_bins_no_missing_values_, categories_, binning_thresholds_, "
    "X_binned, df_inverse_transform, warnings",
    [
        get_example1_pandas(),
        get_example2_pandas(),
        get_example3_pandas(),
        get_example4_pandas(),
    ],
)
def test_encoder_fit_transform_dataframes(
    df,
    max_bins,
    n_bins_no_missing_values_,
    categories_,
    binning_thresholds_,
    X_binned,
    df_inverse_transform,
    warnings,
):
    encoder = Encoder(max_bins=max_bins, verbose=True)
    if warnings is not None:
        with pytest.warns(UserWarning) as warn_records:
            encoder.fit(df)
        assert len(warn_records) == len(warnings)
        for warn_record, warning in zip(warn_records, warnings):
            assert str(warn_record.message) == warning
    else:
        encoder.fit(df)

    assert encoder.n_samples_in_ == df.shape[0]
    assert encoder.n_features_in_ == df.shape[1]
    np.testing.assert_array_equal(
        encoder.n_bins_no_missing_values_, n_bins_no_missing_values_
    )
    # Check that categories are OK
    assert encoder.categories_.keys() == categories_.keys()
    for (categories1, categories2) in zip(
        encoder.categories_.values(), categories_.values()
    ):
        np.testing.assert_array_equal(categories1, categories2)
    # Check that binning thresholds are OK
    assert encoder.binning_thresholds_.keys() == binning_thresholds_.keys()
    for (thresholds1, thresholds2) in zip(
        encoder.binning_thresholds_.values(), binning_thresholds_.values()
    ):
        np.testing.assert_array_equal(thresholds1, thresholds2)

    # Check that dataset is correct
    dataset = encoder.transform(df)
    X_binned_out = features_bitarray_to_array(dataset)
    np.testing.assert_array_equal(X_binned_out, X_binned)

    # Check that reconstructed dataframe is correct
    df_inverse_transform_out = encoder.inverse_transform(dataset)
    pd.testing.assert_frame_equal(df_inverse_transform_out, df_inverse_transform)

    # Test also fit_transform
    encoder = Encoder(max_bins=max_bins)
    dataset = encoder.fit_transform(df)
    df_inverse_transform_out = encoder.inverse_transform(dataset)
    pd.testing.assert_frame_equal(df_inverse_transform_out, df_inverse_transform)


def get_example2_ndarray():
    """An example with a purely numerical ndarray and missing values"""
    random_state = 42
    n_samples = 9
    n_features = 3
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    X[0, 0] = np.nan
    X[4, 2] = np.nan
    X[4, 2] = np.nan
    max_bins = 4
    is_categorical = None
    n_bins_no_missing_values_ = np.array([4, 4, 4])
    categories_ = {}
    binning_thresholds_ = {
        0: np.array([-0.55333513, 0.00809299, 1.03279495]),
        1: np.array([-1.01283112, -0.23415337, 0.0675282]),
        2: np.array([-0.81023398, -0.34993336, 0.48096794]),
    }
    X_binned = np.array(
        [
            [4, 2, 3],
            [3, 1, 2],
            [3, 3, 1],
            [2, 1, 1],
            [2, 0, 4],
            [0, 0, 2],
            [0, 0, 3],
            [1, 2, 0],
            [1, 3, 0],
        ]
    )
    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.arrays.IntervalArray.from_tuples(
        [
            np.nan,
            (1.032794949996995, np.inf),
            (1.032794949996995, np.inf),
            (0.008092985539749242, 1.032794949996995),
            (0.008092985539749242, 1.032794949996995),
            (-np.inf, -0.5533351268830777),
            (-np.inf, -0.5533351268830777),
            (-0.5533351268830777, 0.008092985539749242),
            (-0.5533351268830777, 0.008092985539749242),
        ]
    )
    df_inverse_transform[1] = pd.arrays.IntervalArray.from_tuples(
        [
            (-0.23415337472333597, 0.06752820468792384),
            (-1.0128311203344238, -0.23415337472333597),
            (0.06752820468792384, np.inf),
            (-1.0128311203344238, -0.23415337472333597),
            (-np.inf, -1.0128311203344238),
            (-np.inf, -1.0128311203344238),
            (-np.inf, -1.0128311203344238),
            (-0.23415337472333597, 0.06752820468792384),
            (0.06752820468792384, np.inf),
        ]
    )
    df_inverse_transform[2] = pd.arrays.IntervalArray.from_tuples(
        [
            (0.4809679353479832, np.inf),
            (-0.3499333552597187, 0.4809679353479832),
            (-0.8102339816786275, -0.3499333552597187),
            (-0.8102339816786275, -0.3499333552597187),
            np.nan,
            (-0.3499333552597187, 0.4809679353479832),
            (0.4809679353479832, np.inf),
            (-np.inf, -0.8102339816786275),
            (-np.inf, -0.8102339816786275),
        ]
    )

    warning = [
        "I will consider column 0 as numerical.",
        "I will consider column 1 as numerical.",
        "I will consider column 2 as numerical.",
    ]

    return (
        X,
        max_bins,
        is_categorical,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
        warning,
    )


def get_example3_ndarray():
    """An example with a purely numerical ndarray and missing values and
    is_categorical saying that some of these are categorical (including one with
    missing values)
    """
    random_state = 42
    n_samples = 9
    n_features = 3
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    X[0, 0] = np.nan
    X[4, 2] = np.nan
    X[4, 2] = np.nan
    max_bins = 4
    is_categorical = np.array([True, False, False])

    n_bins_no_missing_values_ = np.array([8, 4, 4])
    categories_ = {
        0: np.array(
            [
                -0.90802408,
                -0.56228753,
                -0.54438272,
                -0.2257763,
                0.24196227,
                0.54256004,
                1.52302986,
                1.57921282,
            ]
        )
    }
    binning_thresholds_ = {
        1: np.array([-1.01283112, -0.23415337, 0.0675282]),
        2: np.array([-0.81023398, -0.34993336, 0.48096794]),
    }

    X_binned = np.array(
        [
            [8, 2, 3],
            [6, 1, 2],
            [7, 3, 1],
            [5, 1, 1],
            [4, 0, 4],
            [1, 0, 2],
            [0, 0, 3],
            [3, 2, 0],
            [2, 3, 0],
        ]
    )

    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.Categorical(
        [
            np.nan,
            1.523030,
            1.579213,
            0.542560,
            0.241962,
            -0.562288,
            -0.908024,
            -0.225776,
            -0.544383,
        ]
    )
    df_inverse_transform[1] = pd.arrays.IntervalArray.from_tuples(
        [
            (-0.23415337472333597, 0.06752820468792384),
            (-1.0128311203344238, -0.23415337472333597),
            (0.06752820468792384, np.inf),
            (-1.0128311203344238, -0.23415337472333597),
            (-np.inf, -1.0128311203344238),
            (-np.inf, -1.0128311203344238),
            (-np.inf, -1.0128311203344238),
            (-0.23415337472333597, 0.06752820468792384),
            (0.06752820468792384, np.inf),
        ]
    )
    df_inverse_transform[2] = pd.arrays.IntervalArray.from_tuples(
        [
            (0.4809679353479832, np.inf),
            (-0.3499333552597187, 0.4809679353479832),
            (-0.8102339816786275, -0.3499333552597187),
            (-0.8102339816786275, -0.3499333552597187),
            np.nan,
            (-0.3499333552597187, 0.4809679353479832),
            (0.4809679353479832, np.inf),
            (-np.inf, -0.8102339816786275),
            (-np.inf, -0.8102339816786275),
        ]
    )
    warning = None

    return (
        X,
        max_bins,
        is_categorical,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
        warning,
    )


def get_example4_ndarray():
    """An example with object dtype and no missing values, with is_category = None."""
    X = np.array(
        [
            ["a", 0.1],
            ["a", 0.2],
            ["b", 0.1],
            ["a", 0.3],
            ["a", 0.0],
            ["c", -1.2],
            ["b", 0.1],
        ]
    )
    max_bins = 4
    is_categorical = None
    n_bins_no_missing_values_ = np.array([3, 4])
    categories_ = {0: np.array(["a", "b", "c"])}
    binning_thresholds_ = {1: np.array([0.05, 0.1, 0.15])}
    X_binned = np.array([[0, 2], [0, 3], [1, 2], [0, 3], [0, 0], [2, 0], [1, 2]])
    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.Categorical(["a", "a", "b", "a", "a", "c", "b"])
    df_inverse_transform[1] = pd.arrays.IntervalArray.from_tuples(
        [
            (0.1, 0.15000000000000002),
            (0.15000000000000002, np.inf),
            (0.1, 0.15000000000000002),
            (0.15000000000000002, np.inf),
            (-np.inf, 0.05),
            (-np.inf, 0.05),
            (0.1, 0.15000000000000002),
        ]
    )
    warning = [
        "I will consider column 0 as categorical.",
        "I will consider column 1 as numerical.",
    ]
    return (
        X,
        max_bins,
        is_categorical,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
        warning,
    )


def get_example5_ndarray():
    """Same as 4 but with is_categorical"""
    pass


# TODO: faudrait capturer les warnings (qui sont normaux) et les verifier...


@pytest.mark.filterwarnings("ignore:I will consider column")
@pytest.mark.parametrize(
    "X, max_bins, is_categorical, n_bins_no_missing_values_, categories_, "
    "binning_thresholds_, "
    "X_binned, df_inverse_transform, warnings",
    [
        get_example1_ndarray(),
        get_example2_ndarray(),
        get_example3_ndarray(),
        get_example4_ndarray(),
    ],
)
def test_encoder_fit_transform_ndarray(
    X,
    max_bins,
    is_categorical,
    n_bins_no_missing_values_,
    categories_,
    binning_thresholds_,
    X_binned,
    df_inverse_transform,
    warnings,
):
    encoder = Encoder(max_bins=max_bins, is_categorical=is_categorical, verbose=True)
    if warnings is not None:
        with pytest.warns(UserWarning) as warn_records:
            encoder.fit(X)
        assert len(warn_records) == len(warnings)
        for warn_record, warning in zip(warn_records, warnings):
            assert str(warn_record.message) == warning
    else:
        encoder.fit(X)
    assert encoder.n_samples_in_ == X.shape[0]
    assert encoder.n_features_in_ == X.shape[1]
    np.testing.assert_array_equal(
        encoder.n_bins_no_missing_values_, n_bins_no_missing_values_
    )
    # Check that categories are OK
    assert encoder.categories_.keys() == categories_.keys()
    for (categories1, categories2) in zip(
        encoder.categories_.values(), categories_.values()
    ):
        if isinstance(categories2, np.ndarray):
            if categories2.dtype.kind in "uif":
                assert categories1 == pytest.approx(categories2, rel=1e-8, abs=1e-8)
            else:
                np.testing.assert_array_equal(categories1, categories2)
        else:
            np.testing.assert_array_equal(categories1, categories2)
    # Check that binning thresholds are OK
    assert encoder.binning_thresholds_.keys() == binning_thresholds_.keys()
    for (thresholds1, thresholds2) in zip(
        encoder.binning_thresholds_.values(), binning_thresholds_.values()
    ):
        assert thresholds1 == pytest.approx(thresholds2, rel=1e-8, abs=1e-8)

    # Check that dataset is correct
    dataset = encoder.transform(X)
    X_binned_out = features_bitarray_to_array(dataset)
    np.testing.assert_array_equal(X_binned_out, X_binned)

    # If we have categorical dtype with float values, exact comparison of dtype will
    # fail.
    has_float_categorical = False
    for _, col in df_inverse_transform.items():
        if col.dtype.name == "category":
            if col.dtype.categories.values.dtype.kind == "f":
                has_float_categorical = True

    # Check that reconstructed dataframe is correct
    df_inverse_transform_out = encoder.inverse_transform(dataset)
    if has_float_categorical:
        assert_frames_equal(
            df_inverse_transform_out,
            df_inverse_transform,
            check_categorical=False,
            check_exact=False,
            atol=1e-7,
            rtol=1e-3,
        )
    else:
        assert_frames_equal(
            df_inverse_transform_out,
            df_inverse_transform,
            check_exact=False,
            atol=1e-7,
            rtol=1e-3,
        )

    # # Test also fit_transform
    encoder = Encoder(max_bins=max_bins, is_categorical=is_categorical)
    dataset = encoder.fit_transform(X)
    df_inverse_transform_out = encoder.inverse_transform(dataset)
    if has_float_categorical:
        pd.testing.assert_frame_equal(
            df_inverse_transform_out, df_inverse_transform, check_categorical=False
        )
    else:
        assert_frames_equal(
            df_inverse_transform_out,
            df_inverse_transform,
            check_exact=False,
            atol=1e-7,
            rtol=1e-3,
        )


@pytest.mark.filterwarnings("ignore:I will consider column")
def test_encoder_errors():
    # unsupported dtype in DataFrame raises an error
    df = pd.DataFrame({"col": pd.to_datetime(["2011-10-01", "2009-08-17"])})
    with pytest.raises(ValueError) as err_info:
        _ = Encoder().fit(df)
    assert err_info.type is ValueError
    assert (
        err_info.value.args[0]
        == "Column col has dtype datetime64[ns] which is not supported by WildWood."
    )

    # object dtype cannot be converted to float when column is declared numerical
    X = np.array(
        [
            ["a", 0.1],
            ["a", 0.2],
            ["b", 0.1],
            ["a", 0.3],
            ["a", 0.0],
            ["c", -1.2],
            ["b", 0.1],
        ]
    )
    is_categorical = [False, False]
    with pytest.raises(
        ValueError,
        match=f"Column 0 is declared as numerical, but it cannot be converted to float",
    ):
        _ = Encoder(is_categorical=is_categorical).fit(X)

    # non-2d ndarray are not supported
    X = np.random.randn(3, 2, 3)
    with pytest.raises(
        ValueError,
        match="X is must be a `pandas.DataFrame` or a two-dimensional `numpy.ndarray`.",
    ):
        _ = Encoder().fit(X)

    # ndarray with weird dtype are not supported
    X = np.empty((3, 2), dtype=np.dtype([("a", np.float64)]))
    with pytest.raises(ValueError) as err_info:
        _ = Encoder().fit(X)
    assert err_info.type is ValueError
    assert (
        err_info.value.args[0]
        == "The dtype of X [('a', '<f8')] is not supported by WildWood"
    )

    # the number of features cannot change between fit and transform
    X1 = np.random.randn(9, 4)
    X2 = np.random.randn(9, 3)
    encoder = Encoder().fit(X1)
    with pytest.raises(ValueError) as err_info:
        _ = encoder.transform(X2)
    assert err_info.type is ValueError
    assert (
        err_info.value.args[0] == "The number of features in X is different from the "
        "number of features of the fitted data. The fitted "
        "data had 4 features and the X has 3 features."
    )

    # the number of features must match the size of is_categorical
    is_categorical = [True, False]
    X = np.random.randn(7, 3)
    with pytest.raises(ValueError) as err_info:
        _ = Encoder(is_categorical=is_categorical).fit(X)
    assert err_info.type is ValueError
    assert (
        err_info.value.args[0]
        == "The number of features in X differs from the size of is_categorical. X has "
        "shape (7, 3) while is_categorical has shape (2,)"
    )


@pytest.mark.filterwarnings("ignore:I will consider column")
def test_encoder_detects_unknowns():
    df = pd.DataFrame(
        {
            "A": [0, 1, 1, 1, 0, 0, 0, 0, 1],
            "B": ["b", "a", "b", "c", "a", "a", "b", "c", "b"],
            "C": [3, 3, 0, -1, 42, 7, 1, 17, 8],
            "D": ["b", "a", "b", "c", "a", "a", "d", "c", "a"],
            "E": [-4, 1, 2, 1, -3, 17, 2, 3.0, -1],
        },
    )
    # Same dataframe as df but with less data and less modalities / unknown modalities
    df2 = pd.DataFrame(
        {
            "A": [0, 1, 1, 1, 0, 0],
            # Same column as df["B"] but with less modalities
            "B": ["b", "b", "c", "c", "b", "b"],
            # Same column as df["C"] but with a missing value
            "C": [3, 3, 0, -1, None, 7],
            # Same column with an unknown modality and missing ones
            "D": ["c", "b", "f", "c", "d", "f"],
            "E": [-1, 4, -67, 128, 2, 0],
        },
    )

    for colname in ["A", "B", "D"]:
        df[colname] = df[colname].astype("category")
        df2[colname] = df2[colname].astype("category")

    max_bins = 4
    encoder = Encoder(max_bins=max_bins)
    encoder.fit(df)

    msg = "Found unknown categories {0} in column {1} during " "transform".format(
        {"f"}, 3
    )
    with pytest.raises(ValueError, match=msg):
        dataset = encoder.transform(df2)

    max_bins = 4
    encoder = Encoder(max_bins=max_bins)
    encoder.fit(df)


# TODO: test that encoder detects if categorical columns change


@pytest.mark.filterwarnings("ignore:I will consider column")
def test_encoder_deals_with_unknown():

    df = pd.DataFrame(
        {
            "A": [0, 1, 1, 1, 0, 0, 0, 0, 1],
            "B": ["b", "a", "b", "c", "a", "a", "b", "c", "b"],
            "C": [3, 3, 0, -1, 42, 7, 1, 17, 8],
            "D": ["b", "a", "b", "c", "a", "a", "d", "c", "a"],
            "E": [-4, 1, 2, 1, -3, 17, 2, 3.0, -1],
        },
    )

    # Same dataframe as df but with less data and less modalities / unknown modalities
    df2 = pd.DataFrame(
        {
            "A": [0, 1, 1, 1, 0, 0],
            # Same column as df["B"] but with less modalities
            "B": ["b", "b", "c", "c", "b", "b"],
            # Same column as df["C"] but with a missing value
            "C": [3, 3, 0, -1, None, 7],
            # Same column with an unknown modality and missing ones
            "D": ["c", "b", "f", "c", "d", "f"],
            "E": [-1, 4, -67, 128, 2, 0],
        },
    )

    for colname in ["A", "B", "D"]:
        df[colname] = df[colname].astype("category")
        df2[colname] = df2[colname].astype("category")

    max_bins = 4
    encoder = Encoder(max_bins=max_bins, handle_unknown="consider_missing")

    encoder.fit(df)
    assert encoder.n_samples_in_ == df.shape[0]
    assert encoder.n_features_in_ == df.shape[1]

    n_bins_no_missing_values_ = np.array([2, 3, 4, 4, 4])
    np.testing.assert_array_equal(
        encoder.n_bins_no_missing_values_, n_bins_no_missing_values_
    )


@pytest.mark.filterwarnings("ignore:I will consider column")
def test_encoder_deals_with_unknown_again():
    df = pd.DataFrame(
        {
            "A": [0, 1, 1, 1, 0, 0, 0, 0, 1],
            "B": ["b", "a", "b", "c", "a", "a", "b", "c", "b"],
            "C": [3, 3, 0, -1, 42, 7, 1, 17, 8],
            "D": ["b", "a", "b", "c", "a", "a", "d", "c", "a"],
            "E": [-4, 1, 2, 1, -3, 17, 2, 3.0, -1],
        },
    )
    # Same dataframe as df but with less data and less modalities / unknown modalities
    df2 = pd.DataFrame(
        {
            "A": [0, 1, 1, 1, None, 0],
            "B": ["b", "b", "c", "c", "b", "b"],
            "C": [4, 2, -4, -1, None, 7],
            "D": ["c", "b", "f", "c", None, "f"],
            "E": [-1, 4, -67, 128, 2, 0],
        },
    )
    for colname in ["A", "B", "D"]:
        df[colname] = df[colname].astype("category")
        df2[colname] = df2[colname].astype("category")

    max_bins = 4
    encoder = Encoder(max_bins=max_bins, handle_unknown="consider_missing")

    encoder.fit(df)
    assert encoder.n_samples_in_ == df.shape[0]
    assert encoder.n_features_in_ == df.shape[1]

    n_bins_no_missing_values_ = np.array([2, 3, 4, 4, 4])
    np.testing.assert_array_equal(
        encoder.n_bins_no_missing_values_, n_bins_no_missing_values_
    )

    categories_ = {
        0: np.array([0, 1]),
        1: np.array(["a", "b", "c"]),
        3: np.array(["a", "b", "c", "d"]),
    }
    assert encoder.categories_.keys() == categories_.keys()
    for (categories1, categories2) in zip(
        encoder.categories_.values(), categories_.values()
    ):
        np.testing.assert_array_equal(categories1, categories2)

    binning_thresholds_ = {2: np.array([1.0, 3.0, 8.0]), 4: np.array([-1.0, 1.0, 2.0])}
    assert encoder.binning_thresholds_.keys() == binning_thresholds_.keys()
    for (thresholds1, thresholds2) in zip(
        encoder.binning_thresholds_.values(), binning_thresholds_.values()
    ):
        np.testing.assert_array_equal(thresholds1, thresholds2)

    is_categorical_ = np.array([True, True, False, True, False])
    np.testing.assert_array_equal(encoder.is_categorical_, is_categorical_)

    dataset = encoder.transform(df2)

    X_binned = np.array(
        [
            [0, 1, 2, 2, 0],
            [1, 1, 1, 1, 3],
            [1, 2, 0, 4, 0],
            [1, 2, 0, 2, 3],
            [2, 1, 4, 4, 2],
            [0, 1, 2, 4, 1],
        ]
    )
    X_binned_out = features_bitarray_to_array(dataset)
    np.testing.assert_array_equal(X_binned_out, X_binned)

    df_inverse_transform = pd.DataFrame()
    df_inverse_transform[0] = pd.Categorical([0, 1, 1, 1, None, 0])
    df_inverse_transform[1] = pd.Categorical(["b", "b", "c", "c", "b", "b"])
    df_inverse_transform[2] = pd.arrays.IntervalArray.from_tuples(
        [(3.0, 8.0), (1.0, 3.0), (-np.inf, 1.0), (-np.inf, 1.0), np.nan, (3.0, 8.0)]
    )
    df_inverse_transform[3] = pd.Categorical(
        ["c", "b", np.nan, "c", np.nan, np.nan], categories=["a", "b", "c"]
    )
    df_inverse_transform[4] = pd.arrays.IntervalArray.from_tuples(
        [
            (-np.inf, -1.0),
            (2.0, np.inf),
            (-np.inf, -1.0),
            (2.0, np.inf),
            (1.0, 2.0),
            (-1.0, 1.0),
        ]
    )
    df_inverse_transform_out = encoder.inverse_transform(dataset)
    pd.testing.assert_frame_equal(df_inverse_transform_out, df_inverse_transform)


@pytest.mark.parametrize("n_samples", [10, 32, 1000, 1_000_000])
@pytest.mark.parametrize(
    "max_values, dtype",
    [
        (np.array([2], dtype=np.uint64), np.uint8),
        (np.array([32], dtype=np.uint64), np.uint8),
        (np.array([17, 2, 64], dtype=np.uint64), np.uint8),
        (np.array([1024, 32, 64], dtype=np.uint64), np.uint16),
        (np.array([123765, 2, 16], dtype=np.uint64), np.uint32),
        (np.array([12323765, 123765, 2, 1024], dtype=np.uint64), np.uint64),
        (np.array([2378, 2, 213, 123765, 7, 64, 1024, 3], dtype=np.uint64), np.uint64),
    ],
)
def test_encoder_large_all_categorical(n_samples, max_values, dtype):
    n_features = max_values.size
    X_in = np.asfortranarray(
        np.random.randint(max_values + 1, size=(n_samples, n_features)), dtype=dtype
    )
    df_in = pd.DataFrame(X_in).astype("category")
    encoder = Encoder()
    dataset_out = encoder.fit_transform(df_in)
    df_out = encoder.inverse_transform(dataset_out)
    assert df_in.equals(df_out)
