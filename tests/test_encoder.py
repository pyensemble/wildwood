# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module performs unittests for the dataset class
"""
import pandas as pd

import pytest
import numpy as np

from wildwood.preprocessing import Encoder, dataset_to_array

np.random.seed(42)


def test_encoder_max_bins():
    encoder = Encoder()
    assert encoder.max_bins == 256
    encoder = Encoder(max_bins=127)
    assert encoder.max_bins == 127
    encoder.max_bins = 42
    assert encoder.max_bins == 42
    with pytest.raises(ValueError, match="max_bins must be an integer number"):
        encoder = Encoder(max_bins=3.14)
    with pytest.raises(ValueError, match="max_bins must be >= 4"):
        encoder.max_bins = 3


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


def get_example1():
    """First example with no missing values, three categoricals and two continuous
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

    for colname in ["A", "B", "D"]:
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

    return (
        df,
        max_bins,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
    )


def get_example2():
    """Example with missing values both for continuous and categoricals"""
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

    return (
        df,
        max_bins,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
    )


def get_example3():
    """Same as first example but with default (256) bins
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

    return (
        df,
        max_bins,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
    )


def get_example4():
    """The same with example 2 but with the default max_bins (256)
    """
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

    return (
        df,
        max_bins,
        n_bins_no_missing_values_,
        categories_,
        binning_thresholds_,
        X_binned,
        df_inverse_transform,
    )


@pytest.mark.parametrize(
    "df, max_bins, n_bins_no_missing_values_, categories_, binning_thresholds_, "
    "X_binned, df_inverse_transform",
    [get_example1(), get_example2(), get_example3(), get_example4()],
)
def test_encoder_fit_transform(
    df,
    max_bins,
    n_bins_no_missing_values_,
    categories_,
    binning_thresholds_,
    X_binned,
    df_inverse_transform,
):
    encoder = Encoder(max_bins=max_bins)
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
    X_binned_out = dataset_to_array(dataset)
    np.testing.assert_array_equal(X_binned_out, X_binned)

    # Check that reconstructed dataframe is correct
    df_inverse_transform_out = encoder.inverse_transform(dataset)
    pd.testing.assert_frame_equal(df_inverse_transform_out, df_inverse_transform)

    # Test also fit_transform
    encoder = Encoder(max_bins=max_bins)
    dataset = encoder.fit_transform(df)
    df_inverse_transform_out = encoder.inverse_transform(dataset)
    pd.testing.assert_frame_equal(df_inverse_transform_out, df_inverse_transform)


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
    X_binned_out = dataset_to_array(dataset)
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
