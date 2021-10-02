import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from wildwood.preprocessing import Encoder, dataset_to_array, array_to_dataset

# random_state = 42
# n_samples = 17
# n_features = 3
# rng = np.random.RandomState(random_state)
# X = rng.randn(n_samples, n_features)
# max_bins = 5

# random_state = 42
# n_samples = 9
# n_features = 3
#
# X = np.array(
#     [
#         ["a", 0.1],
#         ["a", 0.2],
#         ["b", 0.1],
#         ["a", 0.3],
#         ["a", 0.0],
#         ["c", -1.2],
#         ["b", 0.1],
#     ]
# )

# X[0, 0] = np.nan
# X[4, 2] = np.nan
# X[4, 2] = np.nan
# max_bins = 4
# is_categorical = None
# is_categorical = np.array([False, False])


# is_categorical = None

# df = pd.DataFrame({"col": pd.to_datetime(["2011-10-01", "2009-08-17"])})
#
# X = np.empty((3, 2), dtype=np.dtype([("a", np.float64)]))
#
# encoder = Encoder().fit(X)

X1 = np.random.randn(9, 4)
X2 = np.random.randn(9, 3)

is_categorical = [True, False]
X = np.random.randn(7, 3)


encoder = Encoder(is_categorical=is_categorical).fit(X)
# encoder.transform(X2)


# is_categorical = [False, False]
# encoder = Encoder(is_categorical=is_categorical).fit(X)
# "Column 0 is declared as numerical, but it cannot be converted to float"


print("encoder.n_samples_in_:", encoder.n_samples_in_)
print("encoder.n_features_in_:", encoder.n_features_in_)
print("encoder.n_bins_no_missing_values_:", encoder.n_bins_no_missing_values_)
print("encoder.categories_:", encoder.categories_)
print("encoder.binning_thresholds_:", encoder.binning_thresholds_)

# Check that dataset is correct
dataset = encoder.transform(X)
X_binned_out = dataset_to_array(dataset)

print("X_binned_out:", X_binned_out)


df_inverse_transform_out = encoder.inverse_transform(dataset)
print("df_inverse_transform_out:", df_inverse_transform_out)



exit(0)


# df = pd.DataFrame(
#     {
#         "A": [None, "a", "b", None, "a", "a", "d", "c"],
#         "B": [3, None, 0, -1, 42, 7, 1, None],
#         "C": ["b", "a", "b", "c", "a", "a", "d", "c"],
#         "D": [-4, 1, 2, 1, -3, None, 2, 3.0],
#         "E": [-4, 1, 2, 1, -3, 1, 2, 3],
#     }
# )


# @pytest.mark.parametrize("n_samples", [1, 2, 32, 17, 1000, 1_000_000])
# @pytest.mark.parametrize(
#     "max_values, dtype",
#     [
#         (np.array([1], dtype=np.uint64), np.uint8),
#         (np.array([32], dtype=np.uint64), np.uint8),
#         (np.array([17, 2, 64], dtype=np.uint64), np.uint8),
#         (np.array([1024, 32, 64], dtype=np.uint64), np.uint16),
#         (np.array([123765, 2, 16], dtype=np.uint64), np.uint32),
#         (np.array([12323765, 123765, 2, 1024], dtype=np.uint64), np.uint64),
#         (np.array([2378, 2, 213, 123765, 7, 64, 1024, 3], dtype=np.uint64), np.uint64),
#     ],
# )

#
# def test_encoder_large_all_categorical(n_samples, max_values, dtype):
#     n_features = max_values.size
#     X_in = np.asfortranarray(
#         np.random.randint(max_values + 1, size=(n_samples, n_features)), dtype=dtype
#     )
#     df_in = pd.DataFrame(X_in).astype("category")
#
#     # dataset_truth = array_to_dataset(X_in)
#     # encoder =
#     encoder = Encoder()
#     dataset_out = encoder.fit_transform(df_in)
#     df_out = encoder.inverse_transform(dataset_out)
#
#     print("=" * 8)
#     print(df_in.head())
#     print('-' * 4)
#     print(df_out.head())
#     print("=" * 8)
#     print(df_in.info())
#     print('-' * 4)
#     print(df_out.info())
#     print("=" * 8)
#     print(df_in.index)
#     print('-' * 4)
#     print(df_out.index)
#     print("=" * 8)
#     print(df_in.columns)
#     print('-' * 4)
#     print(df_out.columns)
#
#     assert df_in.equals(df_out)
#     # assert pd.testing.assert_frame_equal(df_in, df_out, check_index_type=False,
#     #                                      check_column_type=False)
#
#     # print("out")
#     # assert dataset_out.n_samples == dataset_truth.n_samples
#     # assert dataset_out.n_features == dataset_truth.n_features
#     # np.testing.assert_array_equal(dataset_out.max_values, dataset_truth.max_values)
#     # np.testing.assert_array_equal(dataset_out.n_bits, dataset_truth.n_bits)
#     # np.testing.assert_array_equal(dataset_out.offsets, dataset_truth.offsets)
#     # np.testing.assert_array_equal(
#     #    dataset_out.n_values_in_words, dataset_truth.n_values_in_words
#     #)
#     #np.testing.assert_array_equal(dataset_out.bitmasks, dataset_truth.bitmasks)
#
#
# for n_samples in [17, 32, 17, 1000, 1_000_000]:
#     for max_values, dtype in [
#         # (np.array([1], dtype=np.uint64), np.uint8),
#         (np.array([32], dtype=np.uint64), np.uint8),
#         (np.array([17, 2, 64], dtype=np.uint64), np.uint8),
#         (np.array([1024, 32, 64], dtype=np.uint64), np.uint16),
#         (np.array([123765, 2, 16], dtype=np.uint64), np.uint32),
#         (np.array([12323765, 123765, 2, 1024], dtype=np.uint64), np.uint64),
#         (np.array([2378, 2, 213, 123765, 7, 64, 1024, 3], dtype=np.uint64), np.uint64),
#     ]:
#
# # for n_samples in [7]:
# #     for max_values, dtype in [
# #         # (np.array([1], dtype=np.uint64), np.uint8),
# #         (np.array([32], dtype=np.uint64), np.uint8),
# #         # (np.array([17, 2, 64], dtype=np.uint64), np.uint8),
# #         # (np.array([1024, 32, 64], dtype=np.uint64), np.uint16),
# #         # (np.array([123765, 2, 16], dtype=np.uint64), np.uint32),
# #         # (np.array([12323765, 123765, 2, 1024], dtype=np.uint64), np.uint64),
# #         # (np.array([2378, 2, 213, 123765, 7, 64, 1024, 3], dtype=np.uint64),
# #         #  np.uint64),
# #     ]:
#
#         test_encoder_large_all_categorical(n_samples, max_values, dtype)
#
#
# exit(0)


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

# for colname in ["A", "B", "D"]:
for colname in ["A", "D"]:
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
    # 1: np.array(["a", "b", "c"]),
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
df_inverse_transform[0] = pd.Series([0, 1, 1, 1, None, 0], dtype="category")
df_inverse_transform[1] = pd.Series(["b", "b", "c", "c", "b", "b"], dtype="category")
df_inverse_transform[2] = pd.arrays.IntervalArray.from_tuples(
    [(3.0, 8.0), (1.0, 3.0), (-np.inf, 1.0), (-np.inf, 1.0), np.nan, (3.0, 8.0)]
)
df_inverse_transform[3] = pd.Categorical(
    ["c", "b", np.nan, "c", np.nan, np.nan], categories=categories_[3]
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

print("print(df_inverse_transform_out)")
print(df_inverse_transform_out)

print("df_inverse_transform")
print(df_inverse_transform)


pd.testing.assert_frame_equal(df_inverse_transform_out, df_inverse_transform)


exit(0)


n_samples = 1000
max_values = np.array([1024, 32, 64], dtype=np.uint64)
# dtype = np.uint16

n_features = max_values.size

X_in = np.asfortranarray(
    np.random.randint(max_values + 1, size=(n_samples, n_features))
)
df_in = pd.DataFrame(data=X_in).astype("category")


encoder = Encoder()
dataset_out = encoder.fit_transform(df_in)
df_out = encoder.inverse_transform(dataset_out)

print(df_in.info())
print(df_out.info())

assert_frame_equal(df_out, df_in)


# Et cette partie fail mais a cause de la fonction array_to_dataset qui n'utilise pas
# les max_values observees mais celles qui sont passees..
# dataset_truth = array_to_dataset(X_in, max_values=max_values)
#
#
# assert dataset_out.n_samples == dataset_truth.n_samples
# assert dataset_out.n_features == dataset_truth.n_features
#
# print("dataset_out.max_values:", dataset_out.max_values)
# print("dataset_truth.max_values:", dataset_truth.max_values)
#
# # np.testing.assert_array_equal(dataset_out.max_values, dataset_truth.max_values)
# np.testing.assert_array_equal(dataset_out.n_bits, dataset_truth.n_bits)
# np.testing.assert_array_equal(dataset_out.offsets, dataset_truth.offsets)
# np.testing.assert_array_equal(
#     dataset_out.n_values_in_words, dataset_truth.n_values_in_words
# )
# np.testing.assert_array_equal(dataset_out.bitmasks, dataset_truth.bitmasks)

#
# df = pd.DataFrame(
#     {
#         "A": [0, 1, 1, 1, 0, 0, 0, 0, 1],
#         "B": ["b", "a", "b", "c", "a", "a", "b", "c", "b"],
#         "C": [3, 3, 0, -1, 42, 7, 1, 17, 8],
#         "D": ["b", "a", "b", "c", "a", "a", "d", "c", "a"],
#         "E": [-4, 1, 2, 1, -3, 17, 2, 3.0, -1],
#     },
# )
#
# # Same dataframe as df but with less data and less modalities / unknown modalities
# df2 = pd.DataFrame(
#     {
#         "A": [0, 1, 1, 1, None, 0],
#         # Same column as df["B"] but with less modalities
#         "B": ["b", "b", "c", "c", "b", "b"],
#         # Same column as df["C"] but with a missing value
#         "C": [4, 2, -4, -1, None, 7],
#         # Same column with an unknown modality and missing ones
#         "D": ["c", "b", "f", "c", None, "f"],
#         "E": [-1, 4, -67, 128, 2, 0],
#     },
# )
#
# for colname in ["A", "B", "D"]:
#     df[colname] = df[colname].astype("category")
#     df2[colname] = df2[colname].astype("category")
#
#
# max_bins = 4
#
# encoder = Encoder(max_bins=max_bins, handle_unknown="consider_missing")
#
# print("df:")
# print(df)
#
# # print("ndim:", hasattr(df, "ndim"))
# # print("dtype:", hasattr(df, "dtype"))
# # print("shape:", hasattr(df, "shape"))
#
#
# encoder.fit(df)
#
# print("encoder.n_bins_no_missing_values_:")
# print(encoder.n_bins_no_missing_values_)
#
# print("encoder.binning_thresholds_:")
# print(encoder.binning_thresholds_)
#
# print("encoder.categories_:")
# print(encoder.categories_)
#
# print("encoder.is_categorical_")
# print(encoder.is_categorical_)
#
# print("df2:")
# print(df2)
#
#
# dataset = encoder.transform(df2)
#
#
# X_binned = dataset_to_array(dataset)
#
# print("X_binned:")
# print(X_binned)
#
# X_inverse = encoder.inverse_transform(dataset)
#
# print("X_inverse:")
# print(X_inverse)
