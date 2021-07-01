# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause


"""This modules introduces the Dataset class allowing to store a binned features matrix.
It uses internally a bitarray to save the values of the features in a memory efficient
fashion. It exploits the fact that any columns j of the features matrix X contain
only contiguous non-negative integers {0, 1, 2, ..., max_value_j} obtained through
binning of both categorical and continuous columns.

If a column contains M modalities, it will look for the minimum number of bits required
to save such values, and will stack them into 64 bits words of a contiguous memory
region of a bitarray (a 1D numpy array, using a F-major ordering of the matrix X).

For familiarity with bitwise operations:
https://en.wikipedia.org/wiki/Bitwise_operation
"""

from math import ceil, floor
import numpy as np
from numba import jit, void, uint8, int8, uint16, int16, uint32, int32, uint64, int64
from numba.experimental import jitclass
from .._utils import get_type


# Global jit decorator options
NOPYTHON = True
NOGIL = True
BOUNDSCHECK = False
CACHE = True


_UINT8_MAX = np.iinfo(np.uint8).max
_UINT16_MAX = np.iinfo(np.uint16).max
_UINT32_MAX = np.iinfo(np.uint32).max
_UINT64_MAX = np.iinfo(np.uint64).max


spec_dataset = [
    # Number of samples in the dataset
    ("n_samples", uint64),
    # Number of features in the dataset
    ("n_features", uint64),
    # maximum value in each column
    ("max_values", uint64[::1]),
    # Number of bits used for each values of each columns
    ("n_bits", uint64[::1]),
    # bitarray[offsets[j]:offsets[j+1]] is the array of words for the j-th column
    ("offsets", uint64[::1]),
    # n_values_in_words[j] is the number of values saved in a word for column j
    ("n_values_in_words", uint64[::1]),
    # The bitarray containing all values
    ("bitarray", uint64[::1]),
    # The bitmasks used for each column
    ("bitmasks", uint64[::1]),
]


@jitclass(spec_dataset)
class Dataset(object):
    """This is a class containing the binned features matrix. It uses internally a
    bitarray to save the values of the features in a memory efficient fashion. It
    exploits the fact that all the columns of the features matrix X contain only
    contiguous non-negative integers {0, 1, 2, ..., max_value} obtained through
    binning of both categorical and continuous columns.

    If a column contains M modalities, it will look for the minimum number of bits
    required to save such values, and will stack them into 64 bits words in a
    contiguous memory region of the bitarray (a 1D numpy array, using a F-major
    ordering of the matrix X).

    For familiarity with bitwise operations:
    https://en.wikipedia.org/wiki/Bitwise_operation

    Parameters
    ----------
    n_samples : int
        Number samples (rows) in the dataset

    max_values : ndarray
        Number array of shape (n_features,) containing the maximum value (number of
        bins + 1) in each column.

    Attributes
    ----------
    n_samples : int
        Number samples (rows) in the dataset

    n_features : int
        Number of features (columns) in the dataset

    max_values : ndarray
        Numpy array of shape (n_features,) containing the maximum value (number of
        bins + 1) in each column.

    n_bits : ndarray
        Numpy array of shape (n_features,) such that n_bits[j] is the number of bits
        used for the values of the j-th column

    offsets : ndarray
        Numpy array of shape (n_features + 1,) such that
        bitarray[offsets[j]:offsets[j+1]] is the array of words for the j-th column

    n_values_in_words : ndarray
        Numpy array of shape (n_features,) such that n_values_in_words[j] is the number
        of values saved in a single 64-bits word for the values in column j

    bitmasks : ndarray
        Numpy array of shape (n_features,)  such that bitmasks[j] contains the
        bitmask using the shift and back-shift operations to retrieve values from the
        bitarray

    bitarray : ndarray
        Numpy array of shape (n_total_words,) containing the values of the dataset,
        where n_total_words is the total number of words used (for all columns) to
        store the values.
    """

    def __init__(self, n_samples, max_values):
        self.n_samples = n_samples
        self.n_features = max_values.size
        self.max_values = max_values
        self.n_bits = np.empty(self.n_features, dtype=np.uint64)
        self.offsets = np.empty(self.n_features + 1, dtype=np.uint64)
        self.n_values_in_words = np.empty(self.n_features, dtype=np.uint64)
        self.bitmasks = np.empty(self.n_features, dtype=np.uint64)

        # The first offset is 0
        offset = 0
        self.offsets[0] = offset
        for j, max_value in enumerate(max_values):
            # Number of bits required to save numbers up to n_modalities
            if max_value == 1:
                self.n_bits[j] = 1
                self.n_values_in_words[j] = 64
                self.bitmasks[j] = 1
            else:
                self.n_bits[j] = ceil(np.log2(max_value + 1))
                self.n_values_in_words[j] = floor(64 / self.n_bits[j])
                self.bitmasks[j] = (1 << self.n_bits[j]) - 1

            n_words = ceil(n_samples / self.n_values_in_words[j])
            offset += n_words
            self.offsets[j + 1] = offset

        self.bitarray = np.empty(offset, dtype=np.uint64)


DatasetType = get_type(Dataset)


numba_int_types = [uint8, int8, uint16, int16, uint32, int32, uint64, int64]


# TODO: put back signatures everywhere


@jit(
    # [void(uint64[::1], uint64, uint64, col_type[:]) for col_type in numba_int_types],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    cache=CACHE,
    locals={"i": uint64, "x_ij": uint64, "word": uint64, "pos_in_word": uint64},
)
def _dataset_fill_column(col_bitarray, n_bits, n_values_in_word, col):
    """Private function that fills the values of a column in the dataset.

    Parameters
    ----------
    col_bitarray : ndarray
        Numpy array of shape (n_words,) containing the values of the column, where
        n_words is the number of words used to store its values.

    n_bits : int
        Number of bits used to store one value from the column

    n_values_in_word : int
        Number of values from the column saved in a single 64-bits word

    col : ndarray
        Numpy array of shape (n_samples,) corresponding to the values of a column to
        add to the dataset. This function exploits the fact that the values in col
        contain only contiguous non-negative integers {0, 1, 2, ..., max_value}
        coming from binning of both categorical and continuous columns.
    """
    for i, x_ij in enumerate(col):
        word = i // n_values_in_word
        pos_in_word = i % n_values_in_word
        if pos_in_word == 0:
            col_bitarray[word] = x_ij
        else:
            col_bitarray[word] = (col_bitarray[word] << n_bits) | x_ij

    # We need to shift the last word according to the position of the last value in
    # the word, so that the bits of the values in the last word are on the left
    # of it. If pos_in_word = n_values_in_word - 1 it does nothing, since the
    # word is full and already left-aligned
    col_bitarray[word] = col_bitarray[word] << (
        (n_values_in_word - pos_in_word - 1) * n_bits
    )


@jit(
    # [void(DatasetType, col_type[:, :]) for col_type in numba_int_types],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    cache=CACHE,
    locals={
        "bitarray": uint64[::1],
        "offsets": uint64[::1],
        "n_values_in_words": uint64[::1],
        "n_bits": uint64[::1],
        "n_features": uint64,
        "j": uint64,
        "n_values_in_word": uint64,
        "bitarray_j": uint64[::1],
        "n_bits_j": uint64,
        "i": uint64,
        "x_ij": uint64,
        "word": uint64,
        "pos_in_word": uint64,
    },
)
def _dataset_fill_values(dataset, X):
    """Private function that fills the values in X inside the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to fill with the values in X

    X : ndarray
        Numpy array of shape (n_samples, n_features) corresponding to the matrix of
        features to be transformed in a Dataset. This function exploits the fact
        that all the columns of X contain only contiguous non-negative integers {0,
        1, 2, ..., max_value} obtained through binning of both categorical and
        continuous columns.
    """
    bitarray = dataset.bitarray
    offsets = dataset.offsets
    n_values_in_words = dataset.n_values_in_words
    n_bits = dataset.n_bits
    n_features = dataset.n_features

    for j in range(n_features):
        col_bitarray = bitarray[offsets[j] : offsets[j + 1]]
        _dataset_fill_column(col_bitarray, n_bits[j], n_values_in_words[j], X[:, j])


def dataset_fill_column(dataset, col_idx, col):
    """Fills the values of a column in the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to fill with the values in X

    col_idx : int
        Index of the column in the dataset

    col : ndarray
        Numpy array of shape (n_samples,) corresponding to the values of a column to
        add to the dataset. This function exploits the fact that the values in col
        contain only contiguous non-negative integers {0, 1, 2, ..., max_value}
        coming from binning of both categorical and continuous columns.
    """
    bitarray = dataset.bitarray
    offsets = dataset.offsets
    col_bitarray = bitarray[offsets[col_idx] : offsets[col_idx + 1]]
    n_values_in_word = dataset.n_values_in_words[col_idx]
    n_bits = dataset.n_bits[col_idx]
    _dataset_fill_column(col_bitarray, n_bits, n_values_in_word, col)


def array_to_dataset(X):
    """Converts a numpy array to a Dataset.

    Parameters
    ----------
    X : ndarray
        Numpy array of shape (n_samples, n_features) corresponding to the matrix of
        features to be transformed to a Dataset. This function exploits the fact
        that all the columns of X contain only contiguous non-negative integers {0,
        1, 2, ..., max_value} obtained through binning of both categorical and
        continuous columns.

    Returns
    -------
    output : Dataset
        The dataset corresponding to the values in X.
    """
    n_samples, n_features = X.shape
    max_values = np.empty(n_features, dtype=np.uint64)
    X.max(axis=0, initial=0, out=max_values)
    if hasattr(X, "ndim") and hasattr(X, "dtype") and hasattr(X, "shape"):
        if X.ndim == 2:
            if X.dtype not in (np.uint8, np.uint16, np.uint32, np.uint64):
                raise ValueError(
                    "X dtype must be one of uint8, uint16, uint32 or " "uint64"
                )
        else:
            raise ValueError("X is must be a 2D numpy array")
    else:
        raise ValueError("X is not a numpy array")

    if X.shape[1] != max_values.size:
        raise ValueError("max_values size must match X.shape[1]")

    dataset = Dataset(n_samples, max_values)
    _dataset_fill_values(dataset, X)
    return dataset


def _get_empty_matrix(n_samples, n_features, max_value):
    """A private function that creates an empty F-ordered ndarray with shape
    (n_samples, n_features) and dtype in (uint8, uint16, uint32, uint64) depending on
    the exected maximum value to store in it.

    Parameters
    ----------
    n_samples : int
        Number of samples (number of rows of the matrix)

    n_features : int
        Number of features (number of columns of the matrix)

    max_value : int
        Maximum value expected in the matrix (to choose the dtype)

    Returns
    -------
    output : ndarray
        An ndarray with shape (n_samples, n_features) and minimal dtype to store values
    """
    # Let's find out the correct dtype depending on the max_value
    if max_value <= _UINT8_MAX:
        X = np.empty((n_samples, n_features), dtype=np.uint8, order="F")
    elif _UINT8_MAX < max_value <= _UINT16_MAX:
        X = np.empty((n_samples, n_features), dtype=np.uint16, order="F")
    elif _UINT16_MAX < max_value <= _UINT32_MAX:
        X = np.empty((n_samples, n_features), dtype=np.uint32, order="F")
    elif _UINT32_MAX < max_value <= _UINT64_MAX:
        X = np.empty((n_samples, n_features), dtype=np.uint64, order="F")
    else:
        raise ValueError("X cannot be created")
    return X


@jit(
    [
        void(DatasetType, uint8[:, :]),
        void(DatasetType, uint16[:, :]),
        void(DatasetType, uint32[:, :]),
        void(DatasetType, uint64[:, :]),
        void(DatasetType, uint8[::1, :]),
        void(DatasetType, uint16[::1, :]),
        void(DatasetType, uint32[::1, :]),
        void(DatasetType, uint64[::1, :]),
    ],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    locals={
        "n_samples": uint64,
        "n_features": uint64,
        "n_values_in_words": uint64[::1],
        "offsets": uint64[::1],
        "bitarray": uint64[::1],
        "n_bits": uint64[::1],
        "bitmasks": uint64[::1],
        "j": uint64,
        "n_values_in_word": uint64,
        "bitarray_j": uint64[::1],
        "n_bits_j": uint64,
        "bitmask": uint64,
        "i": uint64,
        "word": uint64,
        "pos_in_word": uint64,
        "b": uint64,
        "n_shifts": uint64,
    },
)
def _dataset_to_array(dataset, X):
    n_samples = dataset.n_samples
    n_features = dataset.n_features
    n_values_in_words = dataset.n_values_in_words
    offsets = dataset.offsets
    bitarray = dataset.bitarray
    n_bits = dataset.n_bits
    bitmasks = dataset.bitmasks

    for j in range(n_features):
        n_values_in_word = n_values_in_words[j]
        bitarray_j = bitarray[offsets[j] : offsets[j + 1]]
        n_bits_j = n_bits[j]
        bitmask = bitmasks[j]
        for i in range(n_samples):
            word = i // n_values_in_word
            pos_in_word = i % n_values_in_word
            b = bitarray_j[word]
            n_shifts = (n_values_in_word - pos_in_word - 1) * n_bits_j
            X[i, j] = (b & (bitmask << n_shifts)) >> n_shifts


def dataset_to_array(dataset):
    X = _get_empty_matrix(
        dataset.n_samples, dataset.n_features, dataset.max_values.max()
    )
    _dataset_to_array(dataset, X)
    return X
