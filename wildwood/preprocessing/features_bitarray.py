# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""This modules introduces the ``FeaturesBitArray`` class, used to store a binned
features matrix. It uses internally a bitarray to save the values of the features in
a memory efficient fashion. It exploits the fact that any column (at index j) of the
binned features matrix contains only contiguous non-negative integers {0, 1, 2, ...,
max_value_j} obtained through binning of both categorical and numerical columns.

If a column contains M different values, it will look for the minimum number of bits
required to save such values, and it will stack them into words (of 64 bits length) of
a contiguous memory region of a bitarray (a 1D ``numpy.ndarray``).

For familiarity with bitwise operations:
https://en.wikipedia.org/wiki/Bitwise_operation
"""

from math import ceil, floor
import numpy as np
from numba import jit, void, uint8, int8, uint16, int16, uint32, int32, uint64, int64
from numba.experimental import jitclass
from .._utils import NOPYTHON, NOGIL, BOUNDSCHECK, FASTMATH, INLINE, CACHE, get_type

CACHE = True

_UINT_LIMITS = {
    'uint8': np.iinfo(np.uint8).max,
    'uint16': np.iinfo(np.uint16).max, 
    'uint32': np.iinfo(np.uint32).max,
    'uint64': np.iinfo(np.uint64).max
}

_VALID_DTYPES = (np.uint8, np.uint16, np.uint32, np.uint64)

spec_features_bit_array = [
    ("n_samples", uint64),
    ("n_features", uint64),
    ("max_values", uint64[::1]),
    ("n_bits", uint64[::1]),
    ("offsets", uint64[::1]),
    ("n_values_in_words", uint64[::1]),
    ("bitarray", uint64[::1]),
    ("bitmasks", uint64[::1]),
]

@jitclass(spec_features_bit_array)
class FeaturesBitArray:
    def __init__(self, n_samples, max_values):
        self.n_samples = n_samples
        self.n_features = max_values.size
        self.max_values = max_values
        
        # Initialize arrays
        self._init_arrays()
        # Calculate values
        self._calculate_bit_layout()

    def _init_arrays(self):
        """Initialize empty arrays with correct types"""
        self.n_bits = np.empty(self.n_features, dtype=np.uint64)
        self.offsets = np.empty(self.n_features + 1, dtype=np.uint64)
        self.n_values_in_words = np.empty(self.n_features, dtype=np.uint64)
        self.bitmasks = np.empty(self.n_features, dtype=np.uint64)
        
    def _calculate_bit_layout(self):
        """Calculate bit layout for all features"""
        offset = 0
        self.offsets[0] = offset
        
        for j, max_value in enumerate(self.max_values):
            # Calculate bits needed
            if max_value <= 1:
                n_bits = 1
                n_values = 64 
                bitmask = 1
            else:
                n_bits = ceil(np.log2(max_value + 1))
                n_values = floor(64 / n_bits)
                bitmask = (1 << n_bits) - 1
                
            self.n_bits[j] = n_bits
            self.n_values_in_words[j] = n_values
            self.bitmasks[j] = bitmask

            # Calculate offset
            offset += ceil(self.n_samples / n_values)
            self.offsets[j + 1] = offset
            
        self.bitarray = np.empty(offset, dtype=np.uint64)

FeaturesBitArrayType = get_type(FeaturesBitArray)

@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH, cache=CACHE)
def _features_bitarray_fill_column(col_bitarray, n_bits, n_values_in_word, col):
    """Fill a single column in the bitarray"""
    for i, x_ij in enumerate(col):
        word = i // n_values_in_word
        pos_in_word = i % n_values_in_word
        col_bitarray[word] = x_ij if pos_in_word == 0 else (col_bitarray[word] << n_bits) | x_ij
    
    # Left-align last word
    last_pos = (len(col) - 1) % n_values_in_word
    col_bitarray[word] <<= ((n_values_in_word - last_pos - 1) * n_bits)

@jit(
    # [void(DatasetType, col_type[:, :]) for col_type in numba_int_types],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    fastmath=FASTMATH,
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
def _features_bitarray_fill_values(features_bitarray, X):
    """Private function that fills the values in X inside the bitarray.

    Parameters
    ----------
    features_bitarray : FeaturesBitArray
        The bitarray to fill with the values in X

    X : numpy.ndarray
        Numpy array of shape (n_samples, n_features) corresponding to the matrix of
        features to be transformed in a bitarray. This function exploits the fact
        that all the columns of X contain only contiguous non-negative integers {0,
        1, 2, ..., max_value} obtained through binning of both categorical and
        continuous columns.
    """
    bitarray = features_bitarray.bitarray
    offsets = features_bitarray.offsets
    n_values_in_words = features_bitarray.n_values_in_words
    n_bits = features_bitarray.n_bits
    n_features = features_bitarray.n_features

    for j in range(n_features):
        col_bitarray = bitarray[offsets[j] : offsets[j + 1]]
        _features_bitarray_fill_column(
            col_bitarray, n_bits[j], n_values_in_words[j], X[:, j]
        )


def features_bitarray_fill_column(features_bitarray, col_idx, col):
    """Fills the values of a column in the bitarray.

    Parameters
    ----------
    features_bitarray : FeaturesBitArray
        The bitarray to fill with the values in X

    col_idx : int
        Index of the column in the X

    col : numpy.ndarray
        Numpy array of shape (n_samples,) corresponding to the values of a column to
        add to the bitarray. This function exploits the fact that the values in col
        contain only contiguous non-negative integers {0, 1, 2, ..., max_value}
        coming from binning of both categorical and continuous columns.
    """
    bitarray = features_bitarray.bitarray
    offsets = features_bitarray.offsets
    col_bitarray = bitarray[offsets[col_idx] : offsets[col_idx + 1]]
    n_values_in_word = features_bitarray.n_values_in_words[col_idx]
    n_bits = features_bitarray.n_bits[col_idx]
    _features_bitarray_fill_column(col_bitarray, n_bits, n_values_in_word, col)


def array_to_bitarray(X):
    """Convert numpy array to bitarray with validation"""
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.dtype not in _VALID_DTYPES:
        raise ValueError("X must be a 2D numpy array with uint dtype")

    n_samples, n_features = X.shape
    max_values = X.max(axis=0, initial=0, out=np.empty(n_features, dtype=np.uint64))
    
    features_bitarray = FeaturesBitArray(n_samples, max_values)
    _features_bitarray_fill_values(features_bitarray, X)
    return features_bitarray

def _get_empty_matrix(n_samples, n_features, max_value):
    """Create empty matrix with optimal dtype"""
    for dtype, limit in _UINT_LIMITS.items():
        if max_value <= limit:
            return np.empty((n_samples, n_features), dtype=getattr(np, dtype), order="F")
    raise ValueError("Values too large for supported dtypes")

@jit(
    uint64(uint64, uint64[::1], uint64, uint64, uint64),
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    fastmath=FASTMATH,
    inline=INLINE,
)
def get_value_from_column(i, bitarray, bitmask, n_values_in_word, n_bits):
    """Get the bin value of a column based on the bitarray

    Parameters
    ----------
    i : uint64
        Sample index

    bitarray :
        Bitarray containing the values of the

    bitmask :

    n_values_in_word :

    n_bits :

    Returns
    -------

    """
    word = i // n_values_in_word
    pos_in_word = i % n_values_in_word
    b = bitarray[word]
    n_shifts = (n_values_in_word - pos_in_word - 1) * n_bits
    return (b & (bitmask << n_shifts)) >> n_shifts


@jit(
    [
        void(FeaturesBitArrayType, uint8[:, :]),
        void(FeaturesBitArrayType, uint16[:, :]),
        void(FeaturesBitArrayType, uint32[:, :]),
        void(FeaturesBitArrayType, uint64[:, :]),
        void(FeaturesBitArrayType, uint8[::1, :]),
        void(FeaturesBitArrayType, uint16[::1, :]),
        void(FeaturesBitArrayType, uint32[::1, :]),
        void(FeaturesBitArrayType, uint64[::1, :]),
    ],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    fastmath=FASTMATH,
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
def _features_bitarray_to_array(features_bitarray, X):
    n_samples = features_bitarray.n_samples
    n_features = features_bitarray.n_features
    n_values_in_words = features_bitarray.n_values_in_words
    offsets = features_bitarray.offsets
    bitarray = features_bitarray.bitarray
    n_bits = features_bitarray.n_bits
    bitmasks = features_bitarray.bitmasks

    for j in range(n_features):
        n_values_in_word = n_values_in_words[j]
        bitarray_j = bitarray[offsets[j] : offsets[j + 1]]
        n_bits_j = n_bits[j]
        bitmask = bitmasks[j]
        for i in range(n_samples):
            X[i, j] = get_value_from_column(
                i, bitarray_j, bitmask, n_values_in_word, n_bits_j
            )


@jit(
    [uint64(FeaturesBitArrayType, uint64, uint64)],
    nopython=NOPYTHON,
    nogil=NOGIL,
    boundscheck=BOUNDSCHECK,
    fastmath=FASTMATH,
    inline=INLINE,
)
def get_value(features_bitarray, i, j):
    n_values_in_word = features_bitarray.n_values_in_words[j]
    bitarray_j = features_bitarray.bitarray[
        features_bitarray.offsets[j] : features_bitarray.offsets[j + 1]
    ]
    n_bits_j = features_bitarray.n_bits[j]
    bitmask = features_bitarray.bitmasks[j]
    return get_value_from_column(i, bitarray_j, bitmask, n_values_in_word, n_bits_j)


def features_bitarray_to_array(features_bitarray):
    X = _get_empty_matrix(
        features_bitarray.n_samples,
        features_bitarray.n_features,
        features_bitarray.max_values.max(),
    )
    _features_bitarray_to_array(features_bitarray, X)
    return X
