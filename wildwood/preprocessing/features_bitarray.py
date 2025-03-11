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

_UINT8_MAX = np.iinfo(np.uint8).max
_UINT16_MAX = np.iinfo(np.uint16).max
_UINT32_MAX = np.iinfo(np.uint32).max
_UINT64_MAX = np.iinfo(np.uint64).max


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
        self.n_bits = np.empty(self.n_features, dtype=np.uint64)
        self.offsets = np.empty(self.n_features + 1, dtype=np.uint64)
        self.n_values_in_words = np.empty(self.n_features, dtype=np.uint64)
        self.bitmasks = np.empty(self.n_features, dtype=np.uint64)
        offset = 0
        self.offsets[0] = offset
        for j, max_value in enumerate(max_values):
            self.n_bits[j] = 1 if max_value <= 1 else ceil(np.log2(max_value + 1))
            self.n_values_in_words[j] = 64 if max_value <= 1 else floor(64 / self.n_bits[j])
            self.bitmasks[j] = 1 if max_value <= 1 else (1 << self.n_bits[j]) - 1
            offset += ceil(n_samples / self.n_values_in_words[j])
            self.offsets[j + 1] = offset
        self.bitarray = np.empty(offset, dtype=np.uint64)

FeaturesBitArrayType = get_type(FeaturesBitArray)

@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH, cache=CACHE)
def _features_bitarray_fill_column(col_bitarray, n_bits, n_values_in_word, col):
    for i, x_ij in enumerate(col):
        word = i // n_values_in_word
        pos_in_word = i % n_values_in_word
        col_bitarray[word] = x_ij if pos_in_word == 0 else (col_bitarray[word] << n_bits) | x_ij
    col_bitarray[word] <<= (n_values_in_word - pos_in_word - 1) * n_bits

@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH, cache=CACHE)
def _features_bitarray_fill_values(features_bitarray, X):
    bitarray, offsets, n_values_in_words, n_bits = features_bitarray.bitarray, features_bitarray.offsets, features_bitarray.n_values_in_words, features_bitarray.n_bits
    for j in range(features_bitarray.n_features):
        _features_bitarray_fill_column(bitarray[offsets[j]:offsets[j + 1]], n_bits[j], n_values_in_words[j], X[:, j])

def features_bitarray_fill_column(features_bitarray, col_idx, col):
    _features_bitarray_fill_column(features_bitarray.bitarray[features_bitarray.offsets[col_idx]:features_bitarray.offsets[col_idx + 1]], features_bitarray.n_bits[col_idx], features_bitarray.n_values_in_words[col_idx], col)

def array_to_bitarray(X):
    n_samples, n_features = X.shape
    max_values = X.max(axis=0, initial=0, out=np.empty(n_features, dtype=np.uint64))
    if X.ndim != 2 or X.dtype not in (np.uint8, np.uint16, np.uint32, np.uint64):
        raise ValueError("X must be a 2D numpy array with dtype uint8, uint16, uint32, or uint64")
    features_bitarray = FeaturesBitArray(n_samples, max_values)
    _features_bitarray_fill_values(features_bitarray, X)
    return features_bitarray

def _get_empty_matrix(n_samples, n_features, max_value):
    dtype = np.uint8 if max_value <= _UINT8_MAX else np.uint16 if max_value <= _UINT16_MAX else np.uint32 if max_value <= _UINT32_MAX else np.uint64
    return np.empty((n_samples, n_features), dtype=dtype, order="F")

@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH, inline=INLINE)
def get_value_from_column(i, bitarray, bitmask, n_values_in_word, n_bits):
    word, pos_in_word = i // n_values_in_word, i % n_values_in_word
    return (bitarray[word] & (bitmask << ((n_values_in_word - pos_in_word - 1) * n_bits))) >> ((n_values_in_word - pos_in_word - 1) * n_bits)

@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH, cache=CACHE)
def _features_bitarray_to_array(features_bitarray, X):
    for j in range(features_bitarray.n_features):
        bitarray_j = features_bitarray.bitarray[features_bitarray.offsets[j]:features_bitarray.offsets[j + 1]]
        for i in range(features_bitarray.n_samples):
            X[i, j] = get_value_from_column(i, bitarray_j, features_bitarray.bitmasks[j], features_bitarray.n_values_in_words[j], features_bitarray.n_bits[j])

@jit(nopython=NOPYTHON, nogil=NOGIL, boundscheck=BOUNDSCHECK, fastmath=FASTMATH, inline=INLINE)
def get_value(features_bitarray, i, j):
    return get_value_from_column(i, features_bitarray.bitarray[features_bitarray.offsets[j]:features_bitarray.offsets[j + 1]], features_bitarray.bitmasks[j], features_bitarray.n_values_in_words[j], features_bitarray.n_bits[j])

def features_bitarray_to_array(features_bitarray):
    X = _get_empty_matrix(features_bitarray.n_samples, features_bitarray.n_features, features_bitarray.max_values.max())
    _features_bitarray_to_array(features_bitarray, X)
    return X
