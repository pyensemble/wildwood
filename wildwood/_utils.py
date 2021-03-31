from math import log, exp
import numpy as np
from numpy.random import randint
from numba import (
    jit,
    void,
    boolean,
    float32,
    float64,
    uint8,
    uintp,
    intp,
    int32,
    uint8,
    uint32,
    from_dtype,
)
from numba import njit as njit_
from numba.experimental import jitclass as jitclass_
from math import log2

# Lazy to change everywhere when numba people decide that jitclass is not
# experimental anymore
jitclass = jitclass_
njit = njit_(fastmath=False, nogil=True, cache=False, boundscheck=False)


# We centralize below the definition of all the numba and numpy dtypes we'll need
# nb_bool = boolean
# np_bool = np.bool
# nb_uint8 = uint8
# np_uint8 = np.uint8
# nb_float32 = float32
# np_float32 = np.float32
# nb_float64 = float64
# np_float64 = np.float64
#
# np_size_t = np.uintp
# nb_size_t = uintp
# np_ssize_t = np.intp
# nb_ssize_t = intp
# nb_int32 = int32
# np_int32 = np.int32
# nb_uint32 = uint32
# np_uint32 = np.uint32

# Some useful constants
# infinity = np.inf
# epsilon = np.finfo("double").eps
# max_int32 = np.iinfo(np.int32).max
# max_size_t = np.iinfo(np_size_t).max
# max_ssize_t = np.iinfo(np_ssize_t).max


def get_numba_type(class_):
    """Gives the numba type of an object is numba.jit decorators are enabled and None
    otherwise. This helps to get correct coverage of the code

    Parameters
    ----------
    class_ : `object`
        A class

    Returns
    -------
    output : `object`
        A numba type of None
    """
    class_type = getattr(class_, "class_type", None)
    if class_type is None:
        return class_type
    else:
        return class_type.instance_type


# TODO: specify signature for the resize function, it it used to resize arrays of
#  records, node_dtype and floats. What about the return type ?


@jit(
    nopython=True,
    nogil=True,
    locals={"new_size": uintp, "d0": uintp, "d1": uintp, "d2": uintp},
)
def resize(a, new_size, zeros=False):
    ndim = a.ndim
    if ndim == 1:
        if zeros:
            new = np.zeros(new_size, a.dtype)
        else:
            new = np.empty(new_size, a.dtype)

        new[: a.size] = a
        return new
    elif ndim == 2:
        d0, d1 = a.shape
        if zeros:
            new = np.zeros((new_size, d1), a.dtype)
        else:
            new = np.empty((new_size, d1), a.dtype)
        new[:d0, :] = a
        return new
    elif ndim == 3:
        d0, d1, d2 = a.shape
        if zeros:
            new = np.zeros((new_size, d1, d2), a.dtype)
        else:
            new = np.empty((new_size, d1, d2), a.dtype)
        new[:d0, :, :] = a
        return new
    else:
        raise ValueError("ndim can only be 1, 2 or 3")


@jit(float32(float32, float32), nogil=True, nopython=True, fastmath=True)
def log_sum_2_exp(a, b):
    """Computation of log( (e^a + e^b) / 2) in an overflow-proof way

    Parameters
    ----------
    a : float
        First number

    b : float32
        Second number

    Returns
    -------
    output : float
        Value of log( (e^a + e^b) / 2) for the given a and b
    """
    # TODO: if |a - b| > 50 skip
    if a > b:
        return a + log((1 + exp(b - a)) / 2)
    else:
        return b + log((1 + exp(a - b)) / 2)


def get_type(class_):
    """Gives the numba type of an object if numba.jit decorators are enabled and None
    otherwise. This helps to get correct coverage of the code

    Parameters
    ----------
    class_ : `object`
        A class

    Returns
    -------
    output : `object`
        A numba type of None
    """
    class_type = getattr(class_, "class_type", None)
    if class_type is None:
        return lambda *_: None
    else:
        return class_type.instance_type


@jit(
    void(uintp[:], uintp[:]),
    nopython=True,
    nogil=True,
    locals={"n_samples": uintp, "population_size": uintp, "i": uintp, "j": uintp},
)
def sample_without_replacement(pool, out):
    """Samples integers without replacement from pool into out inplace.

    Parameters
    ----------
    pool : ndarray of size population_size
        The array of integers to sample from (it containing [0, ..., n_samples-1]

    out : ndarray of size n_samples
        The sampled subsets of integer
    """
    # TODO: faudra verifier que les arbres n'utilisent pas tous les meme
    #  permutations... faudra peut etre avoir a gerer le random_state ici ?
    # We sample n_samples elements from the pool
    n_samples = out.shape[0]
    population_size = pool.shape[0]
    # Initialize the pool
    for i in range(population_size):
        pool[i] = i

    for i in range(n_samples):
        j = randint(population_size - i)
        out[i] = pool[j]
        pool[j] = pool[population_size - i - 1]
