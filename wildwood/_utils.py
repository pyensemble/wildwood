import numpy as np
from numba import (
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
nb_bool = boolean
np_bool = np.bool
nb_uint8 = uint8
np_uint8 = np.uint8
nb_float32 = float32
np_float32 = np.float32
nb_float64 = float64
np_float64 = np.float64
np_size_t = np.uintp
nb_size_t = uintp
nb_int32 = int32
np_int32 = np.int32
nb_uint32 = uint32
np_uint32 = np.uint32

# Some useful constants
infinity = np.inf
epsilon = np.finfo("double").eps
max_int32 = np.iinfo(np.int32).max
max_size_t = np.iinfo(np_size_t).max


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


@njit
def resize(a, new_size, zeros=False):
    if zeros:
        new = np.zeros(new_size, a.dtype)
    else:
        new = np.empty(new_size, a.dtype)
    new[: a.size] = a
    return new


@njit
def resize2d(a, new_size, zeros=False):
    d0, d1 = a.shape
    if zeros:
        new = np.zeros((new_size, d1), a.dtype)
    else:
        new = np.empty((new_size, d1), a.dtype)
    new[:d0, :] = a
    return new


@njit
def resize3d(a, new_size, zeros=False):
    d0, d1, d2 = a.shape
    if zeros:
        new = np.zeros((new_size, d1, d2), a.dtype)
    else:
        new = np.empty((new_size, d1, d2), a.dtype)
    new[:d0, :, :] = a
    return new
