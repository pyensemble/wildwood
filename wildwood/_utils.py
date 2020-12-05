
import numpy as np
from numba import float32, float64, intp, int32, uint32
from numba import njit as njit_
from numba.experimental import jitclass as jitclass_

# Lazy to change everywhere when numba people decide that jitclass is not
# experimental anymore
jitclass = jitclass_
njit = njit_


DTYPE_t = float32
NP_DTYPE_t = np.float32
DOUBLE_t = float64
NP_DOUBLE_t = np.float64
SIZE_t = intp
NP_SIZE_t = np.intp
INT32_t = int32
NP_INT32_t = np.int32
UINT32_t = uint32
NP_UINT32_t = np.uint32

INFINITY = np.inf
EPSILON = np.finfo('double').eps


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
