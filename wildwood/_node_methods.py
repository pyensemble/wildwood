# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains some functions performing computations locally in nodes
"""

from math import log
import numpy as np
from numba import from_dtype, njit, boolean, uint8, intp, uintp, float32
from numba.experimental import jitclass

from ._node import NodeContextType
from ._tree import Tr
# from ._tree import TreeType


