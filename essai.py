import sys
import numpy as np
import pandas as p

import pytest

from numba import jit, uint8, boolean, intp, uintp



@jit(nopython=True, nogil=True, boundscheck=True)
def check_node(a, b):
    assert a == b



@pytest.mark.parametrize(
    "a, b", [(1, 1), (2, 2), (1, 3)]
)
def test_stuff(a, b):
    check_node(a, b)
    
