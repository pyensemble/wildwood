import sys
import numpy as np
import pandas as p

import pytest


x = np.array([2, 7, 2, 10], dtype=np.int)
test = x.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
print(x.dtype, test)


x = np.array([2, 7, 2, 10], dtype=np.uint8)
test = x.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
print(x.dtype, test)


x = np.array([2, 7, 2, 10], dtype=np.int32)
test = x.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
print(x.dtype, test)


x = np.array([2, 7, 2, 10], dtype=np.uint64)
test = x.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
print(x.dtype, test)


x = np.array([2, 7, 2, 10], dtype=np.int16)
test = x.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
print(x.dtype, test)


x = np.array([2, 7, 2, 10], dtype=np.int64)
test = x.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
print(x.dtype, test)

x = np.array([2, 7, 2, 10], dtype=np.intp)
test = x.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
print(x.dtype, test)


x = np.array([2, 7, 2, 10], dtype=np.uintp)
test = x.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
print(x.dtype, test)
