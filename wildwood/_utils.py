import numpy as np
from numba import boolean, float32, float64, intp, int32, uint32, from_dtype
from numba import njit as njit_
from numba.experimental import jitclass as jitclass_
from math import log2

# Lazy to change everywhere when numba people decide that jitclass is not
# experimental anymore
jitclass = jitclass_
njit = njit_(fastmath=True, nogil=True, cache=True)

BOOL_t = boolean
NP_BOOL_t = np.bool
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
EPSILON = np.finfo("double").eps

SIZE_MAX = np.iinfo(NP_SIZE_t).max


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
def resize3d(a, new_size, zeros=False):
    d0, d1, d2 = a.shape
    if zeros:
        new = np.zeros((new_size, d1, d2), a.dtype)
    else:
        new = np.empty((new_size, d1, d2), a.dtype)
    new[:d0, :, :] = a
    return new


from collections import namedtuple


from numba.typed import List

# StackRecord = namedtuple("StackRecord", ["start", "end", "depth", "parent",
#                                          "is_left", "impurity", "n_constant_features"])
# A record on the stack for depth-first tree growing


# NP_NODE_t = np.dtype(node_dtype)
# NODE_t = numba.from_dtype(NP_NODE_t)
#
# spec_stack_record_dtype = [
#     ("left_child", NP_SIZE_t),
#     ("right_child", NP_SIZE_t),
#     ("feature", NP_SIZE_t),
#     ("threshold", NP_DOUBLE_t),
#     ("impurity", NP_DOUBLE_t),
#     ("n_node_samples", NP_SIZE_t),
#     ("weighted_n_node_samples", NP_DOUBLE_t)
# ]
#
#
# NP_NODE_t = np.dtype(node_dtype)


spec_stack_record_dtype = [
    ("start", NP_SIZE_t),
    ("end", NP_SIZE_t),
    ("depth", NP_SIZE_t),
    ("parent", NP_SIZE_t),
    ("is_left", NP_BOOL_t),
    ("impurity", NP_DOUBLE_t),
    ("n_constant_features", NP_SIZE_t),
]

NP_STACK_RECORD_t = np.dtype(spec_stack_record_dtype)
STACK_RECORD_t = from_dtype(NP_STACK_RECORD_t)


spec_stack = [("capacity", SIZE_t), ("top", SIZE_t), ("stack_", STACK_RECORD_t[::1])]


@jitclass(spec_stack)
class Stack(object):
    # cdef class Stack:
    #     """A LIFO data structure.
    #
    #     Attributes
    #     ----------
    #     capacity : SIZE_t
    #         The elements the stack can hold; if more added then ``self.stack_``
    #         needs to be resized.
    #
    #     top : SIZE_t
    #         The number of elements currently on the stack.
    #
    #     stack : StackRecord pointer
    #         The stack of records (upward in the stack corresponds to the right).
    #     """
    def __init__(self, capacity):
        #     def __cinit__(self, SIZE_t capacity):
        #         self.capacity = capacity
        #         self.top = 0
        #         self.stack_ = <StackRecord*> malloc(capacity * sizeof(StackRecord))
        self.capacity = capacity
        self.top = 0
        self.stack_ = np.empty(capacity, dtype=NP_STACK_RECORD_t)


@njit
def stack_push(
    stack, start, end, depth, parent, is_left, impurity, n_constant_features
):
    #     cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
    #                   bint is_left, double impurity,
    #                   SIZE_t n_constant_features) nogil except -1:
    #         """Push a new element onto the stack.
    #
    #         Return -1 in case of failure to allocate memory (and raise MemoryError)
    #         or 0 otherwise.
    #         """
    #         cdef SIZE_t top = self.top
    #         cdef StackRecord* stack = NULL
    #
    #         # Resize if capacity not sufficient
    #         if top >= self.capacity:
    #             self.capacity *= 2
    #             # Since safe_realloc can raise MemoryError, use `except -1`
    #             safe_realloc(&self.stack_, self.capacity)
    #
    #         stack = self.stack_
    #         stack[top].start = start
    #         stack[top].end = end
    #         stack[top].depth = depth
    #         stack[top].parent = parent
    #         stack[top].is_left = is_left
    #         stack[top].impurity = impurity
    #         stack[top].n_constant_features = n_constant_features
    #
    #         # Increment stack pointer
    #         self.top = top + 1
    #         return 0

    top = stack.top
    stack_ = stack.stack_
    # cdef StackRecord* stack = NULL

    # Resize if capacity not sufficient
    if top >= stack.capacity:
        stack.capacity = 2 * stack.capacity
        # Since safe_realloc can raise MemoryError, use `except -1`
        # safe_realloc( & self.stack_, self.capacity)
        stack.stack_ = resize(stack_, stack.capacity)

    stack_top = stack.stack_[top]
    stack_top["start"] = start
    stack_top["end"] = end
    stack_top["depth"] = depth
    stack_top["parent"] = parent
    stack_top["is_left"] = is_left
    stack_top["impurity"] = impurity
    stack_top["n_constant_features"] = n_constant_features

    # We have one more record in the stack
    stack.top = top + 1
    return 0


@njit
def stack_is_empty(stack):
    #     cdef bint is_empty(self) nogil:
    #         return self.top <= 0
    return stack.top <= 0


@njit
def stack_pop(stack):
    #     cdef int pop(self, StackRecord* res) nogil:
    #         """Remove the top element from the stack and copy to ``res``.
    #
    #         Returns 0 if pop was successful (and ``res`` is set); -1
    #         otherwise.
    #         """
    #         cdef SIZE_t top = self.top
    #         cdef StackRecord* stack = self.stack_
    #
    #         if top <= 0:
    #             return -1
    #
    #         res[0] = stack[top - 1]
    #         self.top = top - 1
    #
    #         return 0
    top = stack.top
    # cdef StackRecord* stack = self.stack_
    stack_ = stack.stack_

    # TODO: en fait ce return on ne s'en sert pas apparemment...
    # if top <= 0:
    #     return -1

    stack_record = stack_[top - 1]
    # res[0] = stack[top - 1]
    stack.top = top - 1
    # return 0
    return stack_record


def print_stack(stack):
    s = "Stack("
    s += "capacity={capacity}".format(capacity=stack.capacity)
    s += ", top={top}".format(top=stack.top)
    s += ")"
    print(s)


def get_records(stack):
    import pandas as pd

    nodes = stack.nodes
    columns = [col_name for col_name, _ in spec_stack_record_dtype]
    # columns = ["left_child"]

    return pd.DataFrame.from_records(
        (
            tuple(node[col] for col in columns)
            for i, node in enumerate(nodes)
            if i < stack.node_count
        ),
        columns=columns,
    )


stack = Stack(3)

print_stack(stack)

print()
