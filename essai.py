
import numpy as np
from numba import uintp, jit, void, optional


from numpy.random import randint


@jit(signature=["void(uintp)", "void(uintp, optional(uintp))"], nopython=True,
     nogil=True)
# @njit
def f(x, y=None):
    if y is None:
        print("y is None")
    else:
        print("y is not None")
        print(x + y)


@jit(void(uintp[::1], uintp[::1]), nopython=True, nogil=True, locals={"i": uintp,
                                                                  "j": uintp})
def sample_without_replacement(pool, out):
    """Sample integers without replacement.

    Select n_samples integers from the set [0, n_population) without
    replacement.

    Time complexity: O(n_population +  O(np.random.randint) * n_samples)

    Space complexity of O(n_population + n_samples).


    Parameters
    ----------
    n_population : int
        The size of the set to sample from.

    n_samples : int
        The number of integer to sample.

    random_state : int, RandomState instance or None, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    out : ndarray of shape (n_samples,)
        The sampled subsets of integer.
    """

    n_samples = out.shape[0]
    n_population = pool.shape[0]

    # rng = check_random_state(random_state)
    # rng_randint = rng.randint

    # Initialize the pool
    for i in range(n_population):
        pool[i] = i

    for i in range(n_samples):
        j = randint(n_population - i)
        out[i] = pool[j]
        pool[j] = pool[n_population - i - 1]


@jit(nopython=True, nogil=True)
def main():
    pool = np.arange(0, 11).astype(np.uintp)
    out = np.arange(0, 3).astype(np.uintp)
    print(pool)

    for _ in range(10):
        print(32 * "=")
        sample_without_replacement(pool, out)
        print(pool)
        print(out)


main()

# np_record = np.dtype([("a", np_float32), ("b", np_size_t)])
#
#
# nb_record = from_dtype(np_record)
#
#
# spec_record = [
#     ("a", nb_float32),
#     ("b", nb_size_t)
# ]
#
#
# @jitclass(spec_record)
# class Record(object):
#
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#
# @njit
# def set_record(records, idx, a, b):
#     record = records[idx]
#     record["a"] = a
#     record["b"] = b
#
#
# @njit
# def get_record(records, idx):
#     record = records[idx]
#     return record["a"], record["b"]
#
#
# @njit
# def main():
#     records = np.empty((10,), dtype=np_record)
#     set_record(records, 0, 3.14, 17)
#     record = get_record(records, 0)
#     print(record)
#

# main()


from numba import float32



# import numba
#
# @numba.jit(nopython=True, nogil=True, fastmath=True, locals={"x": numba.float32})
# def main2():
#     x = np.nan
#     print(x)
#     print(np.isnan(x))
#
# main2()






#
# from wildwood._tree import Stack, print_stack
#
# stack = Stack(3)
#
# print_stack(stack)
#
# print()


# @njit
# def myprint():
#
#     print("Split({gain_proxy}".format(gain_proxy=2e-3))
#
#
# myprint()

from cffi import FFI

from numba.core.typing import cffi_utils



# file "example_build.py"

# from cffi import FFI


# ffibuilder = FFI()
#
# ffibuilder.cdef("int foo(int *, int *, int);")
#
# ffibuilder.set_source("_example",
# r"""
#     static int foo(int *buffer_in, int *buffer_out, int x)
#     {
#         /* some algorithm that is seriously faster in C than in Python */
#     }
# """)
#
#
# if __name__ == "__main__":
#     ffibuilder.compile(verbose=True)
#
#
# # file "example.py"
#
# import _ffi, lib
#
# from _example import ffi, lib
#
# buffer_in = ffi.new("int[]", 1000)
# # initialize buffer_in here...
#
# # easier to do all buffer allocations in Python and pass them to C,
# # even for output-only arguments
# buffer_out = ffi.new("int[]", 1000)
#
# result = lib.foo(buffer_in, buffer_out, 1000)


from numba import cfunc, types, carray

# c_sig = types.void(types.CPointer(types.double),
#                    types.CPointer(types.double),
#                    types.intc, types.intc)
#
# @cfunc(c_sig)
# def my_callback(in_, out, m, n):
#     in_array = carray(in_, (m, n))
#     out_array = carray(out, (m, n))
#     for i in range(m):
#         for j in range(n):
#             out_array[i, j] = 2 * in_array[i, j]

# c_sig = types.double(types.CPointer(types.double), types.intp)
#
#
# @cfunc(c_sig)
# def my_callback(in_, n):
#     # in_array = carray(in_, (m, n))
#     # out_array = carray(out, (m, n))
#     s = 0
#     i = 0
#
#     types.voidptr
#
#     while i != n:
#         s += in_[i]
#         i += 1
#         # Pointer arithmetics slang
#         in_ = in_ + types.CPointer(types.double)(1)
#
#     return s
#


# Get the function signature from *my_func*
# sig = cffi_utils.map_type(ffi.typeof('my_func'), use_record_dtype=True)

# Make the cfunc
# from numba import cfunc, carray
#
# @cfunc(sig)
# def foo(ptr, n):
#    base = carray(ptr, n)  # view pointer as an array of my_struct
#    tmp = 0
#    for i in range(n):
#       tmp += base[i].i1 * base[i].f2 / base[i].d3
#       tmp += base[i].af4.sum()  # nested arrays are like normal numpy array
#    return tmp


# @njit
# def main():
#     x = np.linspace(0.0, 10.0, 11)


# print(foo(x, 11))

# # # Node = namedtuple("Node", ["left", "right", "parent"])
# #
#
# spec = [("X", DOUBLE_t[:, ::1])]
#
#
# @jitclass(spec)
# class C(object):
#     def __init__(self, n_samples, n_features):
#         self.X = np.ones((n_samples, n_features))
#
#
# @njit
# def sum1(c):
#     n_samples, n_features = c.X.shape
#     s = 0.0
#     for i in range(n_samples):
#         for j in range(n_features):
#             s += c.X[i, j]
#
#     return s
#
#
# @njit
# def sum2(X):
#     n_samples, n_features = X.shape
#     s = 0.0
#     for i in range(n_samples):
#         for j in range(n_features):
#             s += X[i, j]
#
#     return s
#
#
# n_samples = 3
# n_features = 2
#
# c = C(n_samples, n_features)
# X = np.ones((n_samples, n_features), dtype=NP_DOUBLE_t)
# X = np.ascontiguousarray(X)
# u = sum1(c)
# v = sum2(X)
#
# n_samples = 500_000
# n_features = 5000
# X = np.ones((n_samples, n_features), dtype=NP_DOUBLE_t)
# X = np.ascontiguousarray(X)
#
# c = C(n_samples, n_features)
#
# tic = time()
# u = sum1(c)
# toc = time()
# print(toc - tic)
#
#
# tic = time()
# v = sum2(X)
# toc = time()
# print(toc - tic)


# cdef struct StackRecord:
#     SIZE_t start
#     SIZE_t end
#     SIZE_t depth
#     SIZE_t parent
#     bint is_left
#     double impurity
#     SIZE_t n_constant_features


#
# @njit
# def numpy_sort(Xf, samples, n):
#     idx = np.argsort(Xf)
#     Xf = Xf[idx]
#     samples = samples[idx]
#
#
#
# @njit
# def numba_sort(Xf, samples, n):
#     sort(Xf, samples, n)
#
#
# @njit
# def compile():
#     print('JIT Compile')
#     n = 10
#     Xf = np.random.randn(n)
#     samples = np.arange(0, n)
#     numpy_sort(Xf, samples, n)
#     numba_sort(Xf, samples, n)
#
#
# compile()
#
#
# print("One big")
# n_repeat = 5
# n = 10_000_000
#
# total_time_numba = 0
# total_time_numpy = 0
#
# for i in range(n_repeat):
#     # Generate data
#     Xf = np.random.randn(n)
#     samples = np.arange(0, n)
#
#     # Time numpy
#     tic = time()
#     numpy_sort(Xf, samples, n)
#     toc = time()
#     total_time_numpy += toc - tic
#
#     # Time numba
#     tic = time()
#     numba_sort(Xf, samples, n)
#     toc = time()
#     total_time_numba += toc - tic
#
#
# print("numpy: ", total_time_numpy)
# print("numba: ", total_time_numba)
#
#
# print("Many small")
# n_repeat = 5_000
# n = 10_000
#
# total_time_numba = 0
# total_time_numpy = 0
#
# for i in range(n_repeat):
#     # Generate data
#     Xf = np.random.randn(n)
#     samples = np.arange(0, n)
#
#     # Time numpy
#     tic = time()
#     numpy_sort(Xf, samples, n)
#     toc = time()
#     total_time_numpy += toc - tic
#
#     # Time numba
#     tic = time()
#     numba_sort(Xf, samples, n)
#     toc = time()
#     total_time_numba += toc - tic
#
#
# print("numpy: ", total_time_numpy)
# print("numba: ", total_time_numba)


# print("Xf: ", Xf)

# print("samples: ", samples)
# sort(Xf, samples, n)

# print("Xf: ", Xf)
# print("samples: ", samples)


#
# @njit
# def main():
#
#
#
#     l = List()
#     l.append(42)
#     l.append(3.14)
#     return l
#
# l = main()
# print(l)


# spec = [(("a", uint32))]
#
#
# @jitclass(spec)
# class A(object):
#     def __init__(self, a):
#         self.a = a
#


# @njit(DOUBLE_t(DOUBLE_t[:, ::1]))
# def sum1(X):
#     n_samples, n_features = X.shape
#     s = 0.0
#     for i in range(n_samples):
#         for j in range(n_features):
#             s += X[i, j]
#
#     return s
#
#
# @njit(DOUBLE_t(DOUBLE_t[:, ::1]))
# def sum2(X):
#     n_samples, n_features = X.shape
#     s = 0.0
#     for j in range(n_features):
#         for i in range(n_samples):
#             s += X[i, j]
#
#     return s
#
#
# n_samples = 3
# n_features = 2
# X = np.ones((3, 2), dtype=NP_DOUBLE_t)
# X = np.ascontiguousarray(X)
# # X = np.asfortranarray(X)
# u = sum1(X)
# v = sum2(X)
#
# n_samples = 500_000
# n_features = 1000
# X = np.ones((n_samples, n_features), dtype=NP_DOUBLE_t)
# X = np.ascontiguousarray(X)
#
# tic = time()
# u = sum1(X)
# toc = time()
# print(toc - tic)
#
#
# tic = time()
# v = sum2(X)
# toc = time()
# print(toc - tic)


# a = A(42)
# b = A(13)
# print(a.a, b.a)  # 42 13
#
# a = b
# print(a.a, b.a)  # 13 13
#
# b.a = 27
# print(a.a, b.a)  # 13 27


#
# @jitclass(spec)
# class B(object):
#
#     def __init__(self, a):
#         self.a = a
#
#
# @njit
# def update(obj, a):
#     obj.a = a
#
#
# @njit
# def truc():
#     return 1.2, 3.2
#
#
# @njit
# def main():
#     # a = A(2)
#     # b = B(1)
#     # update(a, 123)
#     # update(b, 42)
#     # print(a.a, b.a)
#     a, b = truc()
#     print(a, b)
#
# main()
#
#
#
# from wildwood._utils import DTYPE_t, NP_DTYPE_t, DOUBLE_t, NP_DOUBLE_t, SIZE_t, \
#     NP_SIZE_t, \
#     INT32_t, NP_UINT32_t, jitclass, njit, get_numba_type
#
#
# spec_node = [
#     ("left_child", SIZE_t),
#     ("right_child", SIZE_t),
#     ("feature", SIZE_t),
#     ("threshold", DOUBLE_t),
#     ("impurity", DOUBLE_t),
#     ("n_node_samples", SIZE_t),
#     ("weighted_n_node_samples", DOUBLE_t)
# ]
#
# node_dtype = [
#     ("left_child", NP_SIZE_t),
#     ("right_child", NP_SIZE_t),
#     ("feature", NP_SIZE_t),
#     ("threshold", NP_DOUBLE_t),
#     ("impurity", NP_DOUBLE_t),
#     ("n_node_samples", NP_SIZE_t),
#     ("weighted_n_node_samples", NP_DOUBLE_t)
# ]
#
# @jitclass(spec_node)
# class Node(object):
#     # cdef struct Node:
#     #     # Base storage structure for the nodes in a Tree object
#     #
#     #     SIZE_t left_child                    # id of the left child of the node
#     #     SIZE_t right_child                   # id of the right child of the node
#     #     SIZE_t feature                       # Feature used for splitting the node
#     #     DOUBLE_t threshold                   # Threshold value at the node
#     #     DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
#     #     SIZE_t n_node_samples                # Number of samples at the node
#     #     DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node
#
#     def __init__(self, left_child, right_child, feature, threshold, impurity,
#                  n_node_samples, weighted_n_node_samples):
#         self.left_child = left_child
#         self.right_child = right_child
#         self.feature = feature
#         self.threshold = threshold
#         self.impurity = impurity
#         self.n_node_samples = n_node_samples
#         self.weighted_n_node_samples = weighted_n_node_samples
#
#
#
#
# @njit
# def set_node(tree, idx, node):
#     # It's a dtype
#     nodes = tree.nodes
#     node_dtype = nodes[idx]
#     node_dtype["left_child"] = node.left_child
#     node_dtype["right_child"] = node.right_child
#     node_dtype["feature"] = node.feature
#     node_dtype["threshold"] = node.threshold
#     node_dtype["impurity"] = node.impurity
#     node_dtype["n_node_samples"] = node.n_node_samples
#     node_dtype["weighted_n_node_samples"] = node.weighted_n_node_samples
#
#
# @njit
# def get_node(tree, idx):
#     # It's a jitclass object
#     nodes = tree.nodes
#     node = nodes[idx]
#     return Node(
#         node["left_child"],
#         node["right_child"],
#         node["feature"],
#         node["threshold"],
#         node["impurity"],
#         node["n_node_samples"],
#         node["weighted_n_node_samples"]
#     )
#
#
# # nodes = np.empty(n_nodes, dtype=node_dtype)
#
# NP_NODE_t = np.dtype(node_dtype)
# NODE_t = from_dtype(NP_NODE_t)
#
#
#






# # >>> struct_dtype = np.dtype([('row', np.float64), ('col', np.float64)])
# # >>> ty = numba.from_dtype(struct_dtype)
# # >>> ty
# # Record([('row', '<f8'), ('col', '<f8')])
# # >>> ty[:, :]
# # unaligned array(Record([('row', '<f8'), ('col', '<f8')]), 2d, A)
# spec_tree = [
#     ("nodes", NODE_t[::1])
# ]
#
# @jitclass(spec_tree)
# class Tree(object):
#
#     def __init__(self, n_nodes):
#         self.nodes = np.empty(0, dtype=NP_NODE_t)
#
#
# # node_dtype = [spec_node_dtype]
#
#
#
# @njit
# def main2():
#     n_nodes = 10
#
#     tree = Tree(n_nodes)
#
#     for i in range(n_nodes):
#         node = Node(1, 2, 3.0, 4.0, 5.0, 6.0, 7.0)
#         set_node(tree, i, node)
#
#     for i in range(n_nodes):
#         node = get_node(tree, i)
#
#
#
#     # nodes[0]["feature"] = 123
#
#     # node = get_node(nodes, 2)
#     # node = Node(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
#
#     # cdef Node dummy;
#     # NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype
#
#     # nodes = np.empty(10, dtype=node_dtype)
#     # print(node.left_child, node.right_child)
#
# tic = time()
# main2()
# toc = time()
#
# print(toc-tic)


# import numpy as np
# from numba import njit
#
# arr = np.array([(1, 2)], dtype=[('a1', 'f8'), ('a2', 'f8')])
# fields_gl = ('a1', 'a2')
#
# @njit
# def get_field_sum(rec):
#     fields_lc = ('a1', 'a2')
#     field_name1 = fields_lc[0]
#     field_name2 = fields_gl[1]
#     return rec[field_name1] + rec[field_name2]
#
# print(get_field_sum(arr[0]))
# returns 3


#
#
# @njit
# def func(heap):
#     a = 1
#     b = 2
#     c = 3
#     node = Node(left=a, right=b, parent=c)
#     heap.append(node)
#
#
# @njit
# def main(n_nodes):
#     heap = [Node(left=1, right=2, parent=3)]
#     for i in range(n_nodes):
#         func(heap)
#
#     truc = 0
#     for i in range(n_nodes):
#         truc += heap[i].left + heap[i].right + heap[i].parent
#
#     # node = heap.pop(0)
#     return truc
#
#     # print(node)
#     # print(heap)
#
#
# n_nodes = 10
# truc = main(n_nodes)
#
# n_nodes = 50_000_000
#
# tic = time()
# main(n_nodes)
# toc = time()
# print(toc - tic)
#
# spec = [
#     ('left', uint32),               # a simple scalar field
#     ('right', uint32),          # an array field
#     ('parent', uint32)
# ]
# @jitclass(spec)
# class TreeNode(object):
#
#     def __init__(self, left, right, parent):
#         self.left = left
#         self.right = right
#         self.parent = parent
#
#
# @njit
# def func2(heap):
#     a = 1
#     b = 2
#     c = 3
#     node = TreeNode(a, b, c)
#     heap.append(node)
#     # heappush(heap, node)
#
#
# @njit
# def main2(n_nodes):
#     heap = [TreeNode(1, 2, 3)]
#     for i in range(n_nodes):
#         func2(heap)
#
#     truc = 0
#     for i in range(n_nodes):
#         truc += heap[i].left + heap[i].right + heap[i].parent
#
#     # node = heap.pop(0)
#     return truc
#
# n_nodes = 10
# truc = main2(n_nodes)
#
# n_nodes = 50_000_000
#
# tic = time()
# main2(n_nodes)
# toc = time()
# print(toc - tic)


# @njit
# def main():
#
#     x = np.empty()
