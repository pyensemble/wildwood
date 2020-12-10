
from numba import njit
from heapq import heappush
from collections import namedtuple
from numpy.random import randint
from numba.experimental import jitclass
from numba import from_dtype
from numba.typed import List
from time import time
from numba import uint32

from wildwood._splitter import sort

from wildwood._utils import SIZE_t, DOUBLE_t, NP_SIZE_t, NP_DOUBLE_t

import numpy as np

# # Node = namedtuple("Node", ["left", "right", "parent"])
#

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



spec = [(
    ("a", uint32)
)]
@jitclass(spec)
class A(object):

    def __init__(self, a):
        self.a = a


@njit
def main():

    a = A(42)
    b = A(13)
    print(a.a, b.a)  # 42 13

    a = b
    print(a.a, b.a)  # 13 13

    b.a = 27
    print(a.a, b.a)  # 13 27

main()


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

