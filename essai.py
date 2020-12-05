
from numba import njit
from heapq import heappush
from collections import namedtuple
from numpy.random import randint
from numba.experimental import jitclass
from time import time
from numba import uint32

import numpy as np

Node = namedtuple("Node", ["left", "right", "parent"])


spec = [(
    ("a", uint32)
)]
@jitclass(spec)
class A(object):

    def __init__(self, a):
        self.a = a



@jitclass(spec)
class B(object):

    def __init__(self, a):
        self.a = a


@njit
def update(obj, a):
    obj.a = a


@njit
def truc():
    return 1.2, 3.2


@njit
def main():
    # a = A(2)
    # b = B(1)
    # update(a, 123)
    # update(b, 42)
    # print(a.a, b.a)
    a, b = truc()
    print(a, b)

main()


from wildwood._utils import DTYPE_t, NP_DTYPE_t, DOUBLE_t, NP_DOUBLE_t, SIZE_t, \
    NP_SIZE_t, \
    INT32_t, NP_UINT32_t, jitclass, njit, get_numba_type


spec_node = [
    ("left_child", SIZE_t),
    ("right_child", SIZE_t),
    ("feature", SIZE_t),
    ("threshold", DOUBLE_t),
    ("impurity", DOUBLE_t),
    ("n_node_samples", SIZE_t),
    ("weighted_n_node_samples", DOUBLE_t)
]

@jitclass(spec_node)
class Node(object):
    # cdef struct Node:
    #     # Base storage structure for the nodes in a Tree object
    #
    #     SIZE_t left_child                    # id of the left child of the node
    #     SIZE_t right_child                   # id of the right child of the node
    #     SIZE_t feature                       # Feature used for splitting the node
    #     DOUBLE_t threshold                   # Threshold value at the node
    #     DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    #     SIZE_t n_node_samples                # Number of samples at the node
    #     DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node

    def __init__(self, left_child, right_child, feature, threshold, impurity,
                 n_node_samples, weighted_n_node_samples):
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.weighted_n_node_samples = weighted_n_node_samples


@njit
def main2():


    node = Node(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # cdef Node dummy;
    # NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

    nodes = np.array(10, dtype=get_numba_type(Node))
    print(nodes)
    pass


import numpy as np
from numba import njit

arr = np.array([(1, 2)], dtype=[('a1', 'f8'), ('a2', 'f8')])
fields_gl = ('a1', 'a2')

@njit
def get_field_sum(rec):
    fields_lc = ('a1', 'a2')
    field_name1 = fields_lc[0]
    field_name2 = fields_gl[1]
    return rec[field_name1] + rec[field_name2]

print(get_field_sum(arr[0]))
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