
from numba import njit
from heapq import heappush
from collections import namedtuple
from numpy.random import randint
from numba.experimental import jitclass
from time import time
from numba import uint32

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