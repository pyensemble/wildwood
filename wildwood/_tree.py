

# # cython: cdivision=True
# # cython: boundscheck=False
# # cython: wraparound=False
#
# # Authors: Gilles Louppe <g.louppe@gmail.com>
# #          Peter Prettenhofer <peter.prettenhofer@gmail.com>
# #          Brian Holt <bdholt1@gmail.com>
# #          Noel Dawe <noel@dawe.me>
# #          Satrajit Gosh <satrajit.ghosh@gmail.com>
# #          Lars Buitinck
# #          Arnaud Joly <arnaud.v.joly@gmail.com>
# #          Joel Nothman <joel.nothman@gmail.com>
# #          Fares Hedayati <fares.hedayati@gmail.com>
# #          Jacob Schreiber <jmschreiber91@gmail.com>
# #          Nelson Liu <nelson@nelsonliu.me>
# #
# # License: BSD 3 clause


# See _tree.pyx for details.

import numpy as np

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix


import numpy as np
from ._utils import DTYPE_t, NP_DTYPE_t, DOUBLE_t, NP_DOUBLE_t, SIZE_t, NP_SIZE_t, \
    INT32_t, NP_UINT32_t, SIZE_MAX, jitclass, njit, get_numba_type, resize, INFINITY,\
    Stack, stack_push, stack_pop, stack_is_empty, EPSILON, resize3d


from ._splitter import splitter_init, splitter_node_reset, spec_split_record, \
    splitter_node_value, BestSplitter, best_splitter_node_split, SplitRecord, \
    best_splitter_init, gini_node_impurity, gini_children_impurity

import numba


spec_node = [
    ("left_child", SIZE_t),
    ("right_child", SIZE_t),
    ("feature", SIZE_t),
    ("threshold", DOUBLE_t),
    ("impurity", DOUBLE_t),
    ("n_node_samples", SIZE_t),
    ("weighted_n_node_samples", DOUBLE_t)
]

# A numpy dtype to save a node in a numpy array
node_dtype = [
    ("left_child", NP_SIZE_t),
    ("right_child", NP_SIZE_t),
    ("feature", NP_SIZE_t),
    ("threshold", NP_DOUBLE_t),
    ("impurity", NP_DOUBLE_t),
    ("n_node_samples", NP_SIZE_t),
    ("weighted_n_node_samples", NP_DOUBLE_t)
]

NP_NODE_t = np.dtype(node_dtype)
NODE_t = numba.from_dtype(NP_NODE_t)


@njit
def set_node(nodes, idx, node):
    # It's a dtype
    node_dtype = nodes[idx]
    node_dtype["left_child"] = node.left_child
    node_dtype["right_child"] = node.right_child
    node_dtype["feature"] = node.feature
    node_dtype["threshold"] = node.threshold
    node_dtype["impurity"] = node.impurity
    node_dtype["n_node_samples"] = node.n_node_samples
    node_dtype["weighted_n_node_samples"] = node.weighted_n_node_samples


@njit
def get_node(nodes, idx):
    # It's a jitclass object
    node = nodes[idx]
    return Node(
        node["left_child"],
        node["right_child"],
        node["feature"],
        node["threshold"],
        node["impurity"],
        node["n_node_samples"],
        node["weighted_n_node_samples"]
    )


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


# cdef class Tree:
#     # The Tree object is a binary tree structure constructed by the
#     # TreeBuilder. The tree structure is used for predictions and
#     # feature importances.
#
#     # Input/Output layout
#     cdef public SIZE_t n_features        # Number of features in X
#     cdef SIZE_t* n_classes               # Number of classes in y[:, k]
#     cdef public SIZE_t n_outputs         # Number of outputs in y
#     cdef public SIZE_t max_n_classes     # max(n_classes)
#
#     # Inner structures: values are stored separately from node structure,
#     # since size is determined at runtime.
#     cdef public SIZE_t max_depth         # Max depth of the tree
#     cdef public SIZE_t node_count        # Counter for node IDs
#     cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
#     cdef Node* nodes                     # Array of nodes
#     cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
#     cdef SIZE_t value_stride             # = n_outputs * max_n_classes

# =============================================================================
# Tree builder
# =============================================================================

spec_tree_builder = [
    ("splitter", get_numba_type(BestSplitter)),
    ("min_samples_split", SIZE_t),
    ("min_samples_leaf", SIZE_t),
    ("min_weight_leaf", DOUBLE_t),
    ("max_depth", SIZE_t),
    ("min_impurity_split", DOUBLE_t),
    ("min_impurity_decrease", DOUBLE_t)
]

@jitclass
class TreeBuilder(object):
    # cdef class TreeBuilder:
    #     # The TreeBuilder recursively builds a Tree object from training samples,
    #     # using a Splitter object for splitting internal nodes and assigning
    #     # values to leaves.
    #     #
    #     # This class controls the various stopping criteria and the node splitting
    #     # evaluation order, e.g. depth-first or best-first.
    #
    #     cdef Splitter splitter              # Splitting algorithm
    #
    #     cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    #     cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
    #     cdef double min_weight_leaf         # Minimum weight in a leaf
    #     cdef SIZE_t max_depth               # Maximal tree depth
    #     cdef double min_impurity_split
    #     cdef double min_impurity_decrease   # Impurity threshold for early stopping

    def __init__(self):
        pass


# @njit
# def tree_builder_build(tree_builder, tree, X, y, sample_weight):
#     #cpdef build(self, Tree tree, object X, np.ndarray y, np.ndarray sample_weight=*)
#     pass


# TODO: faudra remettre le check_input, mais pas ici car pas possible dans @njit

#
# @njit
# def tree_builder_check_input(tree, X, y, sample_weight):
#     #     cdef inline _check_input(self, object X, np.ndarray y,
#     #                              np.ndarray sample_weight):
#     #         """Check input dtype, layout and format"""
#     #         if issparse(X):
#     #             X = X.tocsc()
#     #             X.sort_indices()
#     #
#     #             if X.data.dtype != DTYPE:
#     #                 X.data = np.ascontiguousarray(X.data, dtype=DTYPE)
#     #
#     #             if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
#     #                 raise ValueError("No support for np.int64 index based "
#     #                                  "sparse matrices")
#     #
#     #         elif X.dtype != DTYPE:
#     #             # since we have to copy we will make it fortran for efficiency
#     #             X = np.asfortranarray(X, dtype=DTYPE)
#     #
#     #         if y.dtype != DOUBLE or not y.flags.contiguous:
#     #             y = np.ascontiguousarray(y, dtype=DOUBLE)
#     #
#     #         if (sample_weight is not None and
#     #             (sample_weight.dtype != DOUBLE or
#     #             not sample_weight.flags.contiguous)):
#     #                 sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
#     #                                            order="C")
#     #
#     #         return X, y, sample_weight
#
#         """Check input dtype, layout and format"""
#         if issparse(X):
#             X = X.tocsc()
#             X.sort_indices()
#
#             if X.data.dtype != DTYPE:
#                 X.data = np.ascontiguousarray(X.data, dtype=DTYPE)
#
#             if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
#                 raise ValueError("No support for np.int64 index based "
#                                  "sparse matrices")
#
#         elif X.dtype != DTYPE:
#             # since we have to copy we will make it fortran for efficiency
#             X = np.asfortranarray(X, dtype=DTYPE)
#
#         if y.dtype != DOUBLE or not y.flags.contiguous:
#             y = np.ascontiguousarray(y, dtype=DOUBLE)
#
#         if (sample_weight is not None and
#             (sample_weight.dtype != DOUBLE or
#             not sample_weight.flags.contiguous)):
#                 sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
#                                            order="C")
#
#         return X, y, sample_weight



#
# from cpython cimport Py_INCREF, PyObject, PyTypeObject
#
# from libc.stdlib cimport free
# from libc.math cimport fabs
# from libc.string cimport memcpy
# from libc.string cimport memset
# from libc.stdint cimport SIZE_MAX
#
#
# cimport numpy as np
# np.import_array()
#
#
# from ._utils cimport Stack
# from ._utils cimport StackRecord
# from ._utils cimport PriorityHeap
# from ._utils cimport PriorityHeapRecord
# from ._utils cimport safe_realloc
# from ._utils cimport sizet_ptr_to_ndarray
#




# cdef extern from "numpy/arrayobject.h":
#     object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
#                                 int nd, np.npy_intp* dims,
#                                 np.npy_intp* strides,
#                                 void* data, int flags, object obj)
#     int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)
#
# # =============================================================================
# # Types and constants
# # =============================================================================
#
# from numpy import float32 as DTYPE
# from numpy import float64 as DOUBLE
#
# cdef double INFINITY = np.inf
# cdef double EPSILON = np.finfo('double').eps
#
# # Some handy constants (BestFirstTreeBuilder)

IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0

# TREE_LEAF = -1
TREE_LEAF = SIZE_t(-1)
TREE_UNDEFINED = SIZE_t(-2)

# _TREE_UNDEFINED = SIZE_t(TREE_UNDEFINED)
INITIAL_STACK_SIZE = SIZE_t(10)


# # =============================================================================
# # TreeBuilder
# # =============================================================================
#
# cdef class TreeBuilder:
#     """Interface for different tree building strategies."""
#
#     cpdef build(self, Tree tree, object X, np.ndarray y,
#                 np.ndarray sample_weight=None):
#         """Build a decision tree from the training set (X, y)."""
#         pass
#

#
# # Depth first builder ---------------------------------------------------------


@jitclass(spec_tree_builder)
class DepthFirstTreeBuilder(object):
    # cdef class DepthFirstTreeBuilder(TreeBuilder):
    #     """Build a decision tree in depth-first fashion."""

    def __init__(self, splitter, min_samples_split, min_samples_leaf,
                 min_weight_leaf, max_depth, min_impurity_decrease, min_impurity_split):
        #     def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
        #                   SIZE_t min_samples_leaf, double min_weight_leaf,
        #                   SIZE_t max_depth, double min_impurity_decrease,
        #                   double min_impurity_split):
        #         self.splitter = splitter
        #         self.min_samples_split = min_samples_split
        #         self.min_samples_leaf = min_samples_leaf
        #         self.min_weight_leaf = min_weight_leaf
        #         self.max_depth = max_depth
        #         self.min_impurity_decrease = min_impurity_decrease
        #         self.min_impurity_split = min_impurity_split

        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split



@njit
def depth_first_tree_builder_build(builder, tree, X, y, sample_weight):
#     cpdef build(self, Tree tree, object X, np.ndarray y,
#                 np.ndarray sample_weight=None):
#         """Build a decision tree from the training set (X, y)."""
#
#         # check input
#         X, y, sample_weight = self._check_input(X, y, sample_weight)
#
#         cdef DOUBLE_t* sample_weight_ptr = NULL
#         if sample_weight is not None:
#             sample_weight_ptr = <DOUBLE_t*> sample_weight.data
#
#         # Initial capacity
#         cdef int init_capacity
#
#         if tree.max_depth <= 10:
#             init_capacity = (2 ** (tree.max_depth + 1)) - 1
#         else:
#             init_capacity = 2047
#
#
#         # Etape 1 : initialiser l'arbre
#         tree._resize(init_capacity)
#
#         # Parameters
#         cdef Splitter splitter = self.splitter
#         cdef SIZE_t max_depth = self.max_depth
#         cdef SIZE_t min_samples_leaf = self.min_samples_leaf
#         cdef double min_weight_leaf = self.min_weight_leaf
#         cdef SIZE_t min_samples_split = self.min_samples_split
#         cdef double min_impurity_decrease = self.min_impurity_decrease
#         cdef double min_impurity_split = self.min_impurity_split
#
#         # Recursive partition (without actual recursion)
#         splitter.init(X, y, sample_weight_ptr)
#
#         cdef SIZE_t start
#         cdef SIZE_t end
#         cdef SIZE_t depth
#         cdef SIZE_t parent
#         cdef bint is_left
#         cdef SIZE_t n_node_samples = splitter.n_samples
#         cdef double weighted_n_samples = splitter.weighted_n_samples
#         cdef double weighted_n_node_samples
#         cdef SplitRecord split
#         cdef SIZE_t node_id
#
#         cdef double impurity = INFINITY
#         cdef SIZE_t n_constant_features
#         cdef bint is_leaf
#         cdef bint first = 1
#         cdef SIZE_t max_depth_seen = -1
#         cdef int rc = 0
#
#         cdef Stack stack = Stack(INITIAL_STACK_SIZE)
#         cdef StackRecord stack_record
#
#         with nogil:
#             # push root node onto stack
#             rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
#             if rc == -1:
#                 # got return code -1 - out-of-memory
#                 with gil:
#                     raise MemoryError()
#
#             while not stack.is_empty():
#                 stack.pop(&stack_record)
#
#                 start = stack_record.start
#                 end = stack_record.end
#                 depth = stack_record.depth
#                 parent = stack_record.parent
#                 is_left = stack_record.is_left
#                 impurity = stack_record.impurity
#                 n_constant_features = stack_record.n_constant_features
#
#                 n_node_samples = end - start
#                 splitter.node_reset(start, end, &weighted_n_node_samples)
#
#                 is_leaf = (depth >= max_depth or
#                            n_node_samples < min_samples_split or
#                            n_node_samples < 2 * min_samples_leaf or
#                            weighted_n_node_samples < 2 * min_weight_leaf)
#
#                 if first:
#                     impurity = splitter.node_impurity()
#                     first = 0
#
#                 is_leaf = (is_leaf or
#                            (impurity <= min_impurity_split))
#
#                 if not is_leaf:
#                     splitter.node_split(impurity, &split, &n_constant_features)
#                     # If EPSILON=0 in the below comparison, float precision
#                     # issues stop splitting, producing trees that are
#                     # dissimilar to v0.18
#                     is_leaf = (is_leaf or split.pos >= end or
#                                (split.improvement + EPSILON <
#                                 min_impurity_decrease))
#
#                 node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
#                                          split.threshold, impurity, n_node_samples,
#                                          weighted_n_node_samples)
#
#                 if node_id == SIZE_MAX:
#                     rc = -1
#                     break
#
#                 # Store value for all nodes, to facilitate tree/model
#                 # inspection and interpretation
#                 splitter.node_value(tree.value + node_id * tree.value_stride)
#
#                 if not is_leaf:
#                     # Push right child on stack
#                     rc = stack.push(split.pos, end, depth + 1, node_id, 0,
#                                     split.impurity_right, n_constant_features)
#                     if rc == -1:
#                         break
#
#                     # Push left child on stack
#                     rc = stack.push(start, split.pos, depth + 1, node_id, 1,
#                                     split.impurity_left, n_constant_features)
#                     if rc == -1:
#                         break
#
#                 if depth > max_depth_seen:
#                     max_depth_seen = depth
#
#             if rc >= 0:
#                 rc = tree._resize_c(tree.node_count)
#
#             if rc >= 0:
#                 tree.max_depth = max_depth_seen
#         if rc == -1:
#             raise MemoryError()

    """Build a decision tree from the training set (X, y)."""

    # check input
    # TODO: faudra remettre ca
    # X, y, sample_weight = builder._check_input(X, y, sample_weight)

    # This is the output split
    split = SplitRecord()

    # cdef DOUBLE_t* sample_weight_ptr = NULL
    # if sample_weight is not None:
    #     sample_weight_ptr = <DOUBLE_t*> sample_weight.data

    # Initial capacity
    # cdef int init_capacity

    if tree.max_depth <= 10:
        init_capacity = (2 ** (tree.max_depth + 1)) - 1
    else:
        init_capacity = 2047

    # print("tree.max_depth: ", tree.max_depth)

    # tree._resize(init_capacity)
    # print("In depth_first_tree_builder_build calling tree_resize with init_capacity: "
    #       "", init_capacity)
    tree_resize(tree, init_capacity)

    # Parameters
    splitter = builder.splitter
    max_depth = builder.max_depth
    min_samples_leaf = builder.min_samples_leaf
    min_weight_leaf = builder.min_weight_leaf
    min_samples_split = builder.min_samples_split
    min_impurity_decrease = builder.min_impurity_decrease
    min_impurity_split = builder.min_impurity_split

    # Recursive partition (without actual recursion)
    # splitter.init(X, y, sample_weight_ptr)

    best_splitter_init(splitter, X, y, sample_weight)
    # splitter_init(splitter, X, y, sample_weight)

    # cdef SIZE_t start
    # cdef SIZE_t end
    # cdef SIZE_t depth
    # cdef SIZE_t parent
    # cdef bint is_left
    # cdef SIZE_t n_node_samples = splitter.n_samples
    n_node_samples = splitter.n_samples
    # cdef double weighted_n_samples = splitter.weighted_n_samples
    weighted_n_samples = splitter.weighted_n_samples
    # cdef double weighted_n_node_samples
    # cdef SplitRecord split
    # cdef SIZE_t node_id

    # cdef double impurity = INFINITY
    impurity = INFINITY

    # cdef SIZE_t n_constant_features
    # cdef bint is_leaf
    # cdef bint first = 1
    first = True
    # cdef SIZE_t max_depth_seen = -1
    max_depth_seen = -1
    # cdef int rc = 0
    rc = 0

    stack = Stack(INITIAL_STACK_SIZE)
    # cdef StackRecord stack_record

    # with nogil:
        # push root node onto stack
    # rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)

    rc = stack_push(stack, 0, n_node_samples, 0, TREE_UNDEFINED, 0, INFINITY, 0)

    # if rc == -1:
    #     # got return code -1 - out-of-memory
    #     with gil:
    #         raise MemoryError()

    while not stack_is_empty(stack):
        stack_record = stack_pop(stack)
        # start = stack_record.start
        # end = stack_record.end
        # depth = stack_record.depth
        # parent = stack_record.parent
        # is_left = stack_record.is_left
        # impurity = stack_record.impurity
        # n_constant_features = stack_record.n_constant_features
        start = stack_record["start"]
        end = stack_record["end"]
        depth = stack_record["depth"]
        parent = stack_record["parent"]
        is_left = stack_record["is_left"]
        impurity = stack_record["impurity"]
        n_constant_features = stack_record["n_constant_features"]

        n_node_samples = end - start
        # splitter.node_reset(start, end, &weighted_n_node_samples)
        weighted_n_node_samples = splitter_node_reset(splitter, start, end)

        is_leaf = (depth >= max_depth or
                   n_node_samples < min_samples_split or
                   n_node_samples < 2 * min_samples_leaf or
                   weighted_n_node_samples < 2 * min_weight_leaf)

        if first:

            # TODO: some other way, only for gini here
            # impurity = splitter.node_impurity()
            # dans le code d'origine y'a splitter.node_impurity qui appelle
            # self.criterion
            impurity = gini_node_impurity(splitter.criterion)
            first = False

        # print("impurity: ", impurity)
        # print("min_impurity_split: ", min_impurity_split)
        is_leaf = (is_leaf or (impurity <= min_impurity_split))

        # print("is_leaf: ", is_leaf)
        if not is_leaf:
            # splitter.node_split(impurity, &split, &n_constant_features)

            # TODO: Hmmm y'a un truc qui ne va pas là ! je suis censé renvoyer best,
            #  n_constant_features et les deux derniers arguments ne sont pas utlisés
            # best_splitter_node_split(splitter, impurity, split, n_constant_features)
            split, n_total_constants = best_splitter_node_split(splitter, impurity, n_constant_features)

            # If EPSILON=0 in the below comparison, float precision
            # issues stop splitting, producing trees that are
            # dissimilar to v0.18
            is_leaf = (is_leaf or split.pos >= end or
                       (split.improvement + EPSILON <
                        min_impurity_decrease))

        node_id = tree_add_node(tree, parent, is_left, is_leaf, split.feature,
                                 split.threshold, impurity, n_node_samples,
                                 weighted_n_node_samples)

        # node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
        #                          split.threshold, impurity, n_node_samples,
        #                          weighted_n_node_samples)

        if node_id == SIZE_MAX:
            rc = -1
            break

        # Store value for all nodes, to facilitate tree/model
        # inspection and interpretation
        # TODO oui c'est important ca permet de calculer les predictions...

        splitter_node_value(splitter, tree.values, node_id)
        # splitter_node_value(splitter, tree.value + node_id * tree.value_stride)

        # splitter.node_value(tree.value + node_id * tree.value_stride)

        if not is_leaf:
            # Push right child on stack
            # rc = stack.push(split.pos, end, depth + 1, node_id, 0,
            #                 split.impurity_right, n_constant_features)

            rc = stack_push(stack, split.pos, end, depth + 1, node_id, 0,
                            split.impurity_right, n_constant_features)

            if rc == -1:
                break

            # Push left child on stack
            rc = stack_push(stack, start, split.pos, depth + 1, node_id, 1,
                            split.impurity_left, n_constant_features)
            if rc == -1:
                break

        if depth > max_depth_seen:
            max_depth_seen = depth

    if rc >= 0:
        # print("In depth_first_tree_builder_build calling tree_resize "
        #       "with tree.node_count: ", tree.node_count)
        rc = tree_resize(tree, tree.node_count)
        # rc = tree._resize_c(tree.node_count)

    if rc >= 0:
        tree.max_depth = max_depth_seen

    # TODO: ca ne sert a rien et c'est merdique
    if rc == -1:
        raise MemoryError()


# # Best first builder ----------------------------------------------------------
#
# cdef inline int _add_to_frontier(PriorityHeapRecord* rec,
#                                  PriorityHeap frontier) nogil except -1:
#     """Adds record ``rec`` to the priority queue ``frontier``
#
#     Returns -1 in case of failure to allocate memory (and raise MemoryError)
#     or 0 otherwise.
#     """
#     return frontier.push(rec.node_id, rec.start, rec.end, rec.pos, rec.depth,
#                          rec.is_leaf, rec.improvement, rec.impurity,
#                          rec.impurity_left, rec.impurity_right)
#


# cdef class BestFirstTreeBuilder(TreeBuilder):
#     """Build a decision tree in best-first fashion.
#
#     The best node to expand is given by the node at the frontier that has the
#     highest impurity improvement.
#     """
#     cdef SIZE_t max_leaf_nodes
#
#     def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
#                   SIZE_t min_samples_leaf,  min_weight_leaf,
#                   SIZE_t max_depth, SIZE_t max_leaf_nodes,
#                   double min_impurity_decrease, double min_impurity_split):
#         self.splitter = splitter
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_weight_leaf = min_weight_leaf
#         self.max_depth = max_depth
#         self.max_leaf_nodes = max_leaf_nodes
#         self.min_impurity_decrease = min_impurity_decrease
#         self.min_impurity_split = min_impurity_split


#     cpdef build(self, Tree tree, object X, np.ndarray y,
#                 np.ndarray sample_weight=None):
#         """Build a decision tree from the training set (X, y)."""
#
#         # check input
#         X, y, sample_weight = self._check_input(X, y, sample_weight)
#
#         cdef DOUBLE_t* sample_weight_ptr = NULL
#         if sample_weight is not None:
#             sample_weight_ptr = <DOUBLE_t*> sample_weight.data
#
#         # Parameters
#         cdef Splitter splitter = self.splitter
#         cdef SIZE_t max_leaf_nodes = self.max_leaf_nodes
#         cdef SIZE_t min_samples_leaf = self.min_samples_leaf
#         cdef double min_weight_leaf = self.min_weight_leaf
#         cdef SIZE_t min_samples_split = self.min_samples_split
#
#         # Recursive partition (without actual recursion)
#         splitter.init(X, y, sample_weight_ptr)
#
#         cdef PriorityHeap frontier = PriorityHeap(INITIAL_STACK_SIZE)
#         cdef PriorityHeapRecord record
#         cdef PriorityHeapRecord split_node_left
#         cdef PriorityHeapRecord split_node_right
#
#         cdef SIZE_t n_node_samples = splitter.n_samples
#         cdef SIZE_t max_split_nodes = max_leaf_nodes - 1
#         cdef bint is_leaf
#         cdef SIZE_t max_depth_seen = -1
#         cdef int rc = 0
#         cdef Node* node
#
#         # Initial capacity
#         cdef SIZE_t init_capacity = max_split_nodes + max_leaf_nodes
#         tree._resize(init_capacity)
#
#         with nogil:
#             # add root to frontier
#             rc = self._add_split_node(splitter, tree, 0, n_node_samples,
#                                       INFINITY, IS_FIRST, IS_LEFT, NULL, 0,
#                                       &split_node_left)
#             if rc >= 0:
#                 rc = _add_to_frontier(&split_node_left, frontier)
#
#             if rc == -1:
#                 with gil:
#                     raise MemoryError()
#
#             while not frontier.is_empty():
#                 frontier.pop(&record)
#
#                 node = &tree.nodes[record.node_id]
#                 is_leaf = (record.is_leaf or max_split_nodes <= 0)
#
#                 if is_leaf:
#                     # Node is not expandable; set node as leaf
#                     node.left_child = _TREE_LEAF
#                     node.right_child = _TREE_LEAF
#                     node.feature = _TREE_UNDEFINED
#                     node.threshold = _TREE_UNDEFINED
#
#                 else:
#                     # Node is expandable
#
#                     # Decrement number of split nodes available
#                     max_split_nodes -= 1
#
#                     # Compute left split node
#                     rc = self._add_split_node(splitter, tree,
#                                               record.start, record.pos,
#                                               record.impurity_left,
#                                               IS_NOT_FIRST, IS_LEFT, node,
#                                               record.depth + 1,
#                                               &split_node_left)
#                     if rc == -1:
#                         break
#
#                     # tree.nodes may have changed
#                     node = &tree.nodes[record.node_id]
#
#                     # Compute right split node
#                     rc = self._add_split_node(splitter, tree, record.pos,
#                                               record.end,
#                                               record.impurity_right,
#                                               IS_NOT_FIRST, IS_NOT_LEFT, node,
#                                               record.depth + 1,
#                                               &split_node_right)
#                     if rc == -1:
#                         break
#
#                     # Add nodes to queue
#                     rc = _add_to_frontier(&split_node_left, frontier)
#                     if rc == -1:
#                         break
#
#                     rc = _add_to_frontier(&split_node_right, frontier)
#                     if rc == -1:
#                         break
#
#                 if record.depth > max_depth_seen:
#                     max_depth_seen = record.depth
#
#             if rc >= 0:
#                 rc = tree._resize_c(tree.node_count)
#
#             if rc >= 0:
#                 tree.max_depth = max_depth_seen
#
#         if rc == -1:
#             raise MemoryError()


#     cdef inline int _add_split_node(self, Splitter splitter, Tree tree,
#                                     SIZE_t start, SIZE_t end, double impurity,
#                                     bint is_first, bint is_left, Node* parent,
#                                     SIZE_t depth,
#                                     PriorityHeapRecord* res) nogil except -1:
#         """Adds node w/ partition ``[start, end)`` to the frontier. """
#         cdef SplitRecord split
#         cdef SIZE_t node_id
#         cdef SIZE_t n_node_samples
#         cdef SIZE_t n_constant_features = 0
#         cdef double weighted_n_samples = splitter.weighted_n_samples
#         cdef double min_impurity_decrease = self.min_impurity_decrease
#         cdef double min_impurity_split = self.min_impurity_split
#         cdef double weighted_n_node_samples
#         cdef bint is_leaf
#         cdef SIZE_t n_left, n_right
#         cdef double imp_diff
#
#         splitter.node_reset(start, end, &weighted_n_node_samples)
#
#         if is_first:
#             impurity = splitter.node_impurity()
#
#         n_node_samples = end - start
#         is_leaf = (depth >= self.max_depth or
#                    n_node_samples < self.min_samples_split or
#                    n_node_samples < 2 * self.min_samples_leaf or
#                    weighted_n_node_samples < 2 * self.min_weight_leaf or
#                    impurity <= min_impurity_split)
#
#         if not is_leaf:
#             splitter.node_split(impurity, &split, &n_constant_features)
#             # If EPSILON=0 in the below comparison, float precision issues stop
#             # splitting early, producing trees that are dissimilar to v0.18
#             is_leaf = (is_leaf or split.pos >= end or
#                        split.improvement + EPSILON < min_impurity_decrease)
#
#         node_id = tree._add_node(parent - tree.nodes
#                                  if parent != NULL
#                                  else _TREE_UNDEFINED,
#                                  is_left, is_leaf,
#                                  split.feature, split.threshold, impurity, n_node_samples,
#                                  weighted_n_node_samples)
#         if node_id == SIZE_MAX:
#             return -1
#
#         # compute values also for split nodes (might become leafs later).
#         splitter.node_value(tree.value + node_id * tree.value_stride)
#
#         res.node_id = node_id
#         res.start = start
#         res.end = end
#         res.depth = depth
#         res.impurity = impurity
#
#         if not is_leaf:
#             # is split node
#             res.pos = split.pos
#             res.is_leaf = 0
#             res.improvement = split.improvement
#             res.impurity_left = split.impurity_left
#             res.impurity_right = split.impurity_right
#
#         else:
#             # is leaf => 0 improvement
#             res.pos = end
#             res.is_leaf = 1
#             res.improvement = 0.0
#             res.impurity_left = impurity
#             res.impurity_right = impurity
#
#         return 0


spec_tree = [
    ("n_features", SIZE_t),
    ("n_classes", SIZE_t[::1]),
    ("n_outputs", SIZE_t),
    ("max_n_classes", SIZE_t),
    ("max_depth", SIZE_t),
    ("node_count", SIZE_t),
    ("capacity", SIZE_t),
    # This array contains information about the nodes
    ("nodes", NODE_t[::1]),
    # This array contains values allowing to compute the prediction of each node
    # Its shape is (n_nodes, n_outputs, max_n_classes)
    # TODO: IMPORTANT a priori ca serait mieux ::1 sur le premier axe mais l'init
    #  avec shape (0, ., .) foire dans ce cas avec numba
    ("values", DOUBLE_t[:, :, ::1]),
    # ("values_stride", SIZE_t)
]


@jitclass(spec_tree)
class Tree(object):
    # # =============================================================================
    # # Tree
    # # =============================================================================
    #
    # cdef class Tree:
    #     """Array-based representation of a binary decision tree.
    #
    #     The binary tree is represented as a number of parallel arrays. The i-th
    #     element of each array holds information about the node `i`. Node 0 is the
    #     tree's root. You can find a detailed description of all arrays in
    #     `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    #     nodes, resp. In this case the values of nodes of the other type are
    #     arbitrary!
    #
    #     Attributes
    #     ----------
    #     node_count : int
    #         The number of nodes (internal nodes + leaves) in the tree.
    #
    #     capacity : int
    #         The current capacity (i.e., size) of the arrays, which is at least as
    #         great as `node_count`.
    #
    #     max_depth : int
    #         The depth of the tree, i.e. the maximum depth of its leaves.
    #
    #     children_left : array of int, shape [node_count]
    #         children_left[i] holds the node id of the left child of node i.
    #         For leaves, children_left[i] == TREE_LEAF. Otherwise,
    #         children_left[i] > i. This child handles the case where
    #         X[:, feature[i]] <= threshold[i].
    #
    #     children_right : array of int, shape [node_count]
    #         children_right[i] holds the node id of the right child of node i.
    #         For leaves, children_right[i] == TREE_LEAF. Otherwise,
    #         children_right[i] > i. This child handles the case where
    #         X[:, feature[i]] > threshold[i].
    #
    #     feature : array of int, shape [node_count]
    #         feature[i] holds the feature to split on, for the internal node i.
    #
    #     threshold : array of double, shape [node_count]
    #         threshold[i] holds the threshold for the internal node i.
    #
    #     value : array of double, shape [node_count, n_outputs, max_n_classes]
    #         Contains the constant prediction value of each node.
    #
    #     impurity : array of double, shape [node_count]
    #         impurity[i] holds the impurity (i.e., the value of the splitting
    #         criterion) at node i.
    #
    #     n_node_samples : array of int, shape [node_count]
    #         n_node_samples[i] holds the number of training samples reaching node i.
    #
    #     weighted_n_node_samples : array of int, shape [node_count]
    #         weighted_n_node_samples[i] holds the weighted number of training samples
    #         reaching node i.
    #     """
    #     # Wrap for outside world.
    #     # WARNING: these reference the current `nodes` and `value` buffers, which
    #     # must not be freed by a subsequent memory allocation.
    #     # (i.e. through `_resize` or `__setstate__`)

    def __init__(self, n_features, n_classes, n_outputs):
        #     def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
        #                   int n_outputs):
        #         """Constructor."""
        #         # Input/Output layout
        #         self.n_features = n_features
        #         self.n_outputs = n_outputs
        #         self.n_classes = NULL
        #         safe_realloc(&self.n_classes, n_outputs)
        #
        #         self.max_n_classes = np.max(n_classes)
        #         self.value_stride = n_outputs * self.max_n_classes
        #
        #         cdef SIZE_t k
        #         for k in range(n_outputs):
        #             self.n_classes[k] = n_classes[k]
        #
        #         # Inner structures
        #         self.max_depth = 0
        #         self.node_count = 0
        #         self.capacity = 0
        #         self.value = NULL
        #         self.nodes = NULL
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        # self.n_classes = NULL
        # safe_realloc(&self.n_classes, n_outputs)
        self.n_classes = np.empty(n_outputs, dtype=NP_SIZE_t)
        self.max_n_classes = np.max(n_classes)
        # self.value_stride = n_outputs * self.max_n_classes

        # TODO: no for loop here
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0

        # self.value = NULL
        self.values = np.empty((0, self.n_outputs, self.max_n_classes),
                               dtype=NP_DOUBLE_t)
        self.nodes = np.empty(0, dtype=NP_NODE_t)


def print_tree(tree):
    s = "Tree("
    s += "n_features={n_features}".format(n_features=tree.n_features)
    s += ", n_outputs={n_outputs}".format(n_outputs=tree.n_outputs)
    s += ", n_classes={n_classes}".format(n_classes=tree.n_classes)
    s += ", capacity={capacity}".format(capacity=tree.capacity)
    s += ", node_count={node_count}".format(node_count=tree.node_count)
    s += ")"
    print(s)


def get_nodes(tree):
    import pandas as pd
    nodes = tree.nodes
    columns = [col_name for col_name, _ in node_dtype]
    # columns = ["left_child"]

    return pd.DataFrame.from_records(
        (tuple(node[col] for col in columns) for i, node in enumerate(nodes) if i <
         tree.node_count),
        columns=columns
    )


@njit
def tree_add_node(tree, parent, is_left, is_leaf, feature, threshold, impurity,
                  n_node_samples, weighted_n_node_samples):
    #     cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
    #                           SIZE_t feature, double threshold, double impurity,
    #                           SIZE_t n_node_samples,
    #                           double weighted_n_node_samples) nogil except -1:
    #         """Add a node to the tree.
    #
    #         The new node registers itself as the child of its parent.
    #
    #         Returns (size_t)(-1) on error.
    #         """
    #         cdef SIZE_t node_id = self.node_count
    #
    #         if node_id >= self.capacity:
    #             if self._resize_c() != 0:
    #                 return SIZE_MAX
    #
    #         cdef Node* node = &self.nodes[node_id]
    #         node.impurity = impurity
    #         node.n_node_samples = n_node_samples
    #         node.weighted_n_node_samples = weighted_n_node_samples
    #
    #         if parent != _TREE_UNDEFINED:
    #             if is_left:
    #                 self.nodes[parent].left_child = node_id
    #             else:
    #                 self.nodes[parent].right_child = node_id
    #
    #         if is_leaf:
    #             node.left_child = _TREE_LEAF
    #             node.right_child = _TREE_LEAF
    #             node.feature = _TREE_UNDEFINED
    #             node.threshold = _TREE_UNDEFINED
    #
    #         else:
    #             # left_child and right_child will be set later
    #             node.feature = feature
    #             node.threshold = threshold
    #
    #         self.node_count += 1
    #
    #         return node_id

    node_id = tree.node_count

    if node_id >= tree.capacity:
        # print("In tree_add_node calling tree_resize with no capacity")
        # tree_add_node
        tree_resize(tree)
        # TODO: qu'est ce qui se passe ici ?
        # if tree._resize_c() != 0:
        #     return SIZE_MAX

    nodes = tree.nodes
    node = nodes[node_id]
    # cdef Node* node = &tree.nodes[node_id]
    node["impurity"] = impurity
    node["n_node_samples"] = n_node_samples
    node["weighted_n_node_samples"] = weighted_n_node_samples

    if parent != TREE_UNDEFINED:
        if is_left:
            nodes[parent]["left_child"] = node_id
        else:
            nodes[parent]["right_child"] = node_id

    if is_leaf:
        node["left_child"] = TREE_LEAF
        node["right_child"] = TREE_LEAF
        node["feature"] = TREE_UNDEFINED
        node["threshold"] = TREE_UNDEFINED
    else:
        # left_child and right_child will be set later
        node["feature"] = feature
        node["threshold"] = threshold

    tree.node_count += 1

    return node_id


@njit
def tree_resize(tree, capacity=SIZE_MAX):
    #     cdef int _resize(self, SIZE_t capacity) nogil except -1:
    #         """Resize all inner arrays to `capacity`, if `capacity` == -1, then
    #            double the size of the inner arrays.
    #
    #         Returns -1 in case of failure to allocate memory (and raise MemoryError)
    #         or 0 otherwise.
    #         """
    #         if self._resize_c(capacity) != 0:
    #             # Acquire gil only if we need to raise
    #             with gil:
    #                 raise MemoryError()
    #
    #     cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
    #         """Guts of _resize
    #
    #         Returns -1 in case of failure to allocate memory (and raise MemoryError)
    #         or 0 otherwise.
    #         """
    #         if capacity == self.capacity and self.nodes != NULL:
    #             return 0
    #
    #         if capacity == SIZE_MAX:
    #             if self.capacity == 0:
    #                 capacity = 3  # default initial value
    #             else:
    #                 capacity = 2 * self.capacity
    #
    #         safe_realloc(&self.nodes, capacity)
    #         safe_realloc(&self.value, capacity * self.value_stride)
    #
    #         # value memory is initialised to 0 to enable classifier argmax
    #         if capacity > self.capacity:
    #             memset(<void*>(self.value + self.capacity * self.value_stride), 0,
    #                    (capacity - self.capacity) * self.value_stride *
    #                    sizeof(double))
    #
    #         # if capacity smaller than node_count, adjust the counter
    #         if capacity < self.node_count:
    #             self.node_count = capacity
    #
    #         self.capacity = capacity
    #         return 0

    # TODO: When does this happen ?
    # if capacity == tree.capacity and tree.nodes != NULL:

    # print("----------------")
    # print("In tree.resize with")
    # print("capacity: ", capacity)
    # print("tree.capacity: ", tree.capacity)
    # print("tree.nodes.size: ", tree.nodes.size)

    # TODO: attention grosse difference ici
    # if capacity == tree.capacity and tree.nodes.size > 0:
    if capacity <= tree.capacity and tree.nodes.size > 0:
        return 0

    if capacity == SIZE_MAX:
        if tree.capacity == 0:
            capacity = 3  # default initial value
        else:
            capacity = 2 * tree.capacity

    # print("new capacity: ", capacity)

    # safe_realloc( & tree.nodes, capacity)
    tree.nodes = resize(tree.nodes, capacity)
    # safe_realloc( & tree.value, capacity * tree.value_stride)

    # TODO: je ne comprends toujours pas tres bien a quoi sert ce value mais bon

    # tree.value = resize3d(tree.values, capacity * tree.value_stride, zeros=True)
    tree.values = resize3d(tree.values, capacity, zeros=True)

    # value memory is initialised to 0 to enable classifier argmax
    # if capacity > tree.capacity:
    #     memset( < void * > (tree.value + tree.capacity * tree.value_stride), 0,
    #     (capacity - tree.capacity) * tree.value_stride *
    #     sizeof(double))

    # if capacity smaller than node_count, adjust the counter
    if capacity < tree.node_count:
        tree.node_count = capacity

    tree.capacity = capacity
    return 0


@njit
def tree_get_value_ndarray(tree):
    #     cdef np.ndarray _get_value_ndarray(self):
    #         """Wraps value as a 3-d NumPy array.
    #
    #         The array keeps a reference to this Tree, which manages the underlying
    #         memory.
    #         """
    #         cdef np.npy_intp shape[3]
    #         shape[0] = <np.npy_intp> self.node_count
    #         shape[1] = <np.npy_intp> self.n_outputs
    #         shape[2] = <np.npy_intp> self.max_n_classes
    #         cdef np.ndarray arr
    #         arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
    #         Py_INCREF(self)
    #         if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
    #             raise ValueError("Can't initialize array.")
    #         return arr
    pass

@njit
def tree_get_node_ndarray(tree):
    #     cdef np.ndarray _get_node_ndarray(self):
    #         """Wraps nodes as a NumPy struct array.
    #
    #         The array keeps a reference to this Tree, which manages the underlying
    #         memory. Individual fields are publicly accessible as properties of the
    #         Tree.
    #         """
    #         cdef np.npy_intp shape[1]
    #         shape[0] = <np.npy_intp> self.node_count
    #         cdef np.npy_intp strides[1]
    #         strides[0] = sizeof(Node)
    #         cdef np.ndarray arr
    #         Py_INCREF(NODE_DTYPE)
    #         arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
    #                                    <np.dtype> NODE_DTYPE, 1, shape,
    #                                    strides, <void*> self.nodes,
    #                                    np.NPY_DEFAULT, None)
    #         Py_INCREF(self)
    #         if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
    #             raise ValueError("Can't initialize array.")
    #         return arr
    pass

# @njit
def tree_predict(tree, X):
    #     cpdef np.ndarray predict(self, object X):
    #         """Predict target for X."""
    #         out = self._get_value_ndarray().take(self.apply(X), axis=0,
    #                                              mode='clip')
    #         if self.n_outputs == 1:
    #             out = out.reshape(X.shape[0], self.max_n_classes)
    #         return out

    # out = tree._get_value_ndarray().take(tree_apply(tree, X), axis=0,
    #                                      mode='clip')

    # TODO: numba n'accepte pas l'option axis
    idx_leaves = tree_apply(tree, X)
    out = tree.values.take(idx_leaves, axis=0)

    if tree.n_outputs == 1:
        out = out.reshape(X.shape[0], tree.max_n_classes)
    return out


@njit
def tree_apply(tree, X):
    #     cpdef np.ndarray apply(self, object X):
    #         """Finds the terminal region (=leaf node) for each sample in X."""
    #         if issparse(X):
    #             return self._apply_sparse_csr(X)
    #         else:
    #             return self._apply_dense(X)
    return tree_apply_dense(tree, X)

@njit
def tree_apply_dense(tree, X):
    #     cdef inline np.ndarray _apply_dense(self, object X):
    #         """Finds the terminal region (=leaf node) for each sample in X."""
    #
    #         # Check input
    #         if not isinstance(X, np.ndarray):
    #             raise ValueError("X should be in np.ndarray format, got %s"
    #                              % type(X))
    #
    #         if X.dtype != DTYPE:
    #             raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
    #
    #         # Extract input
    #         cdef const DTYPE_t[:, :] X_ndarray = X
    #         cdef SIZE_t n_samples = X.shape[0]
    #
    #         # Initialize output
    #         cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
    #         cdef SIZE_t* out_ptr = <SIZE_t*> out.data
    #
    #         # Initialize auxiliary data-structure
    #         cdef Node* node = NULL
    #         cdef SIZE_t i = 0
    #
    #         with nogil:
    #             for i in range(n_samples):
    #                 node = self.nodes
    #                 # While node not a leaf
    #                 while node.left_child != _TREE_LEAF:
    #                     # ... and node.right_child != _TREE_LEAF:
    #                     if X_ndarray[i, node.feature] <= node.threshold:
    #                         node = &self.nodes[node.left_child]
    #                     else:
    #                         node = &self.nodes[node.right_child]
    #
    #                 out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset
    #
    #         return out

    # Check input
    # TODO: ces checks sont inutiles non ? On en fait deja avant dans la classe python
    # if not isinstance(X, np.ndarray):
    #     raise ValueError("X should be in np.ndarray format, got %s"
    #                      % type(X))
    #
    # if X.dtype != DTYPE_t:
    #     raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    # Extract input
    # cdef const DTYPE_t[:, :] X_ndarray = X
    X_ndarray = X

    # cdef SIZE_t n_samples = X.shape[0]
    n_samples = X.shape[0]

    # Initialize output
    # cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
    out = np.zeros((n_samples,), dtype=NP_SIZE_t)
    # cdef SIZE_t* out_ptr = <SIZE_t*> out.data

    # Initialize auxiliary data-structure
    # cdef Node* node = NULL
    # cdef SIZE_t i = 0

    # with nogil:

    nodes = tree.nodes

    for i in range(n_samples):
        # Index of the leaf containing the sample
        idx_leaf = 0
        node = nodes[idx_leaf]
        # While node not a leaf
        while node["left_child"] != TREE_LEAF:
            # ... and node.right_child != TREE_LEAF:
            if X_ndarray[i, node["feature"]] <= node["threshold"]:
                idx_leaf = node["left_child"]
            else:
                idx_leaf = node["right_child"]
            node = nodes[idx_leaf]

        # out_ptr[i] = <SIZE_t>(node - tree.nodes)  # node offset
        out[i] = SIZE_t(idx_leaf)

    return out


@njit
def tree_apply_sparse_csr(tree, X):
    #
    #     cdef inline np.ndarray _apply_sparse_csr(self, object X):
    #         """Finds the terminal region (=leaf node) for each sample in sparse X.
    #         """
    #         # Check input
    #         if not isinstance(X, csr_matrix):
    #             raise ValueError("X should be in csr_matrix format, got %s"
    #                              % type(X))
    #
    #         if X.dtype != DTYPE:
    #             raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
    #
    #         # Extract input
    #         cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
    #         cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
    #         cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr
    #
    #         cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
    #         cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
    #         cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data
    #
    #         cdef SIZE_t n_samples = X.shape[0]
    #         cdef SIZE_t n_features = X.shape[1]
    #
    #         # Initialize output
    #         cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
    #                                                        dtype=np.intp)
    #         cdef SIZE_t* out_ptr = <SIZE_t*> out.data
    #
    #         # Initialize auxiliary data-structure
    #         cdef DTYPE_t feature_value = 0.
    #         cdef Node* node = NULL
    #         cdef DTYPE_t* X_sample = NULL
    #         cdef SIZE_t i = 0
    #         cdef INT32_t k = 0
    #
    #         # feature_to_sample as a data structure records the last seen sample
    #         # for each feature; functionally, it is an efficient way to identify
    #         # which features are nonzero in the present sample.
    #         cdef SIZE_t* feature_to_sample = NULL
    #
    #         safe_realloc(&X_sample, n_features)
    #         safe_realloc(&feature_to_sample, n_features)
    #
    #         with nogil:
    #             memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))
    #
    #             for i in range(n_samples):
    #                 node = self.nodes
    #
    #                 for k in range(X_indptr[i], X_indptr[i + 1]):
    #                     feature_to_sample[X_indices[k]] = i
    #                     X_sample[X_indices[k]] = X_data[k]
    #
    #                 # While node not a leaf
    #                 while node.left_child != _TREE_LEAF:
    #                     # ... and node.right_child != _TREE_LEAF:
    #                     if feature_to_sample[node.feature] == i:
    #                         feature_value = X_sample[node.feature]
    #
    #                     else:
    #                         feature_value = 0.
    #
    #                     if feature_value <= node.threshold:
    #                         node = &self.nodes[node.left_child]
    #                     else:
    #                         node = &self.nodes[node.right_child]
    #
    #                 out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset
    #
    #             # Free auxiliary arrays
    #             free(X_sample)
    #             free(feature_to_sample)
    #
    #         return out
    pass

@njit
def tree_decision_path(tree, X):
    #     cpdef object decision_path(self, object X):
    #         """Finds the decision path (=node) for each sample in X."""
    #         if issparse(X):
    #             return self._decision_path_sparse_csr(X)
    #         else:
    #             return self._decision_path_dense(X)
    pass

@njit
def tree_decision_path_dense(tree, X):
    #     cdef inline object _decision_path_dense(self, object X):
    #         """Finds the decision path (=node) for each sample in X."""
    #
    #         # Check input
    #         if not isinstance(X, np.ndarray):
    #             raise ValueError("X should be in np.ndarray format, got %s"
    #                              % type(X))
    #
    #         if X.dtype != DTYPE:
    #             raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
    #
    #         # Extract input
    #         cdef const DTYPE_t[:, :] X_ndarray = X
    #         cdef SIZE_t n_samples = X.shape[0]
    #
    #         # Initialize output
    #         cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
    #         cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data
    #
    #         cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
    #                                                    (1 + self.max_depth),
    #                                                    dtype=np.intp)
    #         cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data
    #
    #         # Initialize auxiliary data-structure
    #         cdef Node* node = NULL
    #         cdef SIZE_t i = 0
    #
    #         with nogil:
    #             for i in range(n_samples):
    #                 node = self.nodes
    #                 indptr_ptr[i + 1] = indptr_ptr[i]
    #
    #                 # Add all external nodes
    #                 while node.left_child != _TREE_LEAF:
    #                     # ... and node.right_child != _TREE_LEAF:
    #                     indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
    #                     indptr_ptr[i + 1] += 1
    #
    #                     if X_ndarray[i, node.feature] <= node.threshold:
    #                         node = &self.nodes[node.left_child]
    #                     else:
    #                         node = &self.nodes[node.right_child]
    #
    #                 # Add the leave node
    #                 indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
    #                 indptr_ptr[i + 1] += 1
    #
    #         indices = indices[:indptr[n_samples]]
    #         cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
    #                                                dtype=np.intp)
    #         out = csr_matrix((data, indices, indptr),
    #                          shape=(n_samples, self.node_count))
    #
    #         return out
    pass

@njit
def tree_decision_path_sparse_csr(tree, X):
    #     cdef inline object _decision_path_sparse_csr(self, object X):
    #         """Finds the decision path (=node) for each sample in X."""
    #
    #         # Check input
    #         if not isinstance(X, csr_matrix):
    #             raise ValueError("X should be in csr_matrix format, got %s"
    #                              % type(X))
    #
    #         if X.dtype != DTYPE:
    #             raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
    #
    #         # Extract input
    #         cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
    #         cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
    #         cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr
    #
    #         cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
    #         cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
    #         cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data
    #
    #         cdef SIZE_t n_samples = X.shape[0]
    #         cdef SIZE_t n_features = X.shape[1]
    #
    #         # Initialize output
    #         cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
    #         cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data
    #
    #         cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
    #                                                    (1 + self.max_depth),
    #                                                    dtype=np.intp)
    #         cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data
    #
    #         # Initialize auxiliary data-structure
    #         cdef DTYPE_t feature_value = 0.
    #         cdef Node* node = NULL
    #         cdef DTYPE_t* X_sample = NULL
    #         cdef SIZE_t i = 0
    #         cdef INT32_t k = 0
    #
    #         # feature_to_sample as a data structure records the last seen sample
    #         # for each feature; functionally, it is an efficient way to identify
    #         # which features are nonzero in the present sample.
    #         cdef SIZE_t* feature_to_sample = NULL
    #
    #         safe_realloc(&X_sample, n_features)
    #         safe_realloc(&feature_to_sample, n_features)
    #
    #         with nogil:
    #             memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))
    #
    #             for i in range(n_samples):
    #                 node = self.nodes
    #                 indptr_ptr[i + 1] = indptr_ptr[i]
    #
    #                 for k in range(X_indptr[i], X_indptr[i + 1]):
    #                     feature_to_sample[X_indices[k]] = i
    #                     X_sample[X_indices[k]] = X_data[k]
    #
    #                 # While node not a leaf
    #                 while node.left_child != _TREE_LEAF:
    #                     # ... and node.right_child != _TREE_LEAF:
    #
    #                     indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
    #                     indptr_ptr[i + 1] += 1
    #
    #                     if feature_to_sample[node.feature] == i:
    #                         feature_value = X_sample[node.feature]
    #
    #                     else:
    #                         feature_value = 0.
    #
    #                     if feature_value <= node.threshold:
    #                         node = &self.nodes[node.left_child]
    #                     else:
    #                         node = &self.nodes[node.right_child]
    #
    #                 # Add the leave node
    #                 indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
    #                 indptr_ptr[i + 1] += 1
    #
    #             # Free auxiliary arrays
    #             free(X_sample)
    #             free(feature_to_sample)
    #
    #         indices = indices[:indptr[n_samples]]
    #         cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
    #                                                dtype=np.intp)
    #         out = csr_matrix((data, indices, indptr),
    #                          shape=(n_samples, self.node_count))
    #
    #         return out
    pass

@njit
def tree_compute_feature_importances(tree, normalize):
    #     cpdef compute_feature_importances(self, normalize=True):
    #         """Computes the importance of each feature (aka variable)."""
    #         cdef Node* left
    #         cdef Node* right
    #         cdef Node* nodes = self.nodes
    #         cdef Node* node = nodes
    #         cdef Node* end_node = node + self.node_count
    #
    #         cdef double normalizer = 0.
    #
    #         cdef np.ndarray[np.float64_t, ndim=1] importances
    #         importances = np.zeros((self.n_features,))
    #         cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data
    #
    #         with nogil:
    #             while node != end_node:
    #                 if node.left_child != _TREE_LEAF:
    #                     # ... and node.right_child != _TREE_LEAF:
    #                     left = &nodes[node.left_child]
    #                     right = &nodes[node.right_child]
    #
    #                     importance_data[node.feature] += (
    #                         node.weighted_n_node_samples * node.impurity -
    #                         left.weighted_n_node_samples * left.impurity -
    #                         right.weighted_n_node_samples * right.impurity)
    #                 node += 1
    #
    #         importances /= nodes[0].weighted_n_node_samples
    #
    #         if normalize:
    #             normalizer = np.sum(importances)
    #
    #             if normalizer > 0.0:
    #                 # Avoid dividing by zero (e.g., when root is pure)
    #                 importances /= normalizer
    #
    #         return importances
    pass



#     property n_classes:
#         def __get__(self):
#             return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)
#     property children_left:
#         def __get__(self):
#             return self._get_node_ndarray()['left_child'][:self.node_count]
#
#     property children_right:
#         def __get__(self):
#             return self._get_node_ndarray()['right_child'][:self.node_count]
#
#     property n_leaves:
#         def __get__(self):
#             return np.sum(np.logical_and(
#                 self.children_left == -1,
#                 self.children_right == -1))
#
#     property feature:
#         def __get__(self):
#             return self._get_node_ndarray()['feature'][:self.node_count]
#
#     property threshold:
#         def __get__(self):
#             return self._get_node_ndarray()['threshold'][:self.node_count]
#
#     property impurity:
#         def __get__(self):
#             return self._get_node_ndarray()['impurity'][:self.node_count]
#
#     property n_node_samples:
#         def __get__(self):
#             return self._get_node_ndarray()['n_node_samples'][:self.node_count]
#
#     property weighted_n_node_samples:
#         def __get__(self):
#             return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]
#
#     property value:
#         def __get__(self):
#             return self._get_value_ndarray()[:self.node_count]

@njit
def tree_compute_partial_dependence():
#     def compute_partial_dependence(self, DTYPE_t[:, ::1] X,
#                                    int[::1] target_features,
#                                    double[::1] out):
#         """Partial dependence of the response on the ``target_feature`` set.
#
#         For each sample in ``X`` a tree traversal is performed.
#         Each traversal starts from the root with weight 1.0.
#
#         At each non-leaf node that splits on a target feature, either
#         the left child or the right child is visited based on the feature
#         value of the current sample, and the weight is not modified.
#         At each non-leaf node that splits on a complementary feature,
#         both children are visited and the weight is multiplied by the fraction
#         of training samples which went to each child.
#
#         At each leaf, the value of the node is multiplied by the current
#         weight (weights sum to 1 for all visited terminal nodes).
#
#         Parameters
#         ----------
#         X : view on 2d ndarray, shape (n_samples, n_target_features)
#             The grid points on which the partial dependence should be
#             evaluated.
#         target_features : view on 1d ndarray, shape (n_target_features)
#             The set of target features for which the partial dependence
#             should be evaluated.
#         out : view on 1d ndarray, shape (n_samples)
#             The value of the partial dependence function on each grid
#             point.
#         """
#         cdef:
#             double[::1] weight_stack = np.zeros(self.node_count,
#                                                 dtype=np.float64)
#             SIZE_t[::1] node_idx_stack = np.zeros(self.node_count,
#                                                   dtype=np.intp)
#             SIZE_t sample_idx
#             SIZE_t feature_idx
#             int stack_size
#             double left_sample_frac
#             double current_weight
#             double total_weight  # used for sanity check only
#             Node *current_node  # use a pointer to avoid copying attributes
#             SIZE_t current_node_idx
#             bint is_target_feature
#             SIZE_t _TREE_LEAF = TREE_LEAF  # to avoid python interactions
#
#         for sample_idx in range(X.shape[0]):
#             # init stacks for current sample
#             stack_size = 1
#             node_idx_stack[0] = 0  # root node
#             weight_stack[0] = 1  # all the samples are in the root node
#             total_weight = 0
#
#             while stack_size > 0:
#                 # pop the stack
#                 stack_size -= 1
#                 current_node_idx = node_idx_stack[stack_size]
#                 current_node = &self.nodes[current_node_idx]
#
#                 if current_node.left_child == _TREE_LEAF:
#                     # leaf node
#                     out[sample_idx] += (weight_stack[stack_size] *
#                                         self.value[current_node_idx])
#                     total_weight += weight_stack[stack_size]
#                 else:
#                     # non-leaf node
#
#                     # determine if the split feature is a target feature
#                     is_target_feature = False
#                     for feature_idx in range(target_features.shape[0]):
#                         if target_features[feature_idx] == current_node.feature:
#                             is_target_feature = True
#                             break
#
#                     if is_target_feature:
#                         # In this case, we push left or right child on stack
#                         if X[sample_idx, feature_idx] <= current_node.threshold:
#                             node_idx_stack[stack_size] = current_node.left_child
#                         else:
#                             node_idx_stack[stack_size] = current_node.right_child
#                         stack_size += 1
#                     else:
#                         # In this case, we push both children onto the stack,
#                         # and give a weight proportional to the number of
#                         # samples going through each branch.
#
#                         # push left child
#                         node_idx_stack[stack_size] = current_node.left_child
#                         left_sample_frac = (
#                             self.nodes[current_node.left_child].weighted_n_node_samples /
#                             current_node.weighted_n_node_samples)
#                         current_weight = weight_stack[stack_size]
#                         weight_stack[stack_size] = current_weight * left_sample_frac
#                         stack_size += 1
#
#                         # push right child
#                         node_idx_stack[stack_size] = current_node.right_child
#                         weight_stack[stack_size] = (
#                             current_weight * (1 - left_sample_frac))
#                         stack_size += 1
#
#             # Sanity check. Should never happen.
#             if not (0.999 < total_weight < 1.001):
#                 raise ValueError("Total weight should be 1.0 but was %.9f" %
#                                  total_weight)
    pass


# cdef class _PathFinder(_CCPPruneController):
#     """Record metrics used to return the cost complexity path."""
#     cdef DOUBLE_t[:] ccp_alphas
#     cdef DOUBLE_t[:] impurities
#     cdef UINT32_t count
#
#     def __cinit__(self,  int node_count):
#         self.ccp_alphas = np.zeros(shape=(node_count), dtype=np.float64)
#         self.impurities = np.zeros(shape=(node_count), dtype=np.float64)
#         self.count = 0
#
#     cdef void save_metrics(self,
#                            DOUBLE_t effective_alpha,
#                            DOUBLE_t subtree_impurities) nogil:
#         self.ccp_alphas[self.count] = effective_alpha
#         self.impurities[self.count] = subtree_impurities
#         self.count += 1
#
#
