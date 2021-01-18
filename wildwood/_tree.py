"""
This contains all the data structures for holding tree data
"""

import numpy as np
from ._utils import (
    np_uint8,
    np_size_t,
    nb_size_t,
    np_ssize_t,
    nb_ssize_t,
    nb_uint8,
    np_bool,
    np_float32,
    nb_float32,
    max_size_t,
    from_dtype,
    njit,
    jitclass,
    resize,
    resize2d,
)


# from ._splitter import (
#     splitter_init,
#     splitter_node_reset,
#     spec_split_record,
#     # splitter_node_value,
#     BestSplitter,
#     best_splitter_node_split,
#     SplitRecord,
#     best_splitter_init,
#     gini_node_impurity,
#     gini_children_impurity,
# )

import numba


# TODO: on a vraiment besoin de tout ca dans un stack_record ?


spec_node_record = [
    ("start_train", np_size_t),
    ("end_train", np_size_t),
    ("start_valid", np_size_t),
    ("end_valid", np_size_t),
    ("depth", np_size_t),
    ("parent", np_ssize_t),
    ("is_left", np_bool),
    ("impurity", np_float32),
    ("n_constant_features", np_size_t),
]

np_node_record = np.dtype(spec_node_record)
nb_node_record = from_dtype(np_node_record)


spec_records = [
    ("capacity", nb_size_t),
    ("top", nb_size_t),
    ("stack_", nb_node_record[::1]),
]


@jitclass(spec_records)
class Records(object):
    """
    A simple LIFO (last in, first out) data structure to stack the nodes to split
    during tree growing

    Attributes
    ----------
    capacity : intp
        The number of elements the stack can hold. If more is necessary then
        self.stack_ is resized

    top : intp
        The number of elements currently on the stack.

    stack_ : array of stack_record data types
        The internal stack of records
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.top = nb_size_t(0)
        self.stack_ = np.empty(capacity, dtype=np_node_record)


@njit
def push_node_record(
    records,
    start_train,
    end_train,
    start_valid,
    end_valid,
    depth,
    parent,
    is_left,
    impurity,
    n_constant_features,
):
    top = records.top
    stack_ = records.stack_
    # Resize the stack if capacity is not enough
    if top >= records.capacity:
        records.capacity = nb_size_t(2 * records.capacity)
        records.stack_ = resize(stack_, records.capacity)

    stack_top = records.stack_[top]
    stack_top["start_train"] = start_train
    stack_top["end_train"] = end_train
    stack_top["start_valid"] = start_valid
    stack_top["end_valid"] = end_valid
    stack_top["depth"] = depth
    stack_top["parent"] = parent
    stack_top["is_left"] = is_left
    stack_top["impurity"] = impurity
    stack_top["n_constant_features"] = n_constant_features

    # We have one more record in the stack
    records.top = top + nb_size_t(1)


@njit
def has_records(records):
    # print("records.top: ", records.top)
    return records.top <= nb_size_t(0)


@njit
def pop_node_record(stack):
    top = stack.top
    stack_ = stack.stack_
    # print("top: ", top)
    # print("stack_: ", stack_)
    # print("top - 1", top-1)
    stack_record = stack_[np_size_t(top - 1)]
    stack.top = nb_size_t(top - 1)
    # print("stack.top: ", stack.top)
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
    columns = [col_name for col_name, _ in spec_node_record]
    # columns = ["left_child"]
    return pd.DataFrame.from_records(
        (
            tuple(node[col] for col in columns)
            for i, node in enumerate(nodes)
            if i < stack.node_count
        ),
        columns=columns,
    )


# A numpy dtype containing the node information saved in the tree
spec_node_tree = [
    ("left_child", nb_ssize_t),
    ("right_child", nb_ssize_t),
    ("feature", nb_ssize_t),
    ("threshold", nb_float32),
    ("bin_threshold", nb_uint8),
    ("impurity", nb_float32),
    ("n_samples_train", nb_size_t),
    ("n_samples_valid", nb_size_t),
    ("weighted_n_samples_train", nb_float32),
    ("weighted_n_samples_valid", nb_float32),
]


np_node_tree = np.dtype(
    [
        ("left_child", np_ssize_t),
        ("right_child", np_ssize_t),
        ("feature", np_ssize_t),
        ("threshold", np_float32),
        ("bin_threshold", np_uint8),
        ("impurity", np_float32),
        ("n_samples_train", np_size_t),
        ("n_samples_valid", np_size_t),
        ("weighted_n_samples_train", np_float32),
        ("weighted_n_samples_valid", np_float32),
    ]
)

nb_node_tree = numba.from_dtype(np_node_tree)


@njit
def set_node_tree(nodes, idx, node):
    """
    Set a node in an array of nodes at index idx

    Parameters
    ----------
    nodes : array of np_node
        Array containing the nodes in the tree.

    idx : intp
        Destination index of the node.

    node : NodeTree
        The node to be inserted.
    """
    node_dtype = nodes[idx]
    node_dtype["left_child"] = node.left_child
    node_dtype["right_child"] = node.right_child
    node_dtype["feature"] = node.feature
    node_dtype["threshold"] = node.threshold
    node_dtype["bin_threshold"] = node.bin_threshold
    node_dtype["impurity"] = node.impurity
    node_dtype["n_samples_train"] = node.n_samples_train
    node_dtype["n_samples_valid"] = node.n_samples_valid
    node_dtype["weighted_n_samples_train"] = node.weighted_n_samples_train
    node_dtype["weighted_n_samples_valid"] = node.weighted_n_samples_valid


@njit
def get_node_tree(nodes, idx):
    """
    Get node at index idx

    Parameters
    ----------
    nodes : array of np_node
        Array containing the nodes in the tree.

    idx : intp
        Index of the node to retrieve.

    Returns
    -------
    output : NodeTree
        Retrieved node
    """
    # It's a jitclass object
    node = nodes[idx]
    return NodeTree(
        node["left_child"],
        node["right_child"],
        node["feature"],
        node["threshold"],
        node["bin_threshold"],
        node["impurity"],
        node["n_samples"],
        node["weighted_n_samples"],
    )


# TODO: do we really need a dataclass for this ? a namedtuple instead ? or juste a
#  dtype ?


@jitclass(spec_node_tree)
class NodeTree(object):
    def __init__(
        self,
        left_child,
        right_child,
        feature,
        threshold,
        bin_threshold,
        impurity,
        n_samples_train,
        n_samples_valid,
        weighted_n_samples_train,
        weighted_n_samples_valid,
    ):
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.bin_threshold = bin_threshold
        self.impurity = impurity
        self.n_samples_train = n_samples_train
        self.n_samples_valid = n_samples_valid
        self.weighted_n_samples_train = weighted_n_samples_train
        self.weighted_n_samples_valid = weighted_n_samples_valid


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

# spec_tree_builder = [
#     ("splitter", get_numba_type(BestSplitter)),
#     ("min_samples_split", SIZE_t),
#     ("min_samples_leaf", SIZE_t),
#     ("min_weight_leaf", double_t),
#     ("max_depth", SIZE_t),
#     ("min_impurity_split", double_t),
#     ("min_impurity_decrease", double_t),
# ]
#
#
# @jitclass
# class TreeBuilder(object):
#     # cdef class TreeBuilder:
#     #     # The TreeBuilder recursively builds a Tree object from training samples,
#     #     # using a Splitter object for splitting internal nodes and assigning
#     #     # values to leaves.
#     #     #
#     #     # This class controls the various stopping criteria and the node splitting
#     #     # evaluation order, e.g. depth-first or best-first.
#     #
#     #     cdef Splitter splitter              # Splitting algorithm
#     #
#     #     cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
#     #     cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
#     #     cdef double min_weight_leaf         # Minimum weight in a leaf
#     #     cdef SIZE_t max_depth               # Maximal tree depth
#     #     cdef double min_impurity_split
#     #     cdef double min_impurity_decrease   # Impurity threshold for early stopping
#
#     def __init__(self):
#         pass


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


@njit
def print_node_tree(node):
    left_child = node["left_child"]
    right_child = node["right_child"]
    feature = node["feature"]
    threshold = node["threshold"]
    bin_threshold = node["bin_threshold"]
    impurity = node["impurity"]

    n_samples_train = node["n_samples_train"]
    n_samples_valid = node["n_samples_valid"]
    weighted_n_samples_train = node["weighted_n_samples_train"]
    weighted_n_samples_valid = node["weighted_n_samples_valid"]

    s = "Node(left_child: {left_child}".format(left_child=left_child)
    s += ", right_child: {right_child}".format(right_child=right_child)
    s += ", feature: {feature}:".format(feature=feature)
    s += ", bin_threshold: {bin_threshold}".format(bin_threshold=bin_threshold)
    s += ", n_samples_train: {n_samples_train}".format(n_samples_train=n_samples_train)
    s += ", n_samples_valid: {n_samples_valid}".format(n_samples_valid=n_samples_valid)
    s += ", weighted_n_samples_train: {weighted_n_samples_train}".format(
        weighted_n_samples_train=weighted_n_samples_train
    )
    # s += ", weighted_n_samples_valid: {weighted_n_samples_valid}:".format(
    #     weighted_n_samples_valid=weighted_n_samples_valid
    # )
    print(s)


IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0

TREE_LEAF = nb_ssize_t(-1)
TREE_UNDEFINED = nb_ssize_t(-2)

# TODO: replace n_classes by pred_size ?

spec_tree = [
    ("n_features", nb_size_t),
    ("n_classes", nb_size_t),
    ("max_depth", nb_size_t),
    ("node_count", nb_size_t),
    ("capacity", nb_size_t),
    # This array contains information about the nodes
    ("nodes", nb_node_tree[::1]),
    # This array contains values allowing to compute the prediction of each node
    # Its shape is (n_nodes, n_outputs, max_n_classes)
    # TODO: IMPORTANT a priori ca serait mieux ::1 sur le premier axe mais l'init
    #  avec shape (0, ., .) foire dans ce cas avec numba
    ("y_pred", nb_float32[:, ::1]),
    # TODO: renommer en ("y_sum_bins", nb_float32[:, ::1])) ?
]


# TODO: pas sur que ca soit utile en fait values avec cette strategie histogramme ?


@jitclass(spec_tree)
class Tree(object):
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        # Both values and nodes arrays have zero on the first axis and are resized
        # later when we know the capacity of the tree
        # The array of nodes contained in the tree
        self.nodes = np.empty(0, dtype=np_node_tree)
        # The array of y sums or counts for each node
        self.y_pred = np.empty((0, self.n_classes), dtype=np_float32)


@njit
def print_nodes(tree):
    for node in tree.nodes:
        print_node_tree(node)


@njit
def print_tree(tree):
    s = "-" * 64 + "\n"
    s += "Tree("
    s += "n_features={n_features}".format(n_features=tree.n_features)
    s += ", n_classes={n_classes}".format(n_classes=tree.n_classes)
    s += ", capacity={capacity}".format(capacity=tree.capacity)
    s += ", node_count={node_count}".format(node_count=tree.node_count)
    s += ")"
    print(s)
    if tree.node_count > 0:
        print_nodes(tree)


def get_nodes(tree):
    import pandas as pd

    nodes = tree.nodes
    columns = [col_name for col_name, _ in np_node_tree]
    # columns = ["left_child"]

    return pd.DataFrame.from_records(
        (
            tuple(node[col] for col in columns)
            for i, node in enumerate(nodes)
            if i < tree.node_count
        ),
        columns=columns,
    )


@njit
def add_node_tree(
    tree,
    parent,
    is_left,
    is_leaf,
    feature,
    threshold,
    bin_threshold,
    impurity,
    n_samples_train,
    n_samples_valid,
    weighted_n_samples_train,
    weighted_n_samples_valid,
):
    # New node index is given by the current number of nodes in the tree
    node_idx = tree.node_count

    if node_idx >= tree.capacity:
        # print("In tree_add_node calling tree_resize with no capacity")
        # tree_add_node
        tree_resize(tree)

    nodes = tree.nodes
    node = nodes[node_idx]
    node["impurity"] = impurity
    node["n_samples_train"] = n_samples_train
    node["n_samples_valid"] = n_samples_valid
    node["weighted_n_samples_train"] = weighted_n_samples_train
    node["weighted_n_samples_valid"] = weighted_n_samples_valid

    # TODO: faudrait remettre ca : if parent != TREE_UNDEFINED:
    if parent != TREE_UNDEFINED:
        if is_left:
            nodes[parent]["left_child"] = node_idx
        else:
            nodes[parent]["right_child"] = node_idx

    if is_leaf:
        pass
        # TODO: ca ne sert a rien ca si ?
        node["left_child"] = TREE_LEAF
        node["right_child"] = TREE_LEAF
        node["feature"] = TREE_UNDEFINED
        node["threshold"] = TREE_UNDEFINED
        node["bin_threshold"] = TREE_UNDEFINED
    else:
        # left_child and right_child will be set later
        node["feature"] = feature
        node["threshold"] = threshold
        node["bin_threshold"] = bin_threshold

    tree.node_count += 1

    return node_idx


@njit
def tree_resize(tree, capacity=max_size_t):
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

    if capacity == max_size_t:
        if tree.capacity == 0:
            capacity = 3  # default initial value
        else:
            capacity = 2 * tree.capacity

    # print("new capacity: ", capacity)
    tree.nodes = resize(tree.nodes, capacity)
    tree.y_pred = resize2d(tree.y_pred, capacity, zeros=True)

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


# TODO: tous les trucs de prediction faut les faire a part comme dans pygbm,
#  on y mettra dedans l'aggregation ? Dans un module _prediction separe

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
    out = tree.y_sum.take(idx_leaves, axis=0)

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
    pass
    # return tree_apply_dense(tree, X)


# @njit
# def tree_apply_dense(tree, X):
#     #     cdef inline np.ndarray _apply_dense(self, object X):
#     #         """Finds the terminal region (=leaf node) for each sample in X."""
#     #
#     #         # Check input
#     #         if not isinstance(X, np.ndarray):
#     #             raise ValueError("X should be in np.ndarray format, got %s"
#     #                              % type(X))
#     #
#     #         if X.dtype != DTYPE:
#     #             raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
#     #
#     #         # Extract input
#     #         cdef const DTYPE_t[:, :] X_ndarray = X
#     #         cdef SIZE_t n_samples = X.shape[0]
#     #
#     #         # Initialize output
#     #         cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
#     #         cdef SIZE_t* out_ptr = <SIZE_t*> out.data
#     #
#     #         # Initialize auxiliary data-structure
#     #         cdef Node* node = NULL
#     #         cdef SIZE_t i = 0
#     #
#     #         with nogil:
#     #             for i in range(n_samples):
#     #                 node = self.nodes
#     #                 # While node not a leaf
#     #                 while node.left_child != _TREE_LEAF:
#     #                     # ... and node.right_child != _TREE_LEAF:
#     #                     if X_ndarray[i, node.feature] <= node.threshold:
#     #                         node = &self.nodes[node.left_child]
#     #                     else:
#     #                         node = &self.nodes[node.right_child]
#     #
#     #                 out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset
#     #
#     #         return out
#
#     # Check input
#     # TODO: ces checks sont inutiles non ? On en fait deja avant dans la classe python
#     # if not isinstance(X, np.ndarray):
#     #     raise ValueError("X should be in np.ndarray format, got %s"
#     #                      % type(X))
#     #
#     # if X.dtype != DTYPE_t:
#     #     raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
#
#     # Extract input
#     # cdef const DTYPE_t[:, :] X_ndarray = X
#     X_ndarray = X
#
#     # cdef SIZE_t n_samples = X.shape[0]
#     n_samples = X.shape[0]
#
#     # Initialize output
#     # cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
#     out = np.zeros((n_samples,), dtype=np_size_t)
#     # cdef SIZE_t* out_ptr = <SIZE_t*> out.data
#
#     # Initialize auxiliary data-structure
#     # cdef Node* node = NULL
#     # cdef SIZE_t i = 0
#
#     # with nogil:
#     nodes = tree.nodes
#
#     for i in range(n_samples):
#         # Index of the leaf containing the sample
#         idx_leaf = 0
#         node = nodes[idx_leaf]
#         # While node not a leaf
#         while node["left_child"] != TREE_LEAF:
#             # ... and node.right_child != TREE_LEAF:
#             if X_ndarray[i, node["feature"]] <= node["threshold"]:
#                 idx_leaf = node["left_child"]
#             else:
#                 idx_leaf = node["right_child"]
#             node = nodes[idx_leaf]
#
#         # out_ptr[i] = <SIZE_t>(node - tree.nodes)  # node offset
#         out[i] = nb_intp(idx_leaf)
#
#     return out


#
# @njit
# def tree_apply_sparse_csr(tree, X):
#     #
#     #     cdef inline np.ndarray _apply_sparse_csr(self, object X):
#     #         """Finds the terminal region (=leaf node) for each sample in sparse X.
#     #         """
#     #         # Check input
#     #         if not isinstance(X, csr_matrix):
#     #             raise ValueError("X should be in csr_matrix format, got %s"
#     #                              % type(X))
#     #
#     #         if X.dtype != DTYPE:
#     #             raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
#     #
#     #         # Extract input
#     #         cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
#     #         cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
#     #         cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr
#     #
#     #         cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
#     #         cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
#     #         cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data
#     #
#     #         cdef SIZE_t n_samples = X.shape[0]
#     #         cdef SIZE_t n_features = X.shape[1]
#     #
#     #         # Initialize output
#     #         cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
#     #                                                        dtype=np.intp)
#     #         cdef SIZE_t* out_ptr = <SIZE_t*> out.data
#     #
#     #         # Initialize auxiliary data-structure
#     #         cdef DTYPE_t feature_value = 0.
#     #         cdef Node* node = NULL
#     #         cdef DTYPE_t* X_sample = NULL
#     #         cdef SIZE_t i = 0
#     #         cdef INT32_t k = 0
#     #
#     #         # feature_to_sample as a data structure records the last seen sample
#     #         # for each feature; functionally, it is an efficient way to identify
#     #         # which features are nonzero in the present sample.
#     #         cdef SIZE_t* feature_to_sample = NULL
#     #
#     #         safe_realloc(&X_sample, n_features)
#     #         safe_realloc(&feature_to_sample, n_features)
#     #
#     #         with nogil:
#     #             memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))
#     #
#     #             for i in range(n_samples):
#     #                 node = self.nodes
#     #
#     #                 for k in range(X_indptr[i], X_indptr[i + 1]):
#     #                     feature_to_sample[X_indices[k]] = i
#     #                     X_sample[X_indices[k]] = X_data[k]
#     #
#     #                 # While node not a leaf
#     #                 while node.left_child != _TREE_LEAF:
#     #                     # ... and node.right_child != _TREE_LEAF:
#     #                     if feature_to_sample[node.feature] == i:
#     #                         feature_value = X_sample[node.feature]
#     #
#     #                     else:
#     #                         feature_value = 0.
#     #
#     #                     if feature_value <= node.threshold:
#     #                         node = &self.nodes[node.left_child]
#     #                     else:
#     #                         node = &self.nodes[node.right_child]
#     #
#     #                 out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset
#     #
#     #             # Free auxiliary arrays
#     #             free(X_sample)
#     #             free(feature_to_sample)
#     #
#     #         return out
#     pass
#
#
# @njit
# def tree_decision_path(tree, X):
#     #     cpdef object decision_path(self, object X):
#     #         """Finds the decision path (=node) for each sample in X."""
#     #         if issparse(X):
#     #             return self._decision_path_sparse_csr(X)
#     #         else:
#     #             return self._decision_path_dense(X)
#     pass
#
#
# @njit
# def tree_decision_path_dense(tree, X):
#     #     cdef inline object _decision_path_dense(self, object X):
#     #         """Finds the decision path (=node) for each sample in X."""
#     #
#     #         # Check input
#     #         if not isinstance(X, np.ndarray):
#     #             raise ValueError("X should be in np.ndarray format, got %s"
#     #                              % type(X))
#     #
#     #         if X.dtype != DTYPE:
#     #             raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
#     #
#     #         # Extract input
#     #         cdef const DTYPE_t[:, :] X_ndarray = X
#     #         cdef SIZE_t n_samples = X.shape[0]
#     #
#     #         # Initialize output
#     #         cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
#     #         cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data
#     #
#     #         cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
#     #                                                    (1 + self.max_depth),
#     #                                                    dtype=np.intp)
#     #         cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data
#     #
#     #         # Initialize auxiliary data-structure
#     #         cdef Node* node = NULL
#     #         cdef SIZE_t i = 0
#     #
#     #         with nogil:
#     #             for i in range(n_samples):
#     #                 node = self.nodes
#     #                 indptr_ptr[i + 1] = indptr_ptr[i]
#     #
#     #                 # Add all external nodes
#     #                 while node.left_child != _TREE_LEAF:
#     #                     # ... and node.right_child != _TREE_LEAF:
#     #                     indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
#     #                     indptr_ptr[i + 1] += 1
#     #
#     #                     if X_ndarray[i, node.feature] <= node.threshold:
#     #                         node = &self.nodes[node.left_child]
#     #                     else:
#     #                         node = &self.nodes[node.right_child]
#     #
#     #                 # Add the leave node
#     #                 indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
#     #                 indptr_ptr[i + 1] += 1
#     #
#     #         indices = indices[:indptr[n_samples]]
#     #         cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
#     #                                                dtype=np.intp)
#     #         out = csr_matrix((data, indices, indptr),
#     #                          shape=(n_samples, self.node_count))
#     #
#     #         return out
#     pass
#
#
# @njit
# def tree_decision_path_sparse_csr(tree, X):
#     #     cdef inline object _decision_path_sparse_csr(self, object X):
#     #         """Finds the decision path (=node) for each sample in X."""
#     #
#     #         # Check input
#     #         if not isinstance(X, csr_matrix):
#     #             raise ValueError("X should be in csr_matrix format, got %s"
#     #                              % type(X))
#     #
#     #         if X.dtype != DTYPE:
#     #             raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
#     #
#     #         # Extract input
#     #         cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
#     #         cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
#     #         cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr
#     #
#     #         cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
#     #         cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
#     #         cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data
#     #
#     #         cdef SIZE_t n_samples = X.shape[0]
#     #         cdef SIZE_t n_features = X.shape[1]
#     #
#     #         # Initialize output
#     #         cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
#     #         cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data
#     #
#     #         cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
#     #                                                    (1 + self.max_depth),
#     #                                                    dtype=np.intp)
#     #         cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data
#     #
#     #         # Initialize auxiliary data-structure
#     #         cdef DTYPE_t feature_value = 0.
#     #         cdef Node* node = NULL
#     #         cdef DTYPE_t* X_sample = NULL
#     #         cdef SIZE_t i = 0
#     #         cdef INT32_t k = 0
#     #
#     #         # feature_to_sample as a data structure records the last seen sample
#     #         # for each feature; functionally, it is an efficient way to identify
#     #         # which features are nonzero in the present sample.
#     #         cdef SIZE_t* feature_to_sample = NULL
#     #
#     #         safe_realloc(&X_sample, n_features)
#     #         safe_realloc(&feature_to_sample, n_features)
#     #
#     #         with nogil:
#     #             memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))
#     #
#     #             for i in range(n_samples):
#     #                 node = self.nodes
#     #                 indptr_ptr[i + 1] = indptr_ptr[i]
#     #
#     #                 for k in range(X_indptr[i], X_indptr[i + 1]):
#     #                     feature_to_sample[X_indices[k]] = i
#     #                     X_sample[X_indices[k]] = X_data[k]
#     #
#     #                 # While node not a leaf
#     #                 while node.left_child != _TREE_LEAF:
#     #                     # ... and node.right_child != _TREE_LEAF:
#     #
#     #                     indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
#     #                     indptr_ptr[i + 1] += 1
#     #
#     #                     if feature_to_sample[node.feature] == i:
#     #                         feature_value = X_sample[node.feature]
#     #
#     #                     else:
#     #                         feature_value = 0.
#     #
#     #                     if feature_value <= node.threshold:
#     #                         node = &self.nodes[node.left_child]
#     #                     else:
#     #                         node = &self.nodes[node.right_child]
#     #
#     #                 # Add the leave node
#     #                 indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
#     #                 indptr_ptr[i + 1] += 1
#     #
#     #             # Free auxiliary arrays
#     #             free(X_sample)
#     #             free(feature_to_sample)
#     #
#     #         indices = indices[:indptr[n_samples]]
#     #         cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
#     #                                                dtype=np.intp)
#     #         out = csr_matrix((data, indices, indptr),
#     #                          shape=(n_samples, self.node_count))
#     #
#     #         return out
#     pass
#
#
# @njit
# def tree_compute_feature_importances(tree, normalize):
#     #     cpdef compute_feature_importances(self, normalize=True):
#     #         """Computes the importance of each feature (aka variable)."""
#     #         cdef Node* left
#     #         cdef Node* right
#     #         cdef Node* nodes = self.nodes
#     #         cdef Node* node = nodes
#     #         cdef Node* end_node = node + self.node_count
#     #
#     #         cdef double normalizer = 0.
#     #
#     #         cdef np.ndarray[np.float64_t, ndim=1] importances
#     #         importances = np.zeros((self.n_features,))
#     #         cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data
#     #
#     #         with nogil:
#     #             while node != end_node:
#     #                 if node.left_child != _TREE_LEAF:
#     #                     # ... and node.right_child != _TREE_LEAF:
#     #                     left = &nodes[node.left_child]
#     #                     right = &nodes[node.right_child]
#     #
#     #                     importance_data[node.feature] += (
#     #                         node.weighted_n_node_samples * node.impurity -
#     #                         left.weighted_n_node_samples * left.impurity -
#     #                         right.weighted_n_node_samples * right.impurity)
#     #                 node += 1
#     #
#     #         importances /= nodes[0].weighted_n_node_samples
#     #
#     #         if normalize:
#     #             normalizer = np.sum(importances)
#     #
#     #             if normalizer > 0.0:
#     #                 # Avoid dividing by zero (e.g., when root is pure)
#     #                 importances /= normalizer
#     #
#     #         return importances
#     pass
#
#
# #     property n_classes:
# #         def __get__(self):
# #             return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)
# #     property children_left:
# #         def __get__(self):
# #             return self._get_node_ndarray()['left_child'][:self.node_count]
# #
# #     property children_right:
# #         def __get__(self):
# #             return self._get_node_ndarray()['right_child'][:self.node_count]
# #
# #     property n_leaves:
# #         def __get__(self):
# #             return np.sum(np.logical_and(
# #                 self.children_left == -1,
# #                 self.children_right == -1))
# #
# #     property feature:
# #         def __get__(self):
# #             return self._get_node_ndarray()['feature'][:self.node_count]
# #
# #     property threshold:
# #         def __get__(self):
# #             return self._get_node_ndarray()['threshold'][:self.node_count]
# #
# #     property impurity:
# #         def __get__(self):
# #             return self._get_node_ndarray()['impurity'][:self.node_count]
# #
# #     property n_node_samples:
# #         def __get__(self):
# #             return self._get_node_ndarray()['n_node_samples'][:self.node_count]
# #
# #     property weighted_n_node_samples:
# #         def __get__(self):
# #             return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]
# #
# #     property value:
# #         def __get__(self):
# #             return self._get_value_ndarray()[:self.node_count]
#
#
# @njit
# def tree_compute_partial_dependence():
#     #     def compute_partial_dependence(self, DTYPE_t[:, ::1] X,
#     #                                    int[::1] target_features,
#     #                                    double[::1] out):
#     #         """Partial dependence of the response on the ``target_feature`` set.
#     #
#     #         For each sample in ``X`` a tree traversal is performed.
#     #         Each traversal starts from the root with weight 1.0.
#     #
#     #         At each non-leaf node that splits on a target feature, either
#     #         the left child or the right child is visited based on the feature
#     #         value of the current sample, and the weight is not modified.
#     #         At each non-leaf node that splits on a complementary feature,
#     #         both children are visited and the weight is multiplied by the fraction
#     #         of training samples which went to each child.
#     #
#     #         At each leaf, the value of the node is multiplied by the current
#     #         weight (weights sum to 1 for all visited terminal nodes).
#     #
#     #         Parameters
#     #         ----------
#     #         X : view on 2d ndarray, shape (n_samples, n_target_features)
#     #             The grid points on which the partial dependence should be
#     #             evaluated.
#     #         target_features : view on 1d ndarray, shape (n_target_features)
#     #             The set of target features for which the partial dependence
#     #             should be evaluated.
#     #         out : view on 1d ndarray, shape (n_samples)
#     #             The value of the partial dependence function on each grid
#     #             point.
#     #         """
#     #         cdef:
#     #             double[::1] weight_stack = np.zeros(self.node_count,
#     #                                                 dtype=np.float64)
#     #             SIZE_t[::1] node_idx_stack = np.zeros(self.node_count,
#     #                                                   dtype=np.intp)
#     #             SIZE_t sample_idx
#     #             SIZE_t feature_idx
#     #             int stack_size
#     #             double left_sample_frac
#     #             double current_weight
#     #             double total_weight  # used for sanity check only
#     #             Node *current_node  # use a pointer to avoid copying attributes
#     #             SIZE_t current_node_idx
#     #             bint is_target_feature
#     #             SIZE_t _TREE_LEAF = TREE_LEAF  # to avoid python interactions
#     #
#     #         for sample_idx in range(X.shape[0]):
#     #             # init stacks for current sample
#     #             stack_size = 1
#     #             node_idx_stack[0] = 0  # root node
#     #             weight_stack[0] = 1  # all the samples are in the root node
#     #             total_weight = 0
#     #
#     #             while stack_size > 0:
#     #                 # pop the stack
#     #                 stack_size -= 1
#     #                 current_node_idx = node_idx_stack[stack_size]
#     #                 current_node = &self.nodes[current_node_idx]
#     #
#     #                 if current_node.left_child == _TREE_LEAF:
#     #                     # leaf node
#     #                     out[sample_idx] += (weight_stack[stack_size] *
#     #                                         self.value[current_node_idx])
#     #                     total_weight += weight_stack[stack_size]
#     #                 else:
#     #                     # non-leaf node
#     #
#     #                     # determine if the split feature is a target feature
#     #                     is_target_feature = False
#     #                     for feature_idx in range(target_features.shape[0]):
#     #                         if target_features[feature_idx] == current_node.feature:
#     #                             is_target_feature = True
#     #                             break
#     #
#     #                     if is_target_feature:
#     #                         # In this case, we push left or right child on stack
#     #                         if X[sample_idx, feature_idx] <= current_node.threshold:
#     #                             node_idx_stack[stack_size] = current_node.left_child
#     #                         else:
#     #                             node_idx_stack[stack_size] = current_node.right_child
#     #                         stack_size += 1
#     #                     else:
#     #                         # In this case, we push both children onto the stack,
#     #                         # and give a weight proportional to the number of
#     #                         # samples going through each branch.
#     #
#     #                         # push left child
#     #                         node_idx_stack[stack_size] = current_node.left_child
#     #                         left_sample_frac = (
#     #                             self.nodes[current_node.left_child].weighted_n_node_samples /
#     #                             current_node.weighted_n_node_samples)
#     #                         current_weight = weight_stack[stack_size]
#     #                         weight_stack[stack_size] = current_weight * left_sample_frac
#     #                         stack_size += 1
#     #
#     #                         # push right child
#     #                         node_idx_stack[stack_size] = current_node.right_child
#     #                         weight_stack[stack_size] = (
#     #                             current_weight * (1 - left_sample_frac))
#     #                         stack_size += 1
#     #
#     #             # Sanity check. Should never happen.
#     #             if not (0.999 < total_weight < 1.001):
#     #                 raise ValueError("Total weight should be 1.0 but was %.9f" %
#     #                                  total_weight)
#     pass
#
#
# # cdef class _PathFinder(_CCPPruneController):
# #     """Record metrics used to return the cost complexity path."""
# #     cdef DOUBLE_t[:] ccp_alphas
# #     cdef DOUBLE_t[:] impurities
# #     cdef UINT32_t count
# #
# #     def __cinit__(self,  int node_count):
# #         self.ccp_alphas = np.zeros(shape=(node_count), dtype=np.float64)
# #         self.impurities = np.zeros(shape=(node_count), dtype=np.float64)
# #         self.count = 0
# #
# #     cdef void save_metrics(self,
# #                            DOUBLE_t effective_alpha,
# #                            DOUBLE_t subtree_impurities) nogil:
# #         self.ccp_alphas[self.count] = effective_alpha
# #         self.impurities[self.count] = subtree_impurities
# #         self.count += 1
# #
# #
