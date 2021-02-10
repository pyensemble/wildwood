"""
This contains all the data structures for holding tree data
"""
from math import exp
import numpy as np
from ._utils import (
    np_bool,
    np_uint8,
    np_size_t,
    np_ssize_t,
    nb_bool,
    nb_size_t,
    nb_ssize_t,
    nb_uint8,
    np_float32,
    nb_float32,
    max_size_t,
    from_dtype,
    njit,
    jitclass,
    resize,
    resize2d,
    log_sum_2_exp
)

import numba

from ._node import node_type, node_dtype

from ._utils import get_type

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
    ("stack", nb_node_record[::1]),
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
        self.stack = np.empty(capacity, dtype=np_node_record)


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
    stack = records.stack
    # Resize the stack if capacity is not enough
    if top >= records.capacity:
        records.capacity = nb_size_t(2 * records.capacity)
        records.stack = resize(stack, records.capacity)

    stack_top = records.stack[top]
    stack_top["start_train"] = start_train
    # print("start_train: ", start_train)
    # print("stack_top['start_train']: ", stack_top["start_train"])
    stack_top["end_train"] = end_train
    stack_top["start_valid"] = start_valid
    stack_top["end_valid"] = end_valid
    stack_top["depth"] = depth
    stack_top["parent"] = parent
    stack_top["is_left"] = is_left
    stack_top["impurity"] = impurity
    stack_top["n_constant_features"] = n_constant_features

    # print("end_valid: ", end_valid)
    # print("stack_top['end_valid']: ", stack_top["end_valid"])

    # We have one more record in the stack
    records.top = top + nb_size_t(1)


@njit
def has_records(records):
    # print("records.top: ", records.top)
    return records.top <= nb_size_t(0)


@njit
def pop_node_record(records):
    # print("================ Begin pop_node_record(records) ================")
    top = records.top
    # print("top: ", top)
    stack = records.stack
    # print("top: ", top)
    # print("stack_: ", stack_)
    # print("top - 1", top-1)
    # print("np_size_t(top - 1):", np_size_t(top - 1))
    stack_record = stack[np_size_t(top - 1)]
    # print(stack_record)
    records.top = nb_size_t(top - 1)
    # print("stack.top: ", stack.top)
    # print("pop_node_record(records):")
    # print(stack_record)
    # print("================ End   pop_node_record(records) ================")

    return (
        stack_record["start_train"],
        stack_record["end_train"],
        stack_record["start_valid"],
        stack_record["end_valid"],
        stack_record["depth"],
        stack_record["parent"],
        stack_record["is_left"],
        stack_record["impurity"],
        stack_record["n_constant_features"]
    )


@njit
def print_records(records):
    # s = "Records("
    # s += "capacity={capacity}".format(capacity=records.capacity)
    # s += ", top={top}".format(top=records.top)
    # s += ")"
    # print(s)
    # print("Records")
    print("****************************************************************")
    print("| Records(capacity=", records.capacity, ", top=", records.top, ")")
    top = 0
    for i in range(records.top):
        print("|", records.stack[i])
    print("****************************************************************")
    # for record in records.stack:
    #     print(record)


def get_records(records):
    import pandas as pd

    stack = records.stack
    columns = [col_name for col_name, _ in spec_node_record]
    # columns = ["left_child"]
    return pd.DataFrame.from_records(
        (
            tuple(node[col] for col in columns)
            for i, node in enumerate(stack)
            if i < records.top
        ),
        columns=columns,
    )




# @njit
# def set_node_tree(nodes, idx, node):
#     """
#     Set a node in an array of nodes at index idx
#
#     Parameters
#     ----------
#     nodes : array of np_node
#         Array containing the nodes in the tree.
#
#     idx : intp
#         Destination index of the node.
#
#     node : NodeTree
#         The node to be inserted.
#     """
#     node_dtype = nodes[idx]
#     if idx != node.node_id:
#         raise ValueError("idx != node.node_id")
#
#     node_dtype["node_id"] = node.node_id
#     node_dtype["parent"] = node.parent
#     node_dtype["left_child"] = node.left_child
#     node_dtype["right_child"] = node.right_child
#     node_dtype["depth"] = node.depth
#     node_dtype["feature"] = node.feature
#     node_dtype["threshold"] = node.threshold
#     node_dtype["bin_threshold"] = node.bin_threshold
#     node_dtype["impurity"] = node.impurity
#     node_dtype["n_samples_train"] = node.n_samples_train
#     node_dtype["n_samples_valid"] = node.n_samples_valid
#     node_dtype["weighted_n_samples_train"] = node.weighted_n_samples_train
#     node_dtype["weighted_n_samples_valid"] = node.weighted_n_samples_valid
#     node_dtype["start_train"] = node.start_train
#     node_dtype["end_train"] = node.end_train
#     node_dtype["start_valid"] = node.start_valid
#     node_dtype["end_valid"] = node.end_valid
#     node_dtype["is_left"] = node.is_left
#     node_dtype["loss_valid"] = node.loss_valid
#     node_dtype["log_weight_tree"] = node.log_weight_tree

# @njit
# def get_node_tree(nodes, idx):
#     """
#     Get node at index idx
#
#     Parameters
#     ----------
#     nodes : array of np_node
#         Array containing the nodes in the tree.
#
#     idx : intp
#         Index of the node to retrieve.
#
#     Returns
#     -------
#     output : NodeTree
#         Retrieved node
#     """
#     # It's a jitclass object
#     node = nodes[idx]
#     return NodeTree(
#         node["node_id"],
#         node["parent"],
#         node["left_child"],
#         node["right_child"],
#         node["depth"],
#         node["feature"],
#         node["threshold"],
#         node["bin_threshold"],
#         node["impurity"],
#         node["n_samples"],
#         node["weighted_n_samples"],
#         node["start_train"],
#         node["end_train"],
#         node["start_valid"],
#         node["end_valid"],
#         node["is_left"],
#     )

#
# # TODO: On a pas vraiment besoin de NodeTree en fait non ?
# @jitclass(spec_node_tree)
# class NodeTree(object):
#     def __init__(
#         self,
#         node_id,
#         parent,
#         left_child,
#         right_child,
#         depth,
#         feature,
#         threshold,
#         bin_threshold,
#         impurity,
#         n_samples_train,
#         n_samples_valid,
#         weighted_n_samples_train,
#         weighted_n_samples_valid,
#         start_train,
#         end_train,
#         start_valid,
#         end_valid,
#         is_left,
#         loss_valid,
#         log_weight_tree
#     ):
#         self.node_id = node_id
#         self.parent = parent
#         self.left_child = left_child
#         self.right_child = right_child
#         self.depth = depth
#         self.feature = feature
#         self.threshold = threshold
#         self.bin_threshold = bin_threshold
#         self.impurity = impurity
#         self.n_samples_train = n_samples_train
#         self.n_samples_valid = n_samples_valid
#         self.weighted_n_samples_train = weighted_n_samples_train
#         self.weighted_n_samples_valid = weighted_n_samples_valid
#
#         self.start_train = start_train
#         self.end_train = end_train
#         self.start_valid = start_valid
#         self.end_valid = end_valid
#         self.is_left = is_left
#         self.loss_valid = loss_valid
#         self.log_weight_tree = log_weight_tree


@njit
def print_node_tree(node):
    node_id = node["node_id"]
    parent = node["parent"]
    left_child = node["left_child"]
    right_child = node["right_child"]
    depth = node["depth"]
    feature = node["feature"]
    threshold = node["threshold"]
    bin_threshold = node["bin_threshold"]
    impurity = node["impurity"]

    n_samples_train = node["n_samples_train"]
    n_samples_valid = node["n_samples_valid"]
    weighted_n_samples_train = node["weighted_n_samples_train"]
    weighted_n_samples_valid = node["weighted_n_samples_valid"]

    s = "Node("
    s += "node_id: {node_id}".format(node_id=node_id)
    s += ", parent: {parent}".format(parent=parent)
    s += ", left_child: {left_child}".format(left_child=left_child)
    s += ", right_child: {right_child}".format(right_child=right_child)
    s += ", depth: {depth}".format(depth=depth)
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
    ("nodes", node_type[::1]),
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
        self.max_depth = nb_size_t(0)
        self.node_count = nb_size_t(0)
        self.capacity = nb_size_t(0)
        # Both values and nodes arrays have zero on the first axis and are resized
        # later when we know the capacity of the tree
        # The array of nodes contained in the tree
        self.nodes = np.empty(0, dtype=node_dtype)
        # The array of y sums or counts for each node
        self.y_pred = np.empty((0, self.n_classes), dtype=np_float32)


TreeType = get_type(Tree)


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

    columns = [
        "node_id",
        "parent",
        "left_child",
        "right_child",
        "is_leaf",
        "depth",
        "feature",
        "threshold",
        "bin_threshold",
        "impurity",
        "n_samples_train",
        "n_samples_valid",
        "weighted_n_samples_train",
        "weighted_n_samples_valid",
        "start_train",
        "end_train",
        "start_valid",
        "end_valid",
        "is_left",
        "loss_valid",
        "log_weight_tree"
    ]

    # columns = [col_name for col_name, _ in np_node_tree]
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
    depth,
    is_left,
    is_leaf,
    feature,
    threshold,
    bin_threshold,
    impurity,
    n_samples_train,
    n_samples_valid,
    w_samples_train,
    w_samples_valid,
    start_train,
    end_train,
    start_valid,
    end_valid,
    loss_valid
):
    # print("================ Begin add_node_tree ================")

    # New node index is given by the current number of nodes in the tree
    node_idx = tree.node_count

    # print("In add_node_tree")
    # print("node_idx >= tree.capacity:", node_idx, tree.capacity)

    if node_idx >= tree.capacity:

        # print("node_idx >= tree.capacity:", node_idx, tree.capacity)
        # print("In tree_add_node calling tree_resize with no capacity")
        # tree_add_node
        tree_resize(tree)

    nodes = tree.nodes
    # print("Adding node in tree node_idx:", node_idx)
    # print("is_leaf:", is_leaf)
    node = nodes[node_idx]

    node["node_id"] = node_idx
    node["parent"] = parent
    node["depth"] = depth
    node["is_leaf"] = is_leaf
    node["impurity"] = impurity
    node["n_samples_train"] = n_samples_train
    node["n_samples_valid"] = n_samples_valid
    node["w_samples_train"] = w_samples_train
    node["w_samples_valid"] = w_samples_valid

    node["start_train"] = start_train
    node["end_train"] = end_train
    node["start_valid"] = start_valid
    node["end_valid"] = end_valid
    node["loss_valid"] = loss_valid

    node["log_weight_tree"] = np.nan

    if parent != TREE_UNDEFINED:
        if is_left:
            nodes[parent]["left_child"] = node_idx
        else:
            nodes[parent]["right_child"] = node_idx

    if is_leaf:
        node["left_child"] = TREE_LEAF
        node["right_child"] = TREE_LEAF
        node["feature"] = TREE_UNDEFINED
        node["threshold"] = TREE_UNDEFINED
        node["bin_threshold"] = TREE_UNDEFINED
    else:
        node["feature"] = feature
        node["threshold"] = threshold
        node["bin_threshold"] = bin_threshold

    tree.node_count += nb_size_t(1)
    # print("================ End   add_node_tree ================")

    return node_idx


@njit
def tree_resize(tree, capacity=max_size_t):
    # print("================ Begin tree_resize ================")

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
        if tree.capacity == nb_size_t(0):
            capacity = nb_size_t(3)  # default initial value
        else:
            capacity = nb_size_t(2 * tree.capacity)

    # print("new capacity: ", capacity)

    # print("resizing nodes...")

    # print("capacity:", capacity)

    tree.nodes = resize(tree.nodes, capacity)

    # new = np.empty(capacity, np_node_record)
    # return

    # print("Done resizing nodes...")

    # print("resizing y_pred...")
    tree.y_pred = resize2d(tree.y_pred, capacity, zeros=True)
    # print("Done resizing y_pred...")

    # value memory is initialised to 0 to enable classifier argmax
    # if capacity > tree.capacity:
    #     memset( < void * > (tree.value + tree.capacity * tree.value_stride), 0,
    #     (capacity - tree.capacity) * tree.value_stride *
    #     sizeof(double))

    # if capacity smaller than node_count, adjust the counter
    if capacity < tree.node_count:
        tree.node_count = capacity

    tree.capacity = capacity

    # print("================ End   tree_resize ================")

    return 0


# TODO: tous les trucs de prediction faut les faire a part comme dans pygbm,
#  on y mettra dedans l'aggregation ? Dans un module _prediction separe


# TODO: pas de jit car numba n'accepte pas l'option axis dans .take (a verifier) mais
#  peut etre qu'on s'en fout en fait ? Et puis ca sera assez rapide


@njit
def tree_predict(tree, X, aggregation, step):
    # Index of the leaves containing the samples in X (note that X has been binned by
    # the forest)

    if aggregation:
        return tree_predict_aggregate(tree, X, step)
    else:
        idx_leaves = tree_apply(tree, X)
        n_samples = X.shape[0]
        # TODO: only binary classification right ?
        out = np.empty((n_samples, 2), dtype=np_float32)
        # Predictions given by each leaf node of the tree
        y_pred = tree.y_pred
        i = 0
        # TODO: idx_leaves.shape[0] == n_samples
        for i in range(n_samples):
            idx_leaf = idx_leaves[i]
            out[i] = y_pred[idx_leaf]

        return out


@njit
def tree_apply(tree, X):
    # TODO: on va supposer que X est deja binnee hein ?
    return tree_apply_dense(tree, X)


@njit
def tree_apply_dense(tree, X):
    # TODO: X is assumed to be binned here
    n_samples = X.shape[0]
    out = np.zeros((n_samples,), dtype=np_size_t)
    nodes = tree.nodes

    for i in range(n_samples):
        # Index of the leaf containing the sample
        idx_leaf = 0
        node = nodes[idx_leaf]
        # While node not a leaf
        while not node["is_leaf"]:
            # ... and node.right_child != TREE_LEAF:
            if X[i, node["feature"]] <= node["bin_threshold"]:
                idx_leaf = node["left_child"]
            else:
                idx_leaf = node["right_child"]
            node = nodes[idx_leaf]

        # out_ptr[i] = <SIZE_t>(node - tree.nodes)  # node offset
        out[i] = nb_size_t(idx_leaf)

    return out



import numba

@numba.jit(nopython=True, nogil=True, locals={"i": nb_size_t, "idx_current": nb_size_t})
def tree_predict_aggregate(tree, X, step):
    n_samples = X.shape[0]
    n_classes = tree.n_classes
    nodes = tree.nodes
    y_pred = tree.y_pred
    out = np.zeros((n_samples, n_classes), dtype=np_float32)

    for i in range(n_samples):
        # print(i)
        # Index of the leaf containing the sample
        # idx_current = nb_size_t(0)
        idx_current = 0
        node = nodes[idx_current]
        # While node not a leaf
        while not node["is_leaf"]:
            if X[i, node["feature"]] <= node["bin_threshold"]:
                idx_current = node["left_child"]
            else:
                idx_current = node["right_child"]
            node = nodes[idx_current]
        # Now idx_current is the index of the leaf node containing X[i]

        # The aggregated prediction of the tree is saved in out[i]
        # y_pred_tree = out[i].view()
        # We first put the predictions of the leaf
        # print(y_pred_tree.shape, y_pred[idx_current].shape)
        # print(idx_current, type(idx_current))
        # print(i, type(i))
        out[i, :] = y_pred[idx_current]
        # Go up in the tree
        idx_current = node["parent"]

        while idx_current != 0:
            # Get the current node
            node = nodes[idx_current]
            # The prediction given by the current node
            node_pred = y_pred[idx_current]
            # The aggregation weights of the current node
            log_weight = step * node["loss_valid"]
            log_weight_tree = node["log_weight_tree"]
            w = exp(log_weight - log_weight_tree)
            # Apply the dark magic recursive formula from CTW
            out[i, :] = 0.5 * w * node_pred + (1 - 0.5 * w) * out[i, :]
            # Go up in the tree
            idx_current = node["parent"]

    return out

# #
# # @njit(void(get_type(TreeClassifier), float32[::1], float32[::1], boolean))
# def tree_classifier_predict(tree, x_t, scores, use_aggregation):
#     nodes = tree.nodes
#     leaf = tree_get_leaf(tree, x_t)
#     if not use_aggregation:
#         node_classifier_predict(tree, leaf, scores)
#         return
#     current = leaf
#     # Allocate once and for all
#     pred_new = np.empty(tree.n_classes, float32)
#     while True:
#         # This test is useless ?
#         if nodes.is_leaf[current]:
#             node_classifier_predict(tree, current, scores)
#         else:
#             weight = nodes.weight[current]
#             log_weight_tree = nodes.log_weight_tree[current]
#             w = exp(weight - log_weight_tree)
#             # Get the predictions of the current node
#             node_classifier_predict(tree, current, pred_new)
#             for c in range(tree.n_classes):
#                 scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c]
#         # Root must be update as well
#         if current == 0:
#             break
#         # And now we go up
#         current = nodes.parent[current]