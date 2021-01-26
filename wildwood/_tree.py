"""
This contains all the data structures for holding tree data
"""

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
def pop_node_record(records):
    top = records.top
    stack = records.stack
    # print("top: ", top)
    # print("stack_: ", stack_)
    # print("top - 1", top-1)
    stack_record = stack[np_size_t(top - 1)]
    records.top = nb_size_t(top - 1)
    # print("stack.top: ", stack.top)
    return stack_record


def print_records(records):
    s = "Records("
    s += "capacity={capacity}".format(capacity=records.capacity)
    s += ", top={top}".format(top=records.top)
    s += ")"
    print(s)


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


# A numpy dtype containing the node information saved in the tree
spec_node_tree = [
    ("node_id", nb_size_t),
    ("parent", nb_ssize_t),
    ("left_child", nb_ssize_t),
    ("right_child", nb_ssize_t),
    ("depth", nb_size_t),
    ("feature", nb_ssize_t),
    ("threshold", nb_float32),
    ("bin_threshold", nb_uint8),
    ("impurity", nb_float32),
    ("n_samples_train", nb_size_t),
    ("n_samples_valid", nb_size_t),
    ("weighted_n_samples_train", nb_float32),
    ("weighted_n_samples_valid", nb_float32),

    # TODO: on ajoute des trucs dont on a pas besoin pour l'entrainement, mais utile
    #  pour debugger
    ("start_train", nb_size_t),
    ("end_train", nb_size_t),
    ("start_valid", nb_size_t),
    ("end_valid", nb_size_t),
    ("is_left", nb_bool),
]


np_node_tree = np.dtype(
    [
        ("node_id", np_size_t),
        ("parent", np_ssize_t),
        ("left_child", np_ssize_t),
        ("right_child", np_ssize_t),
        ("depth", np_size_t),
        ("feature", np_ssize_t),
        ("threshold", np_float32),
        ("bin_threshold", np_uint8),
        ("impurity", np_float32),
        ("n_samples_train", np_size_t),
        ("n_samples_valid", np_size_t),
        ("weighted_n_samples_train", np_float32),
        ("weighted_n_samples_valid", np_float32),

        ("start_train", np_size_t),
        ("end_train", np_size_t),
        ("start_valid", np_size_t),
        ("end_valid", np_size_t),
        ("is_left", np_bool),
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
    if idx != node.node_id:
        raise ValueError("idx != node.node_id")

    node_dtype["node_id"] = node.node_id
    node_dtype["parent"] = node.parent
    node_dtype["left_child"] = node.left_child
    node_dtype["right_child"] = node.right_child
    node_dtype["depth"] = node.depth
    node_dtype["feature"] = node.feature
    node_dtype["threshold"] = node.threshold
    node_dtype["bin_threshold"] = node.bin_threshold
    node_dtype["impurity"] = node.impurity
    node_dtype["n_samples_train"] = node.n_samples_train
    node_dtype["n_samples_valid"] = node.n_samples_valid
    node_dtype["weighted_n_samples_train"] = node.weighted_n_samples_train
    node_dtype["weighted_n_samples_valid"] = node.weighted_n_samples_valid
    node_dtype["start_train"] = node.start_train
    node_dtype["end_train"] = node.end_train
    node_dtype["start_valid"] = node.start_valid
    node_dtype["end_valid"] = node.end_valid
    node_dtype["is_left"] = node.is_left


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
        node["node_id"],
        node["parent"],
        node["left_child"],
        node["right_child"],
        node["depth"],
        node["feature"],
        node["threshold"],
        node["bin_threshold"],
        node["impurity"],
        node["n_samples"],
        node["weighted_n_samples"],
        node["start_train"],
        node["end_train"],
        node["start_valid"],
        node["end_valid"],
        node["is_left"]
    )


# TODO: do we really need a dataclass for this ? a namedtuple instead ? or juste a
#  dtype ?


@jitclass(spec_node_tree)
class NodeTree(object):
    def __init__(
        self,
        node_id,
        parent,
        left_child,
        right_child,
        depth,
        feature,
        threshold,
        bin_threshold,
        impurity,
        n_samples_train,
        n_samples_valid,
        weighted_n_samples_train,
        weighted_n_samples_valid,

        start_train,
        end_train,
        start_valid,
        end_valid,
        is_left
    ):
        self.node_id = node_id
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.feature = feature
        self.threshold = threshold
        self.bin_threshold = bin_threshold
        self.impurity = impurity
        self.n_samples_train = n_samples_train
        self.n_samples_valid = n_samples_valid
        self.weighted_n_samples_train = weighted_n_samples_train
        self.weighted_n_samples_valid = weighted_n_samples_valid

        self.start_train = start_train
        self.end_train = end_train
        self.start_valid = start_valid
        self.end_valid = end_valid
        self.is_left = is_left


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

    columns = [
        "node_id",
        "parent",
        "left_child",
        "right_child",
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
        "is_left"
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
    weighted_n_samples_train,
    weighted_n_samples_valid,

    start_train,
    end_train,
    start_valid,
    end_valid,
):
    # New node index is given by the current number of nodes in the tree
    node_idx = tree.node_count

    if node_idx >= tree.capacity:
        # print("In tree_add_node calling tree_resize with no capacity")
        # tree_add_node
        tree_resize(tree)

    nodes = tree.nodes
    node = nodes[node_idx]

    node["node_id"] = node_idx
    node["parent"] = parent
    node["depth"] = depth
    node["impurity"] = impurity
    node["n_samples_train"] = n_samples_train
    node["n_samples_valid"] = n_samples_valid
    node["weighted_n_samples_train"] = weighted_n_samples_train
    node["weighted_n_samples_valid"] = weighted_n_samples_valid

    node["start_train"] = start_train
    node["end_train"] = end_train
    node["start_valid"] = start_valid
    node["end_valid"] = end_valid

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


# TODO: pas de jit car numba n'accepte pas l'option axis dans .take (a verifier) mais
#  peut etre qu'on s'en fout en fait ? Et puis ca sera assez rapide


@njit
def tree_predict(tree, X):
    # Index of the leaves containing the samples in X (note that X has been binned by
    # the forest)
    idx_leaves = tree_apply(tree, X)
    n_samples = X.shape[0]
    # TODO: only binary classification right ?
    out = np.empty((n_samples, 2))
    # Predictions given by each leaf node of the tree
    y_pred = tree.y_pred
    i = 0
    # TODO: idx_leaves.shape[0] == n_samples
    for i in range(n_samples):
        idx_leaf = idx_leaves[i]
        out[i] = y_pred[idx_leaf]

    # out = tree.y_pred.take(idx_leaves, axis=0)
    # if tree.n_outputs == 1:
    #     out = out.reshape(X.shape[0], tree.max_n_classes)
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
        while node["left_child"] != TREE_LEAF:
            # ... and node.right_child != TREE_LEAF:
            if X[i, node["feature"]] <= node["bin_threshold"]:
                idx_leaf = node["left_child"]
            else:
                idx_leaf = node["right_child"]
            node = nodes[idx_leaf]

        # out_ptr[i] = <SIZE_t>(node - tree.nodes)  # node offset
        out[i] = nb_size_t(idx_leaf)

    return out
