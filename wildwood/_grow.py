# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains some tools to grow a tree: mainly a Records dataclass and a grow
function.
Records is a last-in first-out stack of Record containing partial
information about nodes that will be split or transformed in leaves.
The grow function is the main entry point that grows the decision tree and performs
aggregation.
"""

from math import log

import numpy as np
from numba import jit, from_dtype, void, boolean, uint8, intp, uintp, float32, optional
from numba.types import Tuple
from numba.experimental import jitclass

from ._split import find_node_split, split_indices
from ._node import node_type
from ._tree import add_node_tree, resize_tree, TREE_UNDEFINED, TreeClassifierType
from ._tree_context import TreeClassifierContextType
from ._utils import resize, log_sum_2_exp, get_type

INITIAL_STACK_SIZE = uintp(10)

eps = np.finfo("float32").eps

record_dtype = np.dtype(
    [
        ("parent", np.intp),
        ("depth", np.uintp),
        ("is_left", np.bool),
        ("impurity", np.float32),
        ("start_train", np.uintp),
        ("end_train", np.uintp),
        ("start_valid", np.uintp),
        ("end_valid", np.uintp),
    ]
)

record_type = from_dtype(record_dtype)


records_type = [
    ("capacity", uintp),
    ("top", intp),
    ("stack", record_type[::1]),
]


@jitclass(records_type)
class Records(object):
    """A simple LIFO (last in, first out) stack containing partial information about
    the nodes to be later split or transformed into leaves.

    Attributes
    ----------
    capacity : int
        The number of elements the stack can hold. If more is necessary then
        self.stack is resized

    top : int
        The number of elements currently on the stack

    stack : ndarray
        An array of shape (capacity,) and record_dtype dtype containing the
        current data in the stack
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.top = 0
        self.stack = np.empty(capacity, dtype=record_dtype)


RecordsType = get_type(Records)


@jit(
    void(RecordsType, intp, uintp, boolean, float32, uintp, uintp, uintp, uintp),
    nopython=True,
    nogil=True,
    locals={"stack_top": record_type},
)
def push_record(
    records,
    parent,
    depth,
    is_left,
    impurity,
    start_train,
    end_train,
    start_valid,
    end_valid,
):
    """Adds a node record in the records stack with the given node attributes.

    Parameters
    ----------
    records : Records
        A records dataclass containing the stack of node records

    parent : int
        Index of the parent node

    depth : int
        Depth of the node in the tree

    is_left : bool
        True if the node is a left child, False otherwise

    impurity : float
        Impurity of the node. Used to avoid to split a "pure" node (with impurity=0).

    start_train : int
        Index of the first training sample in the node. We have that
        partition_train[start_train:end_train] contains the indexes of the node's
        training samples

    end_train : int
        End-index of the slice containing the node's training samples indexes

    start_valid : int
        Index of the first validation (out-of-the-bag) sample in the node. We have
        that partition_valid[start_valid:end_valid] contains the indexes of the
        node's validation samples

    end_valid : int
        End-index of the slice containing the node's validation samples indexes
    """
    # Resize the stack if capacity is not enough
    if records.top >= records.capacity:
        records.capacity *= 2
        records.stack = resize(records.stack, records.capacity)

    stack_top = records.stack[records.top]
    stack_top["parent"] = parent
    stack_top["depth"] = depth
    stack_top["is_left"] = is_left
    stack_top["impurity"] = impurity
    stack_top["start_train"] = start_train
    stack_top["end_train"] = end_train
    stack_top["start_valid"] = start_valid
    stack_top["end_valid"] = end_valid
    records.top += 1


@jit(boolean(RecordsType), nopython=True, nogil=True)
def has_records(records):
    """Tests if the stack of records contain remaining records.

    Parameters
    ----------
    records : Records
        A records dataclass containing the stack of node records

    Returns
    -------
    output : bool
        Returns True if there are remaining records in the stack, False otherwise
    """
    return records.top > 0


@jit(
    Tuple((intp, uintp, boolean, float32, uintp, uintp, uintp, uintp))(RecordsType),
    nopython=True,
    nogil=True,
    locals={"stack_top": record_type},
)
def pop_node_record(records):
    """Pops (removes and returns) a node record from the stack of records.

    Parameters
    ----------
    records : Records
        A records dataclass containing the stack of node records

    Returns
    -------
    output : tuple
        Outputs a tuple with eight elements containing the attributes of the node
        removed from the stack. There attributes are as follows:

        - parent : int, Index of the parent node
        - depth : int, Depth of the node in the tree
        - is_left : bool, True if the node is a left child, False otherwise
        - impurity : float, Impurity of the node. Used to avoid to split a "pure"
          node (with impurity=0).
        - start_train : int, Index of the first training sample in the node. We have
          that partition_train[start_train:end_train] contains the indexes of the node's
          training samples
        - end_train : int, End-index of the slice containing the node's training
          samples indexes
        - start_valid : int, Index of the first validation (out-of-the-bag) sample in
          the node. We have that partition_valid[start_valid:end_valid] contains the
          indexes of the node's validation samples
        - end_valid : int, End-index of the slice containing the node's validation
          samples indexes
    """
    records.top -= 1
    stack_top = records.stack[records.top]
    return (
        stack_top["parent"],
        stack_top["depth"],
        stack_top["is_left"],
        stack_top["impurity"],
        stack_top["start_train"],
        stack_top["end_train"],
        stack_top["start_valid"],
        stack_top["end_valid"],
    )


# TODO: we do not compute the correct information gain of the root node, but it's
#  useless

# TODO: for now, we don't specify the signature of grow, since it has first-order
#  functions as argument and I don't know how specify the signature of those


@jit(
    fastmath=False,
    nopython=True,
    nogil=True,
    locals={
        "init_capacity": uintp,
        "records": RecordsType,
        "parent": intp,
        "depth": uintp,
        "is_left": boolean,
        "impurity": float32,
        "start_train": uintp,
        "end_train": uintp,
        "start_valid": uintp,
        "end_valid": uintp,
        "min_samples_split": uintp,
        "min_impurity_split": float32,
        "is_leaf": boolean,
        "bin": uint8,
        "feature": uintp,
        "found_split": boolean,
        "is_split_categorical": boolean,
        "bin_partition": optional(uint8[::1]),
        "bin_partition_size": uint8,
        "threshold": float32,
        "w_samples_valid": float32,
        "pos_train": uintp,
        "pos_valid": uintp,
        "aggregation": boolean,
        "step": float32,
        "node_count": intp,
    },
)
def grow(
    tree,
    tree_context,
    node_context,
    compute_node_context,
    find_best_split_along_feature,
    best_split,
):
    """This function grows a tree in the forest, it is the main entry point for the
    computations when fitting a forest.

    Parameters
    ----------
    tree : TreeClassifier or TreeRegressor
        The tree object which holds nodes and prediction data

    tree_context : TreeClassifierContext or TreeRegressorContext
        A tree context which will contain tree-level information that is useful to
        find splits

    node_context : NodeClassifierContext or NodeRegressorContext
        A node context which will contain node-level information that is useful to
        find splits

    compute_node_context : function
        The function used to compute the node context

    find_best_split_along_feature : function
        The function used to find the best split along a feature

    best_split : SplitClassifier or SplitRegressor
        A temporary split object used for split computations
    """
    # Initialize the tree capacity
    init_capacity = 2047
    resize_tree(tree, init_capacity)
    # Create the stack of node records
    records = Records(INITIAL_STACK_SIZE)

    # Let us first define all the attributes of root
    parent = TREE_UNDEFINED
    depth = 0
    is_left = False
    impurity = np.inf
    start_train = 0
    end_train = tree_context.n_samples_train
    start_valid = 0
    end_valid = tree_context.n_samples_valid

    aggregation = tree_context.aggregation

    push_record(
        records,
        parent,
        depth,
        is_left,
        impurity,
        start_train,
        end_train,
        start_valid,
        end_valid,
    )

    min_samples_split = tree_context.min_samples_split

    while has_records(records):
        # Get information about the current node
        (
            parent,
            depth,
            is_left,
            impurity,
            start_train,
            end_train,
            start_valid,
            end_valid,
        ) = pop_node_record(records)

        # Initialize the node context, this computes the node statistics
        compute_node_context(
            tree_context, node_context, start_train, end_train, start_valid, end_valid
        )

        # TODO: add the max_depth option using something like
        # is_leaf = is_leaf or (depth >= max_depth)
        # This node is a terminal leaf, we won't try to split it

        if aggregation:
            # When using aggregation, we don't split a node if it contains less than
            # min_samples_split training or validation samples
            is_leaf = (node_context.n_samples_train < min_samples_split) or (
                node_context.n_samples_valid < min_samples_split
            )
        else:
            # When not using aggregation, we don't split a node if it contains less
            # than min_samples_split training samples
            is_leaf = node_context.n_samples_train < min_samples_split

        # TODO: check that it's indeed the case
        # We don't split a node if it's pure: whenever its impurity computed on
        # training samples is less than min_impurity split
        min_impurity_split = 0.0
        is_leaf = is_leaf or (impurity <= min_impurity_split)

        # TODO: put back the min_impurity_split option

        if is_leaf:
            bin = 0
            feature = 0
            found_split = False
            is_split_categorical = False
            bin_partition = None
            bin_partition_size = 0
        else:
            find_node_split(
                tree_context, node_context, find_best_split_along_feature, best_split,
            )
            bin = best_split.bin_threshold
            feature = best_split.feature
            found_split = best_split.found_split
            is_split_categorical = best_split.is_split_categorical
            bin_partition = best_split.bin_partition
            bin_partition_size = best_split.bin_partition_size

        # If we did not find a split then the node is a leaf, since we can't split it
        is_leaf = is_leaf or not found_split

        # TODO: correct this when actually using the threshold instead of
        #  bin_threshold
        threshold = 0.42

        node_id = add_node_tree(
            # The tree
            tree,
            # Index of the parent node
            parent,
            # Depth of the node
            depth,
            # Is the node a left child ?
            is_left,
            # Is the node a leaf ?
            is_leaf,
            # The feature used for splitting
            feature,
            # NOT USED FOR NOW
            threshold,
            # The bin threshold used for splitting
            bin,
            # Impurity of the node
            impurity,
            # Number of training samples
            node_context.n_samples_train,
            # Number of validation samples
            node_context.n_samples_valid,
            # Weighted number of training samples
            node_context.w_samples_train,
            # NOT USED FOR NOW
            node_context.w_samples_valid,
            # Index of the first training sample in the node
            start_train,
            # End-index of the slice containing the node's training samples
            end_train,
            # Index of the first validation (out-of-the-bag) sample in the node.
            start_valid,
            # End-index of the slice containing the node's validation samples indexes
            end_valid,
            # Validation loss of the node, computed on validation samples
            node_context.loss_valid,
            # Is the split on a categorical feature?
            is_split_categorical,
            # Whenever the split is on a categorical feature, ndarray such that
            #   bins within `bin_partition[:bin_partition_size]` go to left child
            #   all other bins go to right child
            bin_partition,
            # Whenever the split is on a categorical feature, index such that
            #   bins within `bin_partition[:bin_partition_size]` go to left child
            #   all other bins go to right child;
            # If it is a leaf, bin_partition_size=0
            bin_partition_size,
        )
        # Save in the tree the predictions of the node (works both for regression
        # where node_context.y_pred is a float32 and for classification where
        # node_context.y_pred is a ndarray of shape (n_classes,)
        tree.y_pred[node_id] = node_context.y_pred

        if not is_leaf:
            # If the node is not a leaf, we update partition_train and
            # partition_valid so that they contain training and validation indices of
            # nodes in a contiguous way.
            pos_train, pos_valid = split_indices(
                tree_context, best_split, start_train, end_train, start_valid, end_valid
            )

            # If the node is not a leaf, we add both childs in the node records,
            # so that they can be added in the tree and eventually be split as well.

            # This adds the left child
            push_record(
                # The stack containing the node records
                records,
                # The parent is the previous node_id
                node_id,
                # depth is increased by one
                depth + 1,
                # This is a left child (is_left=True)
                True,
                # Impurities of the childs are kept in the split information
                best_split.impurity_left,
                # start_train of the left child is the same as the parent's
                start_train,
                # end_train of the left child is at the split's position
                pos_train,
                # start_valid of the left child is the same as the parent's
                start_valid,
                # end_valid of the left child is as the split's position
                pos_valid,
            )

            # This adds the right child
            push_record(
                # The stack containing the node records
                records,
                # The parent is the previous node_id
                node_id,
                # depth is increased by one
                depth + 1,
                # This is a right child (is_left=False)
                False,
                # Impurities of the childs are kept in the split information
                best_split.impurity_right,
                # start_train of the right child is at the split's position
                pos_train,
                # end_train of the right child is the same as the parent's
                end_train,
                # start_valid of the right child is at the split's position
                pos_valid,
                # end_valid of the right child is the same as the parent's
                end_valid,
            )

    # We finished to grow the tree. Now, we can compute the tree's aggregation weights.
    step = tree_context.step

    # Since the tree is grown in a depth-first fashion, we know that if we iterate
    # through the nodes in reverse order, we'll always iterate over childs before
    # iteration over parents.
    node_count = tree.node_count

    if aggregation:
        compute_tree_weights(tree.nodes, node_count, step)


@jit(
    [void(node_type[:], intp, float32)],
    fastmath=False,
    nopython=True,
    nogil=True,
    locals={
        "node_idx": intp,
        "node": node_type,
        "loss": float32,
        "left_child": intp,
        "right_child": intp,
        "log_weight_tree_left": float32,
        "log_weight_tree_right": float32,
    },
)
def compute_tree_weights(nodes, node_count, step):
    """Compute tree weights required to apply aggregation with exponential weights
    over all subtrees for the predictions

    Parameters
    ----------
    nodes : ndarray
        A numpy array containing the nodes data

    node_count : int
        Number of nodes in the tree

    step : float
        Step-size used for the computation of the aggregation weights

    References
    ----------
    This corresponds to Algorithm 1 in WildWood's paper
    TODO: Insert reference here
    """
    for node_idx in range(node_count - 1, -1, -1):
        node = nodes[node_idx]
        if node["is_leaf"]:
            # If the node is a leaf, the logarithm of its tree weight is simply
            #   step * loss
            node["log_weight_tree"] = -step * node["loss_valid"]
        else:
            # If the node is not a leaf, then we apply context tree weighting
            loss = -step * node["loss_valid"]
            left_child = intp(node["left_child"])
            right_child = intp(node["right_child"])
            log_weight_tree_left = nodes[left_child]["log_weight_tree"]
            log_weight_tree_right = nodes[right_child]["log_weight_tree"]
            node["log_weight_tree"] = log_sum_2_exp(
                loss, log_weight_tree_left + log_weight_tree_right
            )


@jit(
    void(TreeClassifierType, TreeClassifierContextType, float32),
    nopython=True,
    nogil=True,
    boundscheck=False,
    locals={
        "nodes": node_type[::1],
        "n_classes": uintp,
        "y_pred": float32[::1],
        "y": float32[::1],
        "sample_weights": float32[::1],
        "partition_train": uintp[::1],
        "partition_valid": uintp[::1],
        "node_count": uintp,
        "node": node_type,
        "w_samples_train": float32,
        "train_indices": uintp[::1],
        "label": uintp,
        "sample_weight": float32,
        "valid_indices": uintp[::1],
        "loss_valid": float32,
        "w_samples_valid": float32,
    },
)
def recompute_node_predictions(tree, tree_context, dirichlet):
    """This function recomputes the node predictions and validation loss of nodes.
    This is triggered by a change of the dirichlet parameter.

    Parameters
    ----------
    tree : TreeClassifier
        The tree object which holds nodes and prediction data

    tree_context : TreeClassifierContext
        A tree context which will contain tree-level information that is useful to
        find splits

    dirichlet : float
        The dirichlet parameter

    """
    nodes = tree.nodes
    n_classes = tree.n_classes
    y_pred = np.zeros(n_classes, dtype=np.float32)
    y = tree_context.y
    sample_weights = tree_context.sample_weights
    partition_train = tree_context.partition_train
    partition_valid = tree_context.partition_valid

    # Recompute y_pred with the new dirichlet parameter
    for node_idx, node in enumerate(nodes[: tree.node_count]):
        w_samples_train = 0.0
        train_indices = partition_train[node["start_train"] : node["end_train"]]
        y_pred.fill(0.0)
        for sample in train_indices:
            label = uintp(y[sample])
            sample_weight = sample_weights[sample]
            w_samples_train += sample_weight
            y_pred[label] += sample_weight

        for k in range(n_classes):
            y_pred[k] = (y_pred[k] + dirichlet) / (
                w_samples_train + n_classes * dirichlet
            )

        tree.y_pred[node_idx, :] = y_pred

        # recompute valid_losses
        valid_indices = partition_valid[node["start_valid"] : node["end_valid"]]
        loss_valid = 0.0
        w_samples_valid = 0.0

        for sample in valid_indices:
            sample_weight = sample_weights[sample]
            w_samples_valid += sample_weight
            label = uintp(y[sample])
            # TODO: aggregation loss is hard-coded here. Call a function instead
            #  when implementing other losses
            loss_valid -= sample_weight * log(y_pred[label])

        node["loss_valid"] = loss_valid

    # TODO : compute tree weights anyway ? the program can be duped by switching
    #  aggregation off, changing dirichlet and switching aggregation back on
    if tree_context.aggregation:
        compute_tree_weights(tree.nodes, tree.node_count, tree_context.step)
