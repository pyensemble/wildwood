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

import numpy as np
from numba import jit, from_dtype, void, boolean, uint8, intp, uintp, float32
from numba.types import Tuple
from numba.experimental import jitclass

from ._node import NodeContext, compute_node_context

from ._splitting import (
    find_node_split,
    split_indices,
)


from ._tree import (
    add_node_tree,
    tree_resize,
    TREE_UNDEFINED,
)

from ._utils import resize, log_sum_2_exp, get_type

INITIAL_STACK_SIZE = uintp(10)


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
    return records.top <= 0


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


# TODO: clean this function. 1. Remove useless stuff, clean the code. 2. Put locals
#  3. Put docstring


@jit(
    fastmath=False,
    nopython=True,
    nogil=True,
    locals={
        "init_capacity": uintp,
        # "n_samples_train": uintp,
        # "n_samples_valid": uintp,
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
        # split
        "bin": uint8,
        "feature": uintp,
        "found_split": boolean,
        "threshold": float32,
        "weighted_n_samples_valid": float32,
        "pos_train": uintp,
        "pos_valid": uintp,
        "aggregation": boolean,
        "step": float32,
        "node_count": intp,
        "node_idx": intp
    },
)
def grow(tree, tree_context, node_context):
    # Initialize the tree capacity
    init_capacity = 2047
    tree_resize(tree, init_capacity)
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

    # TODO: this option will come for the forest later
    min_samples_split = 2

    while not has_records(records):
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

        # We don't split a node if it contains less than min_samples_split training
        # or validation samples
        is_leaf = (node_context.n_samples_train < min_samples_split) or (
            node_context.n_samples_valid < min_samples_split
        )

        # We don't split a node if it's pure: whenever it's impurity computed on
        # training samples is less than min_impurity split
        min_impurity_split = 0.0
        is_leaf = is_leaf or (impurity <= min_impurity_split)

        # TODO: put back the min_impurity_split option

        if is_leaf:
            split = None
            bin = 0
            feature = 0
            found_split = False
            # TODO: pourquoi on mettrai impurity = infini ici ?
        else:
            split = find_node_split(tree_context, node_context)
            bin = split.bin
            feature = split.feature
            found_split = split.found_split

        # If we did not find a split then the node is a leaf, since we can't split it
        is_leaf = is_leaf or not found_split
        # TODO: correct this when actually using the threshold instead of
        #  bin_threshold
        threshold = 0.42
        weighted_n_samples_valid = 42.0

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
            weighted_n_samples_valid,
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
        )

        # Save in the tree the predictions of the node
        tree.y_pred[node_id, :] = node_context.y_pred

        if not is_leaf:
            # If the node is not a leaf, we update partition_train and
            # partition_valid so that they contain training and validation indices of
            # nodes in a contiguous way.
            pos_train, pos_valid = split_indices(
                tree_context, split, start_train, end_train, start_valid, end_valid
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
                split.impurity_left,
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
                split.impurity_right,
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

    aggregation = tree_context.aggregation
    # step = tree_context.step
    step = 1.0

    # Since the tree is grown in a depth-first fashion, we know that if we iterate
    # through the nodes in reverse order, we'll always iterate over childs before
    # iteration over parents.
    node_count = tree.node_count

    # TODO: mettre ca dans une fonction a part...
    if aggregation:
        for node_idx in range(node_count - 1, -1, -1):
            node = tree.nodes[node_idx]
            if node["is_leaf"]:
                # If the node is a leaf, the logarithm of its tree weight is simply
                  # step * loss
                node["log_weight_tree"] = step * node["loss_valid"]
            else:
                # If the node is not a leaf, then we apply context tree weighting
                weight = step * node["loss_valid"]
                left_child = intp(node["left_child"])
                right_child = intp(node["right_child"])
                # print("left_child: ", left_child, ", right_child: ", right_child)
                log_weight_tree_left = tree.nodes[left_child]["log_weight_tree"]
                log_weight_tree_right = tree.nodes[right_child]["log_weight_tree"]
                node["log_weight_tree"] = log_sum_2_exp(
                    weight, log_weight_tree_left + log_weight_tree_right
                )

            node_idx -= 1
