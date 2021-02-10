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
from numba import jit, njit, from_dtype, void, boolean, intp, uintp, float32
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
        ("parent", np.uintp),
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
    ("top", uintp),
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
        self.top = uintp(0)
        self.stack = np.empty(capacity, dtype=record_dtype)


RecordsType = get_type(Records)


@jit(
    void(RecordsType, uintp, uintp, boolean, float32, uintp, uintp, uintp, uintp),
    nopython=True,
    nogil=True,
    boundscheck=False,
    locals={"top": uintp, "stack": record_type[::1], "stack_top": record_type},
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
    top = records.top
    stack = records.stack
    # Resize the stack if capacity is not enough
    if top >= records.capacity:
        records.capacity = uintp(2 * records.capacity)
        records.stack = resize(stack, records.capacity)

    stack_top = records.stack[top]
    stack_top["parent"] = parent
    stack_top["depth"] = depth
    stack_top["is_left"] = is_left
    stack_top["impurity"] = impurity
    stack_top["start_train"] = start_train
    stack_top["end_train"] = end_train
    stack_top["start_valid"] = start_valid
    stack_top["end_valid"] = end_valid
    records.top += 1


# TODO: ICI ICI ICI ICI attention on dirait que cette fonction plantes !!!!
@jit(boolean(RecordsType), nopython=True, nogil=True, boundscheck=False)
def has_records(records):
    # print("records.top: ", records.top)
    return records.top <= 0


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
    stack_record = stack[uintp(top - 1)]
    # print(stack_record)
    records.top = uintp(top - 1)
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
        # stack_record["n_constant_features"]
    )


# @njit
# def print_records(records):
#     # s = "Records("
#     # s += "capacity={capacity}".format(capacity=records.capacity)
#     # s += ", top={top}".format(top=records.top)
#     # s += ")"
#     # print(s)
#     # print("Records")
#     print("****************************************************************")
#     print("| Records(capacity=", records.capacity, ", top=", records.top, ")")
#     top = 0
#     for i in range(records.top):
#         print("|", records.stack[i])
#     print("****************************************************************")
#     # for record in records.stack:
#     #     print(record)
#
#
# def get_records(records):
#     import pandas as pd
#
#     stack = records.stack
#     columns = [col_name for col_name, _ in record_type]
#     # columns = ["left_child"]
#     return pd.DataFrame.from_records(
#         (
#             tuple(node[col] for col in columns)
#             for i, node in enumerate(stack)
#             if i < records.top
#         ),
#         columns=columns,
#     )
#
#


@njit
def grow(tree, tree_context, node_context):
    # print("**** Inside grow")

    # This is the output split
    # TODO: redo this ?
    # if tree.max_depth <= 10:
    #     init_capacity = (2 ** (tree.max_depth + 1)) - 1
    # else:
    #     init_capacity = 2047

    init_capacity = 2

    # print("tree.max_depth: ", tree.max_depth)
    #
    # print("tree.max_depth: ", tree.max_depth)
    # print("tree_resize(tree, init_capacity)")
    # print("init_capacity: ", init_capacity)
    tree_resize(tree, init_capacity)
    # print("Done.")

    # print(tree.nodes.shape)
    # print(tree.y_pred.shape)

    # TODO: get back parameters from the tree
    # TODO: decide a posteriori what goes into SplittingContext and Tree ?

    # TODO: creer un local_context ici

    n_samples_train = tree_context.n_samples_train
    n_samples_valid = tree_context.n_samples_valid

    max_depth_seen = -1

    # print("n_samples_train:", n_samples_train)
    # print("n_samples_valid:", n_samples_valid)

    # TODO: explain why there is a stack and a tree
    records = Records(INITIAL_STACK_SIZE)

    # print(records)

    # Let us first define all the attributes of root
    start_train = uintp(0)

    end_train = uintp(n_samples_train)
    start_valid = uintp(0)
    end_valid = uintp(n_samples_valid)
    depth = uintp(0)
    # root as no parent
    parent = TREE_UNDEFINED
    is_left = False
    impurity = np.inf
    n_constant_features = 0

    # print("start_train:", start_train)
    # print("end_valid:", end_valid)

    # Add the root node into the stack
    # print("Pushing root in the stack")

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
    # print("Done pushing root in the stack")

    # TODO: this option will come for the forest later
    min_samples_split = 2

    # print_records(records)
    # print(get_records(records))

    while not has_records(records):
        # print("================ while not has_records(records) ================")

        # Get information about the current node
        # TODO: plutot creer un node ici

        # print_records(records)

        # Get information about the current node
        (
            start_train,
            end_train,
            start_valid,
            end_valid,
            depth,
            parent,
            is_left,
            impurity,
            # n_constant_features,
        ) = pop_node_record(records)

        # node_record = pop_node_record(records)
        # print("node_record = pop_node_record(records)")
        # print("node_record: ", node_record)

        # return

        # parent = node_record["parent"]
        # depth = node_record["depth"]
        # start_train = node_record["start_train"]
        # end_train = node_record["end_train"]
        # start_valid = node_record["start_valid"]
        # end_valid = node_record["end_valid"]
        # is_left = node_record["is_left"]
        # # TODO: we get the impurity from the record
        # impurity = node_record["impurity"]

        # print("parent: ", parent)
        # print("depth: ", depth)
        # print("start_train: ", start_train)
        # print("end_train: ", end_train)
        # print("start_valid: ", start_valid)
        # print("end_valid: ", end_valid)
        # print("is_left: ", is_left)
        #
        # print(node_record)
        # return

        # print("records.top: ", records.top)

        # Initialize the node context, this computes the node statistics
        compute_node_context(
            tree_context, node_context, start_train, end_train, start_valid, end_valid
        )

        # print("node_context.n_samples_train: ", node_context.n_samples_train)
        # print("node_context.n_samples_valid: ", node_context.n_samples_valid)

        # n_constant_features = node_record["n_constant_features"]
        # Mettre a jour le local_context ici ?

        # splitter.node_reset(start, end, &weighted_n_node_samples)
        # weighted_n_node_samples = splitter_node_reset(splitter, start, end)
        # TODO: for now, simple stopping criterion. We'll add more options later
        # is_leaf = (
        #         depth >= max_depth
        #         or n_samples_train_node < min_samples_split
        #         or n_samples_node < 2 * min_samples_leaf
        #         or weighted_n_node_samples < 2 * min_weight_leaf
        # )

        # This node is a terminal leaf, we won't try to split it

        # We won't split a node if the number of training or validation samples it
        # contains is too small. If the number of validation sample is too small,
        # we won't use nodes with no validation sample anyway !

        # TODO: faire tres attention a ce test. On ne split pas un node si il
        #  contient moins de 1 ou 2 points de train et de valid ou si le node est pur
        #  (impurity=0)
        is_leaf = (node_context.n_samples_train < min_samples_split) or (
            node_context.n_samples_valid < min_samples_split
        )

        # print("is_leaf: ", is_leaf)

        # TODO: mais apres si le n_samples_train_node ou le n_samples_valid_node de
        #  l'un des deux enfants est nul, on fera en fait pas le split

        # TODO: for now, simple stopping criterion. We'll add more options later
        # if first:
        #     # impurity = splitter.node_impurity()
        #     # dans le code d'origine y'a splitter.node_impurity qui appelle
        #     # self.criterion
        #
        #     # impurity = gini_node_impurity(splitter.criterion)
        #
        #     impurity = gini_node_impurity(
        #         tree.n_outputs,
        #         tree.n_classes,
        #         splitter.criterion.sum_total,
        #         splitter.criterion.weighted_n_node_samples,
        #     )
        #
        #     first = False
        #
        # # print("impurity: ", impurity)
        # # print("min_impurity_split: ", min_impurity_split)

        # If the node is pure it's a leaf: we won't split it
        # TODO: is_leaf = is_leaf or (impurity == 0)

        min_impurity_split = 0.0

        is_leaf = is_leaf or (impurity <= min_impurity_split)

        # print("is_leaf: ", is_leaf)

        # If the node is not a leaf

        if is_leaf:
            # print("Node is a leaf")
            split = None
            bin = 0
            feature = 0
            found_split = False
            # TODO: pourquoi on mettrai impurity = infini ici ?
            # impurity = -infinity
            # Faudrait que split soit defini dans tous les cas...
        else:
            # print("Node is not a leaf")
            split = find_node_split(tree_context, node_context)
            bin = split.bin
            feature = split.feature
            found_split = split.found_split

            # print("Found split with feature=", feature, "and bin=", bin)

            # # TODO: ici on calcule le vrai information gain
            # impurity = split.gain_proxy

            # split, n_total_constants = best_splitter_node_split(
            #     splitter, impurity, n_constant_features, X_idx_sort
            # )

            # TODO: for now, simple stopping criterion. We'll add more options later
            # # If EPSILON=0 in the below comparison, float precision
            # # issues stop splitting, producing trees that are
            # # dissimilar to v0.18
            # is_leaf = (
            #         is_leaf
            #         or split.pos >= end
            #         or (split.improvement + EPSILON < min_impurity_decrease)
            # )

        # If we did not find a split then the node is a leaf, since we can't split it
        is_leaf = is_leaf or not found_split

        # TODO: on met un threshold a la con ici
        threshold = 0.42
        # n_samples_node = 42
        # weighted_n_node_samples = 42.0
        weighted_n_samples_valid = 42.0

        node_id = add_node_tree(
            tree,
            parent,
            depth,
            # TODO: faudrait calculer correctement is_left
            is_left,
            is_leaf,
            feature,
            threshold,
            bin,
            # TODO: attention c'est pas le vrai information gain mais le proxy. A
            #  voir si c'est utile ou pas de le calculer
            impurity,
            node_context.n_samples_train,
            node_context.n_samples_valid,
            node_context.w_samples_train,
            # TODO: pour l'instant on ne le calcule pas, on verra si c'est utile
            weighted_n_samples_valid,
            start_train,
            end_train,
            start_valid,
            end_valid,
            node_context.loss_valid,
        )

        # return
        # print_tree(tree)
        # TODO: remettre le y_sum correct
        tree.y_pred[node_id, :] = node_context.y_pred

        # print("is_leaf: ", is_leaf)
        if not is_leaf:
            # If it's not a leaf then we need to add both childs in the node record,
            # so that they can be added in the tree and eventually be splitted later.

            # First, we need to update partition_train and partition_valid
            pos_train, pos_valid = split_indices(
                tree_context, split, start_train, end_train, start_valid, end_valid
            )

            # Then, we can add both childs records: add the left child
            push_record(
                records,
                node_id,  # The parent is the previous node_id
                depth + 1,  # depth is increased by one
                True,  # is_left=True
                split.impurity_left,  # Impurity of childs are saved in the split
                start_train,  # start_train from the node
                pos_train,  # end_train is at split position
                start_valid,
                pos_valid,
            )

            push_record(
                records,
                node_id,  # The parent is the previous node_id
                depth + 1,  # depth is increased by one
                False,  # is_left=False
                split.impurity_right,  # Impurity of childs are saved in the split
                pos_train,  # strain_train is at split position
                end_train,  # end_train is at split position
                pos_valid,
                end_valid,
            )

        # TODO: on nse servira de ca plus tard
        if depth > max_depth_seen:
            max_depth_seen = depth

    # We finished to grow the tree, now we can compute the tree aggregation weights

    # TREE_LEAF = nb_ssize_t(-1)

    aggregation = tree_context.aggregation

    step = float32(1.0)
    # Since the tree is grown in a depth-first fashion, we know that if we iterate
    # through the nodes in reverse order, we'll always iterate over childs before
    # iteration over parents.

    # tree.node_count
    # for node_idx in range():
    # for node in tree.nodes[::-1]:

    node_count = int(tree.node_count)
    # print("node_count: ", node_count)
    #
    # print(type(node_count))

    # TODO: mettre ca dans une fonction a part...
    if aggregation:
        for node_idx in range(node_count - 1, -1, -1):
            # print("node_idx: ", print(node_idx))
            node = tree.nodes[node_idx]
            # print("node_idx:", node_idx)
            # print("node_id:", node["node_id"])
            # print("is_leaf: ", node["is_leaf"])

            if node["is_leaf"]:
                # If the node is a leaf, the logarithm of its tree weight is simply step *
                # loss
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
