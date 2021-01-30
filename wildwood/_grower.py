"""

C'est un C/C depuis pygbm

This module contains the TreeGrower class.

TreeGrowee builds a regression tree fitting a Newton-Raphson step, based on
the gradients and hessians of the training data.
"""
from heapq import heappush, heappop
import numpy as np
from time import time

from ._splitting import (
    NodeContext,
    find_node_split,
    init_node_context,
    split_indices,
    # split_indices,
    # find_node_split,
    # find_node_split_subtraction,
)


from ._tree import (
    Records,
    # Tree,
    add_node_tree,
    tree_resize,
    push_node_record,
    pop_node_record,
    has_records,
    print_tree,
    print_records,
    get_records,
    get_nodes,
    TREE_UNDEFINED,
)
from ._utils import njit, infinity, nb_size_t


INITIAL_STACK_SIZE = nb_size_t(10)


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
    start_train = nb_size_t(0)

    end_train = nb_size_t(n_samples_train)
    start_valid = nb_size_t(0)
    end_valid = nb_size_t(n_samples_valid)
    depth = nb_size_t(0)
    # root as no parent
    parent = TREE_UNDEFINED
    is_left = False
    impurity = infinity
    n_constant_features = 0

    # print("start_train:", start_train)
    # print("end_valid:", end_valid)

    # Add the root node into the stack
    # print("Pushing root in the stack")
    push_node_record(
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
            n_constant_features,
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
        init_node_context(
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
            # TODO: pourquoi on mettrai impurity = infini ici ?
            # impurity = -infinity
            # Faudrait que split soit defini dans tous les cas...
        else:
            # print("Node is not a leaf")

            split = find_node_split(tree_context, node_context)
            bin = split.bin
            feature = split.feature
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

        # TODO: Once we've found the best split for this node, we add the node in the
        #  tree

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
            push_node_record(
                records,
                start_train,  # strain_train from the node
                pos_train,  # TODO
                start_valid,  # TODO
                pos_valid,  # TODO
                depth + 1,  # depth is increased by one
                node_id,  # The parent is the previous node_id
                True,  # is_left=True
                split.impurity_left,  # Impurity of childs was saved in the split
                n_constant_features,  # TODO: for now we don't care about that
            )

            # Add the right child
            push_node_record(
                records,
                pos_train,  # TODO
                end_train,  # TODO
                pos_valid,  # TODO
                end_valid,  # TODO
                depth + 1,  # depth is increased by one
                node_id,  # The parent is the previous node_id
                False,  # is_left=False
                split.impurity_right,  # Impurity of childs was saved in the split
                n_constant_features,  # TODO: for now we don't care about that
            )

        # TODO: on nse servira de ca plus tard
        if depth > max_depth_seen:
            max_depth_seen = depth

    # print_tree(tree)
    # print(get_nodes(tree))

    # print_records(records)
    # print(get_records(records))

    # TODO: put back this crap
    # if rc >= 0:
    #     # print("In depth_first_tree_builder_build calling tree_resize "
    #     #       "with tree.node_count: ", tree.node_count)
    #     rc = tree_resize(tree, tree.node_count)
    #     # rc = tree._resize_c(tree.node_count)
    #
    # if rc >= 0:
    #     tree.max_depth = max_depth_seen

    # TODO: ca ne sert a rien et c'est merdique
    # if rc == -1:
    #     raise MemoryError()

    # Ca c'est le code de pygbm :
    # while self.can_split_further():
    #     self.split_next()


# class TreeGrower:
#     """Tree grower class used to build a tree.
#
#     The tree is fitted to predict the values of a Newton-Raphson step. The
#     splits are considered in a best-first fashion, and the quality of a
#     split is defined in splitting._split_gain.
#
#     Parameters
#     ----------
#     X_binned : array-like of int, shape=(n_samples, n_features)
#         The binned input samples. Must be Fortran-aligned.
#     gradients : array-like, shape=(n_samples,)
#         The gradients of each training sample. Those are the gradients of the
#         loss w.r.t the predictions, evaluated at iteration ``i - 1``.
#     hessians : array-like, shape=(n_samples,)
#         The hessians of each training sample. Those are the hessians of the
#         loss w.r.t the predictions, evaluated at iteration ``i - 1``.
#     max_leaf_nodes : int or None, optional(default=None)
#         The maximum number of leaves for each tree. If None, there is no
#         maximum limit.
#     max_depth : int or None, optional(default=None)
#         The maximum depth of each tree. The depth of a tree is the number of
#         nodes to go from the root to the deepest leaf.
#     min_samples_leaf : int, optional(default=20)
#         The minimum number of samples per leaf.
#     min_gain_to_split : float, optional(default=0.)
#         The minimum gain needed to split a node. Splits with lower gain will
#         be ignored.
#     max_bins : int, optional(default=256)
#         The maximum number of bins. Used to define the shape of the
#         histograms.
#     n_bins_per_feature : array-like of int or int, optional(default=None)
#         The actual number of bins needed for each feature, which is lower or
#         equal to ``max_bins``. If it's an int, all features are considered to
#         have the same number of bins. If None, all features are considered to
#         have ``max_bins`` bins.
#     l2_regularization : float, optional(default=0)
#         The L2 regularization parameter.
#     min_hessian_to_split : float, optional(default=1e-3)
#         The minimum sum of hessians needed in each node. Splits that result in
#         at least one child having a sum of hessians less than
#         min_hessian_to_split are discarded.
#     shrinkage : float, optional(default=1)
#         The shrinkage parameter to apply to the leaves values, also known as
#         learning rate.
#     """
#
#     def __init__(
#         self,
#         X,
#         y,
#         train_indices,
#         valid_indices,
#         sample_weight_train,
#         sample_weight_valid,
#         n_bins=255
#         # self,
#         # X_binned,
#         # gradients,
#         # hessians,
#         # max_leaf_nodes=None,
#         # max_depth=None,
#         # min_samples_leaf=20,
#         # min_gain_to_split=0.0,
#         # max_bins=256,
#         # n_bins_per_feature=None,
#         # l2_regularization=0.0,
#         # min_hessian_to_split=1e-3,
#         # shrinkage=1.0,
#     ):
#
#         # self._validate_parameters(
#         #     X_binned,
#         #     max_leaf_nodes,
#         #     max_depth,
#         #     min_samples_leaf,
#         #     min_gain_to_split,
#         #     l2_regularization,
#         #     min_hessian_to_split,
#         # )
#         #
#         # if n_bins_per_feature is None:
#         #     n_bins_per_feature = max_bins
#         #
#         # if isinstance(n_bins_per_feature, int):
#         #     n_bins_per_feature = np.array(
#         #         [n_bins_per_feature] * X_binned.shape[1], dtype=np.uint32
#         #     )
#
#         self.splitting_context = SplittingContext(
#             X,
#             train_indices,
#             valid_indices,
#             # sample_weight_train,
#             # sample_weight_valid,
#             n_bins,
#             # TODO: pass n_bins_per_feature,
#             # None
#             # TODO: remettre options qui font sens
#             # gradients,
#             # hessians,
#             # l2_regularization,
#             # min_hessian_to_split,
#             # min_samples_leaf,
#             # min_gain_to_split,
#         )
#         # self.max_leaf_nodes = max_leaf_nodes
#         # self.max_depth = max_depth
#         # self.min_samples_leaf = min_xsamples_leaf
#         self.X_binned = X
#         # self.min_gain_to_split = min_gain_to_split
#         # self.shrinkage = shrinkage
#
#         # Remplacer ca par le stack des forets
#         self.splittable_nodes = []
#         self.finalized_leaves = []
#
#         # self.total_find_split_time = 0.0  # time spent finding the best splits
#         # self.total_apply_split_time = 0.0  # time spent splitting nodes
#         self._intilialize_root()
#         self.n_nodes = 1
#
#     # def _validate_parameters(
#     #     self,
#     #     X_binned,
#     #     max_leaf_nodes,
#     #     max_depth,
#     #     min_samples_leaf,
#     #     min_gain_to_split,
#     #     l2_regularization,
#     #     min_hessian_to_split,
#     # ):
#     #     """Validate parameters passed to __init__.
#     #
#     #     Also validate parameters passed to SplittingContext because we cannot
#     #     raise exceptions in a jitclass.
#     #     """
#     #     if X_binned.dtype != np.uint8:
#     #         raise NotImplementedError("Explicit feature binning required for now")
#     #     if not X_binned.flags.f_contiguous:
#     #         raise ValueError(
#     #             "X_binned should be passed as Fortran contiguous "
#     #             "array for maximum efficiency."
#     #         )
#     #     if max_leaf_nodes is not None and max_leaf_nodes < 1:
#     #         raise ValueError(
#     #             f"max_leaf_nodes={max_leaf_nodes} should not be" f" smaller than 1"
#     #         )
#     #     if max_depth is not None and max_depth < 1:
#     #         raise ValueError(f"max_depth={max_depth} should not be" f" smaller than 1")
#     #     if min_samples_leaf < 1:
#     #         raise ValueError(
#     #             f"min_samples_leaf={min_samples_leaf} should " f"not be smaller than 1"
#     #         )
#     #     if min_gain_to_split < 0:
#     #         raise ValueError(
#     #             f"min_gain_to_split={min_gain_to_split} " f"must be positive."
#     #         )
#     #     if l2_regularization < 0:
#     #         raise ValueError(
#     #             f"l2_regularization={l2_regularization} must be " f"positive."
#     #         )
#     #     if min_hessian_to_split < 0:
#     #         raise ValueError(
#     #             f"min_hessian_to_split={min_hessian_to_split} " f"must be positive."
#     #         )
#
#     def grow(self):
#         """Grow the tree, from root to leaves."""
#         # On fait pousser l'arbre de la facon suivante :
#         # 1. On commence par ajouter la racine dans le stack
#         # 2. Tant que le stack LIFO n'est pas vide
#         #    2.1 On pop un element du stack
#         #    2.2 On decide si on veut splitter calcule son histogramme, on decide si on
#         #    peut
#         #    splitter (
#         #    min_sample_leaf) on cherche son meilleur split. Si pas de split on en
#         #    fait une feuille finalisee. Sinon on ajoute ses deux enfants
#         #    a la stack
#         #
#
#         # TODO: j'en suis ICI ICI ICI (2021 / 01 / 11)
#         # TODO: Ici il faut reprendre le code de depth_first_tree_builder_build avec le
#         # stack et tout
#
#         # This is the output split
#         # split = SplitRecord()
#
#         if tree.max_depth <= 10:
#             init_capacity = (2 ** (tree.max_depth + 1)) - 1
#         else:
#             init_capacity = 2047
#
#         tree_resize(tree, init_capacity)
#
#         # Parameters
#         # splitter = builder.splitter
#
#         # max_depth = builder.max_depth
#         # min_samples_leaf = builder.min_samples_leaf
#         # min_weight_leaf = builder.min_weight_leaf
#         # min_samples_split = builder.min_samples_split
#         # min_impurity_decrease = builder.min_impurity_decrease
#         # min_impurity_split = builder.min_impurity_split
#
#         # Recursive partition (without actual recursion)
#         # splitter.init(X, y, sample_weight_ptr)
#
#         # best_splitter_init(splitter, X, y, sample_weight)
#         # splitter_init(splitter, X, y, sample_weight)
#
#         # cdef SIZE_t start
#         # cdef SIZE_t end
#         # cdef SIZE_t depth
#         # cdef SIZE_t parent
#         # cdef bint is_left
#         # cdef SIZE_t n_node_samples = splitter.n_samples
#         n_node_samples = splitter.n_samples
#         # cdef double weighted_n_samples = splitter.weighted_n_samples
#         weighted_n_samples = splitter.weighted_n_samples
#         # cdef double weighted_n_node_samples
#         # cdef SplitRecord split
#         # cdef SIZE_t node_id
#
#         # cdef double impurity = INFINITY
#         impurity = INFINITY
#
#         # cdef SIZE_t n_constant_features
#         # cdef bint is_leaf
#         # cdef bint first = 1
#         first = True
#         # cdef SIZE_t max_depth_seen = -1
#         max_depth_seen = -1
#         # cdef int rc = 0
#         rc = 0
#
#         stack = Stack(INITIAL_STACK_SIZE)
#         # cdef StackRecord stack_record
#
#         # with nogil:
#         # push root node onto stack
#         # rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
#
#         rc = stack_push(stack, 0, n_node_samples, 0, TREE_UNDEFINED, 0, INFINITY, 0)
#
#         # if rc == -1:
#         #     # got return code -1 - out-of-memory
#         #     with gil:
#         #         raise MemoryError()
#
#         while not stack_is_empty(stack):
#             stack_record = stack_pop(stack)
#             # start = stack_record.start
#             # end = stack_record.end
#             # depth = stack_record.depth
#             # parent = stack_record.parent
#             # is_left = stack_record.is_left
#             # impurity = stack_record.impurity
#             # n_constant_features = stack_record.n_constant_features
#             start = stack_record["start"]
#             end = stack_record["end"]
#             depth = stack_record["depth"]
#             parent = stack_record["parent"]
#             is_left = stack_record["is_left"]
#             impurity = stack_record["impurity"]
#             n_constant_features = stack_record["n_constant_features"]
#
#             n_node_samples = end - start
#             # splitter.node_reset(start, end, &weighted_n_node_samples)
#             weighted_n_node_samples = splitter_node_reset(splitter, start, end)
#
#             is_leaf = (
#                 depth >= max_depth
#                 or n_node_samples < min_samples_split
#                 or n_node_samples < 2 * min_samples_leaf
#                 or weighted_n_node_samples < 2 * min_weight_leaf
#             )
#
#             if first:
#                 # TODO: some other way, only for gini here
#                 # impurity = splitter.node_impurity()
#                 # dans le code d'origine y'a splitter.node_impurity qui appelle
#                 # self.criterion
#
#                 # impurity = gini_node_impurity(splitter.criterion)
#
#                 impurity = gini_node_impurity(
#                     tree.n_outputs,
#                     tree.n_classes,
#                     splitter.criterion.sum_total,
#                     splitter.criterion.weighted_n_node_samples,
#                 )
#
#                 first = False
#
#             # print("impurity: ", impurity)
#             # print("min_impurity_split: ", min_impurity_split)
#             is_leaf = is_leaf or (impurity <= min_impurity_split)
#
#             # print("is_leaf: ", is_leaf)
#             if not is_leaf:
#                 # splitter.node_split(impurity, &split, &n_constant_features)
#
#                 # TODO: Hmmm y'a un truc qui ne va pas là ! je suis censé renvoyer best,
#                 #  n_constant_features et les deux derniers arguments ne sont pas utlisés
#                 # best_splitter_node_split(splitter, impurity, split, n_constant_features)
#                 split, n_total_constants = best_splitter_node_split(
#                     splitter, impurity, n_constant_features, X_idx_sort
#                 )
#
#                 # If EPSILON=0 in the below comparison, float precision
#                 # issues stop splitting, producing trees that are
#                 # dissimilar to v0.18
#                 is_leaf = (
#                     is_leaf
#                     or split.pos >= end
#                     or (split.improvement + EPSILON < min_impurity_decrease)
#                 )
#
#             node_id = tree_add_node(
#                 tree,
#                 parent,
#                 is_left,
#                 is_leaf,
#                 split.feature,
#                 split.threshold,
#                 impurity,
#                 n_node_samples,
#                 weighted_n_node_samples,
#             )
#
#             # node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
#             #                          split.threshold, impurity, n_node_samples,
#             #                          weighted_n_node_samples)
#
#             if node_id == SIZE_MAX:
#                 rc = -1
#                 break
#
#             # Store value for all nodes, to facilitate tree/model
#             # inspection and interpretation
#             # TODO oui c'est important ca permet de calculer les predictions...
#
#             # splitter_node_value(splitter, tree.values, node_id)
#
#             tree.values[node_id, :, :] = splitter.criterion.sum_total
#             # splitter.criterion.sum_total
#             # # #
#             # #     sum_total = criterion.sum_total
#             #     dest[node_id, :, :] = sum_total
#
#             # splitter_node_value(splitter, tree.value + node_id * tree.value_stride)
#
#             # splitter.node_value(tree.value + node_id * tree.value_stride)
#
#             if not is_leaf:
#                 # Push right child on stack
#                 # rc = stack.push(split.pos, end, depth + 1, node_id, 0,
#                 #                 split.impurity_right, n_constant_features)
#
#                 #     stack, start, end, depth, parent, is_left, impurity, n_constant_features
#                 rc = stack_push(
#                     stack,
#                     split.pos,
#                     end,
#                     depth + 1,
#                     node_id,
#                     0,
#                     split.impurity_right,
#                     n_constant_features,
#                 )
#
#                 if rc == -1:
#                     break
#
#                 # Push left child on stack
#                 rc = stack_push(
#                     stack,
#                     start,
#                     split.pos,
#                     depth + 1,
#                     node_id,
#                     1,
#                     split.impurity_left,
#                     n_constant_features,
#                 )
#                 if rc == -1:
#                     break
#
#             if depth > max_depth_seen:
#                 max_depth_seen = depth
#
#         if rc >= 0:
#             # print("In depth_first_tree_builder_build calling tree_resize "
#             #       "with tree.node_count: ", tree.node_count)
#             rc = tree_resize(tree, tree.node_count)
#             # rc = tree._resize_c(tree.node_count)
#
#         if rc >= 0:
#             tree.max_depth = max_depth_seen
#
#         # TODO: ca ne sert a rien et c'est merdique
#         if rc == -1:
#             raise MemoryError()
#
#         # Ca c'est le code de pygbm :
#         # while self.can_split_further():
#         #     self.split_next()
#
#     def _intilialize_root(self):
#         n_samples = self.X_binned.shape[0]
#         depth = 0
#         # if self.splitting_context.constant_hessian:
#         #     hessian = self.splitting_context.hessians[0] * n_samples
#         # else:
#         #     hessian = self.splitting_context.hessians.sum()
#
#         self.root = TreeNode(
#             depth=depth,
#             sample_indices=self.splitting_context.partition.view(),
#             sum_gradients=self.splitting_context.gradients.sum(),
#             sum_hessians=hessian,
#         )
#
#         # TODO: remettre ici
#         # if self.max_leaf_nodes is not None and self.max_leaf_nodes == 1:
#         #     self._finalize_leaf(self.root)
#         #     return
#         # if self.root.n_samples < 2 * self.min_samples_leaf:
#         #     # Do not even bother computing any splitting statistics.
#         #     self._finalize_leaf(self.root)
#         #     return
#
#         self._compute_spittability(self.root)
#
#     def _compute_spittability(self, node, only_hist=False):
#         """Compute histograms and best possible split of a node.
#
#         If the best possible gain is 0 of if the constraints aren't met
#         (min_samples_leaf, min_hessian_to_split, min_gain_to_split) then the
#         node is finalized (transformed into a leaf), else it is pushed on
#         the splittable node heap.
#
#         Parameters
#         ----------
#         node : TreeNode
#             The node to evaluate.
#         only_hist : bool, optional (default=False)
#             Whether to only compute the histograms and the SplitInfo. It is
#             set to ``True`` when ``_compute_spittability`` was called by a
#             sibling node: we only want to compute the histograms (which also
#             computes the ``SplitInfo``), not finalize or push the node. If
#             ``_compute_spittability`` is called again by the grower on this
#             same node, the histograms won't be computed again.
#         """
#         # Compute split_info and histograms if not already done
#         if node.split_info is None and node.histograms is None:
#             # If the sibling has less samples, compute its hist first (with
#             # the regular method) and use the subtraction method for the
#             # current node
#             if node.sibling is not None:  # root has no sibling
#                 if node.sibling.n_samples < node.n_samples:
#                     self._compute_spittability(node.sibling, only_hist=True)
#                     # As hist of sibling is now computed we'll use the hist
#                     # subtraction method for the current node.
#                     node.hist_subtraction = True
#
#             tic = time()
#             if node.hist_subtraction:
#                 split_info, histograms = find_node_split_subtraction(
#                     self.splitting_context,
#                     node.sample_indices,
#                     node.parent.histograms,
#                     node.sibling.histograms,
#                 )
#             else:
#                 split_info, histograms = find_node_split(
#                     self.splitting_context, node.sample_indices
#                 )
#             toc = time()
#             node.find_split_time = toc - tic
#             self.total_find_split_time += node.find_split_time
#             node.construction_speed = node.n_samples / node.find_split_time
#             node.split_info = split_info
#             node.histograms = histograms
#
#         if only_hist:
#             # _compute_spittability was called by a sibling. We only needed to
#             # compute the histogram.
#             return
#
#         if node.split_info.gain <= 0:  # no valid split
#             # Note: this condition is reached if either all the leaves are
#             # pure (best gain = 0), or if no split would satisfy the
#             # constraints, (min_hessians_to_split, min_gain_to_split,
#             # min_samples_leaf)
#             self._finalize_leaf(node)
#
#         else:
#             heappush(self.splittable_nodes, node)
#
#     def split_next(self):
#         """Split the node with highest potential gain.
#
#         Returns
#         -------
#         left : TreeNode
#             The resulting left child.
#         right : TreeNode
#             The resulting right child.
#         """
#         if len(self.splittable_nodes) == 0:
#             raise StopIteration("No more splittable nodes")
#
#         # Consider the node with the highest loss reduction (a.k.a. gain)
#         node = heappop(self.splittable_nodes)
#
#         tic = time()
#         (sample_indices_left, sample_indices_right) = split_indices(
#             self.splitting_context, node.split_info, node.sample_indices
#         )
#         toc = time()
#         node.apply_split_time = toc - tic
#         self.total_apply_split_time += node.apply_split_time
#
#         depth = node.depth + 1
#         n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
#         n_leaf_nodes += 2
#
#         left_child_node = TreeNode(
#             depth,
#             sample_indices_left,
#             node.split_info.gradient_left,
#             node.split_info.hessian_left,
#             parent=node,
#         )
#         right_child_node = TreeNode(
#             depth,
#             sample_indices_right,
#             node.split_info.gradient_right,
#             node.split_info.hessian_right,
#             parent=node,
#         )
#         left_child_node.sibling = right_child_node
#         right_child_node.sibling = left_child_node
#         node.right_child = right_child_node
#         node.left_child = left_child_node
#         self.n_nodes += 2
#
#         if self.max_depth is not None and depth == self.max_depth:
#             self._finalize_leaf(left_child_node)
#             self._finalize_leaf(right_child_node)
#             return left_child_node, right_child_node
#
#         if self.max_leaf_nodes is not None and n_leaf_nodes == self.max_leaf_nodes:
#             self._finalize_leaf(left_child_node)
#             self._finalize_leaf(right_child_node)
#             self._finalize_splittable_nodes()
#             return left_child_node, right_child_node
#
#         if left_child_node.n_samples < self.min_samples_leaf * 2:
#             self._finalize_leaf(left_child_node)
#         else:
#             self._compute_spittability(left_child_node)
#
#         if right_child_node.n_samples < self.min_samples_leaf * 2:
#             self._finalize_leaf(right_child_node)
#         else:
#             self._compute_spittability(right_child_node)
#
#         return left_child_node, right_child_node
#
#     def can_split_further(self):
#         """Return True if there are still nodes to split."""
#         return len(self.splittable_nodes) >= 1
#
#     def _finalize_leaf(self, node):
#         """Compute the prediction value that minimizes the objective function.
#
#         This sets the node.value attribute (node is a leaf iff node.value is
#         not None).
#
#         See Equation 5 of:
#         XGBoost: A Scalable Tree Boosting System, T. Chen, C. Guestrin, 2016
#         https://arxiv.org/abs/1603.02754
#         """
#         node.value = (
#             -self.shrinkage
#             * node.sum_gradients
#             / (node.sum_hessians + self.splitting_context.l2_regularization)
#         )
#         self.finalized_leaves.append(node)
#
#     def _finalize_splittable_nodes(self):
#         """Transform all splittable nodes into leaves.
#
#         Used when some constraint is met e.g. maximum number of leaves or
#         maximum depth."""
#         while len(self.splittable_nodes) > 0:
#             node = self.splittable_nodes.pop()
#             self._finalize_leaf(node)
#
#     def make_predictor(self, numerical_thresholds=None):
#         """Make a TreePredictor object out of the current tree.
#
#         Parameters
#         ----------
#         numerical_thresholds : array-like of floats, optional (default=None)
#             The actual thresholds values of each bin, expected to be in sorted
#             increasing order. None if the training data was pre-binned.
#
#         Returns
#         -------
#         A TreePredictor object.
#         """
#         predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
#         self._fill_predictor_node_array(
#             predictor_nodes, self.root, numerical_thresholds=numerical_thresholds
#         )
#         has_numerical_thresholds = numerical_thresholds is not None
#         return TreePredictor(
#             nodes=predictor_nodes, has_numerical_thresholds=has_numerical_thresholds
#         )
#
#     def _fill_predictor_node_array(
#         self, predictor_nodes, grower_node, numerical_thresholds=None, next_free_idx=0
#     ):
#         """Helper used in make_predictor to set the TreePredictor fields."""
#         node = predictor_nodes[next_free_idx]
#         node["count"] = grower_node.n_samples
#         node["depth"] = grower_node.depth
#         if grower_node.split_info is not None:
#             node["gain"] = grower_node.split_info.gain
#         else:
#             node["gain"] = -1
#
#         if grower_node.value is not None:
#             # Leaf node
#             node["is_leaf"] = True
#             node["value"] = grower_node.value
#             return next_free_idx + 1
#         else:
#             # Decision node
#             split_info = grower_node.split_info
#             feature_idx, bin_idx = split_info.feature_idx, split_info.bin_idx
#             node["feature_idx"] = feature_idx
#             node["bin_threshold"] = bin_idx
#             if numerical_thresholds is not None:
#                 node["threshold"] = numerical_thresholds[feature_idx][bin_idx]
#             next_free_idx += 1
#
#             node["left"] = next_free_idx
#             next_free_idx = self._fill_predictor_node_array(
#                 predictor_nodes,
#                 grower_node.left_child,
#                 numerical_thresholds=numerical_thresholds,
#                 next_free_idx=next_free_idx,
#             )
#
#             node["right"] = next_free_idx
#             return self._fill_predictor_node_array(
#                 predictor_nodes,
#                 grower_node.right_child,
#                 numerical_thresholds=numerical_thresholds,
#                 next_free_idx=next_free_idx,
#             )
