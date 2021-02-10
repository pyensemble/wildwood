"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

import numbers
import warnings
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, check_array
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args

from numba import _helperlib


from ._grow import grow

# from ._criterion import Criterion
# from ._splitter import BestSplitter
# from ._tree_old import DepthFirstTreeBuilder

# from . import _tree_old

from ._utils import np_float32, np_uint8, np_size_t, np_ssize_t

from ._node import NodeContext
from ._splitting import TreeContext

from ._tree import Tree, get_nodes, tree_predict


# from ._tree import BestFirstTreeBuilder
# from ._tree_old import Tree

# from ._tree import _build_pruned_tree_ccp
# from ._tree import ccp_pruning_path
# from . import _tree_old, _splitter, _criterion, _utils


# __all__ = ["DecisionTreeClassifier"]

# TODO: dans wildwood le TreeBinaryClassifier est prive et ne devrait pas etre utilise
#  en dehors d'une foret, mais bon pour l'instant on fait ca viiiite


# =============================================================================
# Types and constants
# =============================================================================


# DTYPE = ._utils.np_dtype_t
# DOUBLE = _utils.np_double_t

# CRITERIA_CLF = {
#     "gini": _criterion.Gini,
#     # "entropy": _criterion.Entropy
# }

# CRITERIA_REG = {"mse": _criterion.MSE,
#                 "friedman_mse": _criterion.FriedmanMSE,
#                 "mae": _criterion.MAE,
#                 "poisson": _criterion.Poisson}

# DENSE_SPLITTERS = {
#     "best": _splitter.BestSplitter,
#     # "random": _splitter.RandomSplitter
# }

# SPARSE_SPLITTERS = {"best": _splitter.BestSparseSplitter,
#                     "random": _splitter.RandomSparseSplitter}

# =============================================================================
# Base decision tree
# =============================================================================


class TreeBase(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        criterion,
        loss,
        step,
        aggregation,
        dirichlet,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        categorical_features=None,
        max_features,
        random_state=None,
        verbose=0,
    ):
        self.criterion = criterion
        self.loss = loss
        self.step = step
        self.aggregation = aggregation
        self.dirichlet = dirichlet
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_features = categorical_features
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

    def get_depth(self):
        """Return the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves

    def fit(
        self, X, y, train_indices, valid_indices, sample_weights, check_input=False,
    ):

        # print("Training {clf}".format(clf=self.__class__.__name__))
        # print("With X: ", X)

        # TODO: reprendre cette methode, mettre des property pour les attributs de
        #  classe
        # random_state = check_random_state(self.random_state)
        # random_state = 42

        # TODO: everything should be checked already here, since it's only used from
        #  a forest
        # if check_input:
        #     # Need to validate separately here.
        #     # We can't pass multi_ouput=True because that would allow y to be
        #     # csr.
        #     check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
        #     check_y_params = dict(ensure_2d=False, dtype=None)
        #     X, y = self._validate_data(
        #         X, y, validate_separately=(check_X_params, check_y_params)
        #     )
        #     if issparse(X):
        #         X.sort_indices()
        #
        #         if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
        #             raise ValueError(
        #                 "No support for np.int64 index based " "sparse matrices"
        #             )
        #
        #     if self.criterion == "poisson":
        #         if np.any(y < 0):
        #             raise ValueError(
        #                 "Some value(s) of y are negative which is"
        #                 " not allowed for Poisson regression."
        #             )
        #         if np.sum(y) <= 0:
        #             raise ValueError(
        #                 "Sum of y is not positive which is "
        #                 "necessary for Poisson regression."
        #             )

        # Determine output settings

        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        # TODO: on a enleve ca pour avoir un y qui est 1D
        # if y.ndim == 1:
        #     # reshape is necessary to preserve the data contiguity against vs
        #     # [:, np.newaxis] that does not.
        #     y = np.reshape(y, (-1, 1))

        # self.n_outputs_ = y.shape[1]
        self.n_outputs_ = 1

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            # if self.class_weight is not None:
            y_original = np.copy(y)

            # TODO: Faudra aussi remettre ca
            # y_encoded = np.zeros(y.shape, dtype=int)
            #
            # for k in range(self.n_outputs_):
            #     classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            #     self.classes_.append(classes_k)
            #     self.n_classes_.append(classes_k.shape[0])
            # y = y_encoded

            # if self.class_weight is not None:
            #     expanded_class_weight = compute_sample_weight(
            #         self.class_weight, y_original
            #     )

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != np_float32 or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np_float32)

        # Check parameters
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth
        # max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0.0 < self.min_samples_leaf <= 0.5:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the integer %s" % self.min_samples_split
                )
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0.0 < self.min_samples_split <= 1.0:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the float %s" % self.min_samples_split
                )
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match "
                "number of samples=%d" % (len(y), n_samples)
            )
        # if not 0 <= self.min_weight_fraction_leaf <= 0.5:
        #     raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        # if not isinstance(max_leaf_nodes, numbers.Integral):
        #     raise ValueError(
        #         "max_leaf_nodes must be integral number but was " "%r" % max_leaf_nodes
        #     )
        # if -1 < max_leaf_nodes < 2:
        #     raise ValueError(
        #         ("max_leaf_nodes {0} must be either None " "or larger than 1").format(
        #             max_leaf_nodes
        #         )
        #     )

        # if sample_weight is not None:
        #     sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        # if expanded_class_weight is not None:
        #     if sample_weight is not None:
        #         sample_weight = sample_weight * expanded_class_weight
        #     else:
        #         sample_weight = expanded_class_weight

        # # Set min_weight_leaf from min_weight_fraction_leaf
        # if sample_weight is None:
        #     min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        # else:
        #     min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # min_impurity_split = self.min_impurity_split
        # if min_impurity_split is not None:
        #     warnings.warn(
        #         "The min_impurity_split parameter is deprecated. "
        #         "Its default value has changed from 1e-7 to 0 in "
        #         "version 0.23, and it will be removed in 0.25. "
        #         "Use the min_impurity_decrease parameter instead.",
        #         FutureWarning,
        #     )
        #
        #     if min_impurity_split < 0.0:
        #         raise ValueError(
        #             "min_impurity_split must be greater than " "or equal to 0"
        #         )
        # else:
        #     min_impurity_split = 0

        # if self.min_impurity_decrease < 0.0:
        #     raise ValueError(
        #         "min_impurity_decrease must be greater than " "or equal to 0"
        #     )

        # # TODO: Remove in v0.26
        # if X_idx_sorted != "deprecated":
        #     warnings.warn(
        #         "The parameter 'X_idx_sorted' is deprecated and has "
        #         "no effect. It will be removed in v0.26. You can "
        #         "suppress this warning by not passing any value to "
        #         "the 'X_idx_sorted' parameter.",
        #         FutureWarning,
        #     )

        # Build tree

        # TODO: ICI ICI ICI faut y aller

        # print("In tree fit we have the following stuff")
        # print(X, X.dtype)
        # print(y)
        # print(train_indices)
        # print(valid_indices)
        # print(sample_weights)

        # Faut coder un grower d'arbre...

        # for k in range(self.n_trees_per_iteration_):

        # print("n_classes_[0]: ", self.n_classes_[0])

        # TODO: Faudra changer le self.n_classes_ a terme
        # n_classes = self.n_classes_[0]

        n_classes = 2
        max_bins = 255
        # TODO: on obtiendra cette info via le binner qui est dans la foret
        n_samples, n_features = X.shape
        n_bins_per_feature = max_bins * np.ones(n_features)
        n_bins_per_feature = n_bins_per_feature.astype(np_ssize_t)

        # print(y.flags)
        # print("y.dtype: ", y.dtype)
        # print("train_indices.flags: ", train_indices.flags)
        # print("train_indices.dtypes: ", train_indices.dtype)

        # Then, we create the tree object, which is mostly a data container for the
        # nodes.
        tree = Tree(n_features, n_classes)

        # TODO: faudra verifier ca aussi
        max_features = 2

        dirichlet = self.dirichlet
        aggregation = self.aggregation
        step = self.step

        # We build a tree context, that contains global information about
        # the data, in particular the way we'll organize data into contiguous
        # node indexes both for training and validation samples
        tree_context = TreeContext(
            X,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            n_classes,
            max_bins,
            n_bins_per_feature,
            max_features,
            aggregation,
            dirichlet,
            step,
        )

        # print("Creating context...")
        node_context = NodeContext(tree_context)
        # print("Done creating context")

        # On ne peut pas passer self a grow car self est une classe python...
        # print("grow(tree, tree_context, node_context)")
        grow(tree, tree_context, node_context)

        self._tree = tree
        self._tree_context = tree_context

        # X,
        # y,
        # train_indices,
        # valid_indices,
        # sample_weight_train,
        # sample_weight_valid,
        # check_input = False,

        # criterion = self.criterion

        # TODO: criterion est de toute facon un string
        # if not isinstance(criterion, Criterion):
        #     if is_classification:
        #         criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
        #                                                  self.n_classes_)
        #     else:
        #         criterion = CRITERIA_REG[self.criterion](self.n_outputs_,
        #                                                  n_samples)

        # if is_classification:
        #     criterion = CRITERIA_CLF[self.criterion](self.n_outputs_, self.n_classes_)
        # else:
        #     raise NotImplementedError()
        #     # criterion = CRITERIA_REG[self.criterion](self.n_outputs_,
        #     #                                          n_samples)
        #
        # # SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS
        # SPLITTERS = DENSE_SPLITTERS
        # splitter = self.splitter
        # TODO: pareil ici de toute facon splitter est un string
        # if not isinstance(self.splitter, Splitter):
        #     splitter = SPLITTERS[self.splitter](criterion,
        #                                         self.max_features_,
        #                                         min_samples_leaf,
        #                                         min_weight_leaf,
        #                                         random_state)
        # splitter = _splitter.BestSplitter(criterion,
        #     self.max_features_,
        #     min_samples_leaf,
        #     min_weight_leaf,
        #     self.random_state)
        #
        # if is_classifier(self):
        #     self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)
        # else:
        #     raise NotImplementedError()
        #     self.tree_ = Tree(self.n_features_,
        #                       # TODO: tree should't need this in this case
        #                       np.array([1] * self.n_outputs_, dtype=np.intp),
        #                       self.n_outputs_)
        #
        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        # if max_leaf_nodes < 0:
        # TODO: remettre le builder
        # builder = DepthFirstTreeBuilder(
        #     splitter,
        #     min_samples_split,
        #     min_samples_leaf,
        #     min_weight_leaf,
        #     max_depth,
        #     self.min_impurity_decrease,
        #     min_impurity_split,
        # )
        # else:
        #     # builder = BestFirstTreeBuilder(splitter, min_samples_split,
        #     #                                min_samples_leaf,
        #     #                                min_weight_leaf,
        #     #                                max_depth,
        #     #                                max_leaf_nodes,
        #     #                                self.min_impurity_decrease,
        #     #                                min_impurity_split)
        #     raise NotImplementedError()
        # builder.build(self.tree_, X, y, sample_weight)
        # TODO: Let's pre-sort the features
        # print("Computing X_idx_sort...")
        # X_idx_sort = np.argsort(X, axis=0).astype(np.intp)
        # print("Done...")
        # TODO: set the random_state
        # self._set_random_state()
        # _tree.depth_first_tree_builder_build(
        #     builder, self.tree_, X, y, sample_weight, X_idx_sort
        # )
        # if self.n_outputs_ == 1 and is_classifier(self):
        #     self.n_classes_ = self.n_classes_[0]
        #     self.classes_ = self.classes_[0]
        #
        # self._put_back_random_state()
        # self._prune_tree()
        # TODO: on ne prune pas, quelle drole d'idee !!!
        return self

    def get_nodes(self):
        return get_nodes(self._tree)

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        if check_input:

            X = check_array(X, accept_sparse="csr", dtype=np_float32)
            # X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr",
            #                         reset=False)
            if issparse(X) and (
                X.indices.dtype != np.intc or X.indptr.dtype != np.intc
            ):
                raise ValueError(
                    "No support for np.int64 index based " "sparse matrices"
                )
        else:
            # The number of features is checked regardless of `check_input`
            self._check_n_features(X, reset=False)
        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)

        # proba = self.tree_.predict(X)

        proba = _tree_old.tree_predict(self.tree_, X)
        n_samples = X.shape[0]

        # Classification
        if is_classifier(self):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)

            else:
                class_type = self.classes_[0].dtype
                predictions = np.zeros((n_samples, self.n_outputs_), dtype=class_type)
                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1), axis=0
                    )

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]

            else:
                return proba[:, :, 0]

    def apply(self, X, check_input=True):
        """Return the index of the leaf that each sample is predicted as.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    def _set_random_state(self):
        # This uses a trick by Alexandre Gramfort,
        #   see https://github.com/numba/numba/issues/3249
        if self._random_state >= 0:
            ## if self._using_numba:
            r = np.random.RandomState(self._random_state)
            ptr = _helperlib.rnd_get_np_state_ptr()
            ints, index = r.get_state()[1:3]
            _helperlib.rnd_set_state(ptr, (index, [int(x) for x in ints]))
            self._ptr = ptr
            self._r = r
            # else:
            #     np.random.seed(self._random_state)

    def _put_back_random_state(self):
        # This uses a trick by Alexandre Gramfort,
        #   see https://github.com/numba/numba/issues/3249
        if self._random_state >= 0:
            # if self._using_numba:
            ptr = self._ptr
            r = self._r
            index, ints = _helperlib.rnd_get_state(ptr)
            r.set_state(("MT19937", ints, index, 0, 0.0))

    @property
    def random_state(self):
        """:obj:`int` or :obj:`None`: Controls the randomness involved in the trees."""
        if self._random_state == -1:
            return 0
        else:
            return self._random_state

    @random_state.setter
    def random_state(self, val):
        # if self.no_python:
        #     raise ValueError(
        #         "You cannot modify `random_state` after calling `partial_fit`"
        #     )
        # else:
        if val is None:
            self._random_state = -1
        elif not isinstance(val, int):
            raise ValueError("`random_state` must be of type `int`")
        elif val < 0:
            raise ValueError("`random_state` must be >= 0")
        else:
            self._random_state = val

    # @property
    # def feature_importances_(self):
    #     """Return the feature importances.
    #
    #     The importance of a feature is computed as the (normalized) total
    #     reduction of the criterion brought by that feature.
    #     It is also known as the Gini importance.
    #
    #     Warning: impurity-based feature importances can be misleading for
    #     high cardinality features (many unique values). See
    #     :func:`sklearn.inspection.permutation_importance` as an alternative.
    #
    #     Returns
    #     -------
    #     feature_importances_ : ndarray of shape (n_features,)
    #         Normalized total reduction of criteria by feature
    #         (Gini importance).
    #     """
    #     check_is_fitted(self)
    #
    #     return self.tree_.compute_feature_importances()

    @property
    def n_nodes_(self):
        # TODO: what if the tree is not trained yet ?
        return self._tree.node_count


class TreeBinaryClassifier(ClassifierMixin, TreeBase):
    def __init__(
        self,
        *,
        criterion="gini",
        loss="log",
        step=1.0,
        aggregation=True,
        dirichlet=0.5,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        categorical_features=None,
        max_features="auto",
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            criterion=criterion,
            loss=loss,
            step=step,
            aggregation=aggregation,
            dirichlet=dirichlet,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            categorical_features=categorical_features,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(
        self, X, y, train_indices, valid_indices, sample_weights, check_input=False,
    ):

        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """

        TreeBase.fit(
            self, X, y, train_indices, valid_indices, sample_weights, check_input,
        )
        return self

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """

        # TODO: on suppose que l'arbre est fitte et que les donnes sont binnees
        # check_is_fitted(self)
        # X = self._validate_X_predict(X, check_input)

        # proba = self.tree_.predict(X)

        # TODO: pas encore des proba mais juste des sums
        proba = tree_predict(
            self._tree, X, self._tree_context.aggregation, self._tree_context.step
        )

        # proba = _tree_old.tree_predict(self.tree_, X)

        # print(proba.shape)
        # normalizer = proba.sum(axis=1)[:, np.newaxis]
        # print(normalizer)
        # proba /= normalizer

        # if self.n_outputs_ == 1:
        #     proba = proba[:, : self.n_classes_]
        #     normalizer = proba.sum(axis=1)[:, np.newaxis]
        #     normalizer[normalizer == 0.0] = 1.0
        #     proba /= normalizer

        return proba

        # else:
        #     all_proba = []
        #
        #     for k in range(self.n_outputs_):
        #         proba_k = proba[:, k, : self.n_classes_[k]]
        #         normalizer = proba_k.sum(axis=1)[:, np.newaxis]
        #         normalizer[normalizer == 0.0] = 1.0
        #         proba_k /= normalizer
        #         all_proba.append(proba_k)
        #
        #     return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


# class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
#     """A decision tree regressor.
#
#     Read more in the :ref:`User Guide <tree>`.
#
#     Parameters
#     ----------
#     criterion : {"mse", "friedman_mse", "mae", "poisson"}, default="mse"
#         The function to measure the quality of a split. Supported criteria
#         are "mse" for the mean squared error, which is equal to variance
#         reduction as feature selection criterion and minimizes the L2 loss
#         using the mean of each terminal node, "friedman_mse", which uses mean
#         squared error with Friedman's improvement score for potential splits,
#         "mae" for the mean absolute error, which minimizes the L1 loss using
#         the median of each terminal node, and "poisson" which uses reduction in
#         Poisson deviance to find splits.
#
#         .. versionadded:: 0.18
#            Mean Absolute Error (MAE) criterion.
#
#         .. versionadded:: 0.24
#             Poisson deviance criterion.
#
#     splitter : {"best", "random"}, default="best"
#         The strategy used to choose the split at each node. Supported
#         strategies are "best" to choose the best split and "random" to choose
#         the best random split.
#
#     max_depth : int, default=None
#         The maximum depth of the tree. If None, then nodes are expanded until
#         all leaves are pure or until all leaves contain less than
#         min_samples_split samples.
#
#     min_samples_split : int or float, default=2
#         The minimum number of samples required to split an internal node:
#
#         - If int, then consider `min_samples_split` as the minimum number.
#         - If float, then `min_samples_split` is a fraction and
#           `ceil(min_samples_split * n_samples)` are the minimum
#           number of samples for each split.
#
#         .. versionchanged:: 0.18
#            Added float values for fractions.
#
#     min_samples_leaf : int or float, default=1
#         The minimum number of samples required to be at a leaf node.
#         A split point at any depth will only be considered if it leaves at
#         least ``min_samples_leaf`` training samples in each of the left and
#         right branches.  This may have the effect of smoothing the model,
#         especially in regression.
#
#         - If int, then consider `min_samples_leaf` as the minimum number.
#         - If float, then `min_samples_leaf` is a fraction and
#           `ceil(min_samples_leaf * n_samples)` are the minimum
#           number of samples for each node.
#
#         .. versionchanged:: 0.18
#            Added float values for fractions.
#
#     min_weight_fraction_leaf : float, default=0.0
#         The minimum weighted fraction of the sum total of weights (of all
#         the input samples) required to be at a leaf node. Samples have
#         equal weight when sample_weight is not provided.
#
#     max_features : int, float or {"auto", "sqrt", "log2"}, default=None
#         The number of features to consider when looking for the best split:
#
#         - If int, then consider `max_features` features at each split.
#         - If float, then `max_features` is a fraction and
#           `int(max_features * n_features)` features are considered at each
#           split.
#         - If "auto", then `max_features=n_features`.
#         - If "sqrt", then `max_features=sqrt(n_features)`.
#         - If "log2", then `max_features=log2(n_features)`.
#         - If None, then `max_features=n_features`.
#
#         Note: the search for a split does not stop until at least one
#         valid partition of the node samples is found, even if it requires to
#         effectively inspect more than ``max_features`` features.
#
#     random_state : int, RandomState instance or None, default=None
#         Controls the randomness of the estimator. The features are always
#         randomly permuted at each split, even if ``splitter`` is set to
#         ``"best"``. When ``max_features < n_features``, the algorithm will
#         select ``max_features`` at random at each split before finding the best
#         split among them. But the best found split may vary across different
#         runs, even if ``max_features=n_features``. That is the case, if the
#         improvement of the criterion is identical for several splits and one
#         split has to be selected at random. To obtain a deterministic behaviour
#         during fitting, ``random_state`` has to be fixed to an integer.
#         See :term:`Glossary <random_state>` for details.
#
#     max_leaf_nodes : int, default=None
#         Grow a tree with ``max_leaf_nodes`` in best-first fashion.
#         Best nodes are defined as relative reduction in impurity.
#         If None then unlimited number of leaf nodes.
#
#     min_impurity_decrease : float, default=0.0
#         A node will be split if this split induces a decrease of the impurity
#         greater than or equal to this value.
#
#         The weighted impurity decrease equation is the following::
#
#             N_t / N * (impurity - N_t_R / N_t * right_impurity
#                                 - N_t_L / N_t * left_impurity)
#
#         where ``N`` is the total number of samples, ``N_t`` is the number of
#         samples at the current node, ``N_t_L`` is the number of samples in the
#         left child, and ``N_t_R`` is the number of samples in the right child.
#
#         ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
#         if ``sample_weight`` is passed.
#
#         .. versionadded:: 0.19
#
#     min_impurity_split : float, default=0
#         Threshold for early stopping in tree growth. A node will split
#         if its impurity is above the threshold, otherwise it is a leaf.
#
#         .. deprecated:: 0.19
#            ``min_impurity_split`` has been deprecated in favor of
#            ``min_impurity_decrease`` in 0.19. The default value of
#            ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
#            will be removed in 0.25. Use ``min_impurity_decrease`` instead.
#
#     ccp_alpha : non-negative float, default=0.0
#         Complexity parameter used for Minimal Cost-Complexity Pruning. The
#         subtree with the largest cost complexity that is smaller than
#         ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
#         :ref:`minimal_cost_complexity_pruning` for details.
#
#         .. versionadded:: 0.22
#
#     Attributes
#     ----------
#     feature_importances_ : ndarray of shape (n_features,)
#         The feature importances.
#         The higher, the more important the feature.
#         The importance of a feature is computed as the
#         (normalized) total reduction of the criterion brought
#         by that feature. It is also known as the Gini importance [4]_.
#
#         Warning: impurity-based feature importances can be misleading for
#         high cardinality features (many unique values). See
#         :func:`sklearn.inspection.permutation_importance` as an alternative.
#
#     max_features_ : int
#         The inferred value of max_features.
#
#     n_features_ : int
#         The number of features when ``fit`` is performed.
#
#     n_outputs_ : int
#         The number of outputs when ``fit`` is performed.
#
#     tree_ : Tree instance
#         The underlying Tree object. Please refer to
#         ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
#         :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
#         for basic usage of these attributes.
#
#     See Also
#     --------
#     DecisionTreeClassifier : A decision tree classifier.
#
#     Notes
#     -----
#     The default values for the parameters controlling the size of the trees
#     (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
#     unpruned trees which can potentially be very large on some data sets. To
#     reduce memory consumption, the complexity and size of the trees should be
#     controlled by setting those parameter values.
#
#     References
#     ----------
#
#     .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning
#
#     .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
#            and Regression Trees", Wadsworth, Belmont, CA, 1984.
#
#     .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
#            Learning", Springer, 2009.
#
#     .. [4] L. Breiman, and A. Cutler, "Random Forests",
#            https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
#
#     Examples
#     --------
#     >>> from sklearn.datasets import load_diabetes
#     >>> from sklearn.model_selection import cross_val_score
#     >>> from sklearn.tree import DecisionTreeRegressor
#     >>> X, y = load_diabetes(return_X_y=True)
#     >>> regressor = DecisionTreeRegressor(random_state=0)
#     >>> cross_val_score(regressor, X, y, cv=10)
#     ...                    # doctest: +SKIP
#     ...
#     array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
#            0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
#     """
#
#     @_deprecate_positional_args
#     def __init__(
#         self,
#         *,
#         criterion="mse",
#         splitter="best",
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         min_weight_fraction_leaf=0.0,
#         max_features=None,
#         random_state=None,
#         max_leaf_nodes=None,
#         min_impurity_decrease=0.0,
#         min_impurity_split=None,
#         ccp_alpha=0.0
#     ):
#         super().__init__(
#             criterion=criterion,
#             splitter=splitter,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             min_weight_fraction_leaf=min_weight_fraction_leaf,
#             max_features=max_features,
#             max_leaf_nodes=max_leaf_nodes,
#             random_state=random_state,
#             min_impurity_decrease=min_impurity_decrease,
#             min_impurity_split=min_impurity_split,
#             ccp_alpha=ccp_alpha,
#         )
#
#     def fit(
#         self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"
#     ):
#         """Build a decision tree regressor from the training set (X, y).
#
#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             The training input samples. Internally, it will be converted to
#             ``dtype=np.float32`` and if a sparse matrix is provided
#             to a sparse ``csc_matrix``.
#
#         y : array-like of shape (n_samples,) or (n_samples, n_outputs)
#             The target values (real numbers). Use ``dtype=np.float64`` and
#             ``order='C'`` for maximum efficiency.
#
#         sample_weight : array-like of shape (n_samples,), default=None
#             Sample weights. If None, then samples are equally weighted. Splits
#             that would create child nodes with net zero or negative weight are
#             ignored while searching for a split in each node.
#
#         check_input : bool, default=True
#             Allow to bypass several input checking.
#             Don't use this parameter unless you know what you do.
#
#         X_idx_sorted : deprecated, default="deprecated"
#             This parameter is deprecated and has no effect.
#             It will be removed in v0.26.
#
#             .. deprecated :: 0.24
#
#         Returns
#         -------
#         self : DecisionTreeRegressor
#             Fitted estimator.
#         """
#
#         super().fit(
#             X,
#             y,
#             sample_weight=sample_weight,
#             check_input=check_input,
#             X_idx_sorted=X_idx_sorted,
#         )
#         return self
#
#     def _compute_partial_dependence_recursion(self, grid, target_features):
#         """Fast partial dependence computation.
#
#         Parameters
#         ----------
#         grid : ndarray of shape (n_samples, n_target_features)
#             The grid points on which the partial dependence should be
#             evaluated.
#         target_features : ndarray of shape (n_target_features)
#             The set of target features for which the partial dependence
#             should be evaluated.
#
#         Returns
#         -------
#         averaged_predictions : ndarray of shape (n_samples,)
#             The value of the partial dependence function on each grid point.
#         """
#         grid = np.asarray(grid, dtype=DTYPE, order="C")
#         averaged_predictions = np.zeros(
#             shape=grid.shape[0], dtype=np.float64, order="C"
#         )
#
#         self.tree_.compute_partial_dependence(
#             grid, target_features, averaged_predictions
#         )
#         return averaged_predictions
