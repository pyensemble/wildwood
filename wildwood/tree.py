# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains wildwood's trees. These are not intended for end-users.
"""

import numbers
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, check_array
from sklearn.base import ClassifierMixin
from sklearn.base import is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args

from numba import _helperlib


from ._grow import grow
from ._node import NodeContext
from ._tree_context import TreeContext
from ._tree import Tree, get_nodes, tree_predict_proba


# =============================================================================
# Base decision tree
# =============================================================================

# TODO: tout le preprocessing doit etre fait dans la classe forest. Les arbres ne
#  doivent pas etre accessibles en dehors de la foret


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

        if getattr(y, "dtype", None) != np.float32 or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.float32)

        # Check parameters
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth
        # max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

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
        n_bins_per_feature = n_bins_per_feature.astype(np.intp)

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



        return self

    def get_nodes(self):
        return get_nodes(self._tree)

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        if check_input:

            X = check_array(X, accept_sparse="csr", dtype=np.float32)
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

        proba = _tree_old.tree_predict_proba(self.tree_, X)
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
        proba = tree_predict_proba(
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

