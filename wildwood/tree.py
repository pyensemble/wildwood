# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains wildwood's trees. These are not intended for end-users.
"""

# TODO: the contents of this module should go in _tree.py with the jitclasses

from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from numba import float32, uintp

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._split import (
    SplitClassifier,
    SplitRegressor,
    find_best_split_classifier_along_feature,
    find_best_split_regressor_along_feature,
    copy_split_classifier,
    copy_split_regressor,
)
from ._grow import grow
from ._node import (
    NodeClassifierContext,
    NodeRegressorContext,
    compute_node_classifier_context,
    compute_node_regressor_context,
)
from ._tree_context import TreeClassifierContext, TreeRegressorContext
from ._tree import (
    _TreeClassifier,
    _TreeRegressor,
    get_nodes,
    tree_classifier_predict_proba,
    tree_regressor_predict,
    tree_regressor_weighted_depth,
    get_nodes_regressor,
    get_nodes_classifier,
)


class TreeBase(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        *,
        n_bins,
        criterion,
        loss,
        step,
        aggregation,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        categorical_features,
        max_features,
        random_state,
        verbose=0,
    ):
        self._tree = None
        self._tree_context = None

        self.n_bins = n_bins
        self.criterion = criterion
        self.loss = loss
        self.step = step
        self.aggregation = aggregation
        # print("self.aggregation:", self.aggregation)
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

    def get_nodes(self):
        return get_nodes(self._tree)


class TreeClassifier(ClassifierMixin, TreeBase):
    def __init__(
        self,
        *,
        n_bins,
        n_classes,
        criterion,
        loss,
        step,
        aggregation,
        dirichlet,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        categorical_features,
        max_features,
        random_state,
        verbose=0,
    ):
        super().__init__(
            n_bins=n_bins,
            criterion=criterion,
            loss=loss,
            step=step,
            aggregation=aggregation,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            categorical_features=categorical_features,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_classes = n_classes
        self.dirichlet = dirichlet

    def fit(self, X, y, train_indices, valid_indices, sample_weights):
        n_classes = self.n_classes
        max_bins = self.n_bins - 1
        # TODO: on obtiendra cette info via le binner qui est dans la foret
        n_samples, n_features = X.shape
        n_bins_per_feature = max_bins * np.ones(n_features)
        n_bins_per_feature = n_bins_per_feature.astype(np.intp)

        # Create the tree object, which is mostly a data container for the nodes
        tree = _TreeClassifier(n_features, n_classes)

        # We build a tree context, that contains global information about
        # the data, in particular the way we'll organize data into contiguous
        # node indexes both for training and validation samples
        tree_context = TreeClassifierContext(
            X,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            self.n_classes,
            self.n_bins - 1,
            n_bins_per_feature,
            self.max_features,
            self.aggregation,
            self.dirichlet,
            self.step,
        )

        node_context = NodeClassifierContext(tree_context)
        best_split = SplitClassifier(tree_context.n_classes)
        candidate_split = SplitClassifier(tree_context.n_classes)
        compute_node_context = compute_node_classifier_context
        grow(
            tree,
            tree_context,
            node_context,
            compute_node_context,
            find_best_split_classifier_along_feature,
            copy_split_classifier,
            best_split,
            candidate_split,
        )
        self._tree = tree
        self._tree_context = tree_context
        return self

    def predict_proba(self, X):
        proba = tree_classifier_predict_proba(
            self._tree, X, self._tree_context.aggregation, self._tree_context.step
        )
        return proba

    def get_nodes(self):
        return get_nodes_classifier(self._tree)


class TreeRegressor(TreeBase, RegressorMixin):
    def __init__(
        self,
        *,
        n_bins,
        criterion,
        loss,
        step=1.0,
        aggregation=True,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        categorical_features=None,
        max_features="auto",
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            n_bins=n_bins,
            criterion=criterion,
            loss=loss,
            step=step,
            aggregation=aggregation,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            categorical_features=categorical_features,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, X, y, train_indices, valid_indices, sample_weights):
        max_bins = self.n_bins - 1
        # TODO: on obtiendra cette info via le binner qui est dans la foret
        n_samples, n_features = X.shape
        n_bins_per_feature = max_bins * np.ones(n_features)
        n_bins_per_feature = n_bins_per_feature.astype(np.intp)

        # Create the tree object, which is mostly a data container for the nodes
        tree = _TreeRegressor(n_features)

        # We build a tree context, that contains global information about
        # the data, in particular the way we'll organize data into contiguous
        # node indexes both for training and validation samples
        tree_context = TreeRegressorContext(
            X,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            self.n_bins - 1,
            n_bins_per_feature,
            uintp(self.max_features),
            self.aggregation,
            float32(self.step),
        )

        node_context = NodeRegressorContext(tree_context)
        best_split = SplitRegressor()
        candidate_split = SplitRegressor()
        compute_node_context = compute_node_regressor_context

        grow(
            tree,
            tree_context,
            node_context,
            compute_node_context,
            find_best_split_regressor_along_feature,
            copy_split_regressor,
            best_split,
            candidate_split,
        )
        self._tree = tree
        self._tree_context = tree_context
        return self

    def predict(self, X):
        y_pred = tree_regressor_predict(
            self._tree, X, self._tree_context.aggregation, self._tree_context.step
        )
        return y_pred

    def weighted_depth(self, X):
        return tree_regressor_weighted_depth(self._tree, X, self.step)

    def get_nodes(self):
        return get_nodes_regressor(self._tree)
