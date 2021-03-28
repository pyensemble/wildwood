# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains wildwood's trees. These are not intended for end-users.
"""

from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import MultiOutputMixin
from sklearn.utils.validation import check_is_fitted

from ._grow import grow
from ._node import NodeContext
from ._tree_context import TreeContext
from ._tree import Tree, get_nodes, tree_predict_proba


# =============================================================================
# Base decision tree
# =============================================================================


class TreeBase(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        n_bins,
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
        self._tree = None
        self._tree_context = None

        self.n_bins = n_bins
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
        # check_is_fitted(self)
        if self._tree is None:
            raise ValueError("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator." % {'name': type(self).__name__})
        return self._tree.max_depth

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


class TreeBinaryClassifier(ClassifierMixin, TreeBase):
    def __init__(
        self,
        *,
        n_bins,
        n_classes,
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
        random_state=None,  # random_state of a Tree is only for (samples) bootstrap use
        verbose=0,
    ):
        super().__init__(
            n_bins=n_bins,
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
        self.n_classes = n_classes

    def fit(self, X, y, train_indices, valid_indices, sample_weights):
        n_classes = self.n_classes
        max_bins = self.n_bins - 1
        # TODO: on obtiendra cette info via le binner qui est dans la foret
        n_samples, n_features = X.shape
        n_bins_per_feature = max_bins * np.ones(n_features)
        n_bins_per_feature = n_bins_per_feature.astype(np.intp)

        # Create the tree object, which is mostly a data container for the nodes
        tree = Tree(n_features, n_classes)

        # We build a tree context, that contains global information about
        # the data, in particular the way we'll organize data into contiguous
        # node indexes both for training and validation samples
        tree_context = TreeContext(
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
            self.categorical_features
        )

        node_context = NodeContext(tree_context)
        grow(tree, tree_context, node_context)
        self._tree = tree
        self._tree_context = tree_context
        return self

    def predict_proba(self, X):
        proba = tree_predict_proba(
            self._tree, X, self._tree_context.aggregation, self._tree_context.step
        )
        return proba
