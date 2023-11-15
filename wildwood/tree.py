# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains wildwood's trees. These are not intended for end-users.
"""

# TODO: the contents of this module should go in _tree.py with the jitclasses

from abc import ABCMeta
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._split import (
    SplitClassifier,
    SplitRegressor,
    find_best_split_classifier_along_feature,
    find_best_split_regressor_along_feature,
)
from ._grow import grow, recompute_node_predictions, compute_tree_weights
from ._node import (
    NodeClassifierContext,
    NodeRegressorContext,
    compute_node_classifier_context,
    compute_node_regressor_context,
)
from ._tree_context import (
    TreeClassifierContext,
    tree_classifier_context_type,
    TreeRegressorContext,
    tree_regressor_context_type,
)
from ._tree import (
    _TreeClassifier,
    tree_classifier_type,
    _TreeRegressor,
    tree_regressor_type,
    tree_classifier_predict_proba,
    tree_regressor_predict,
    tree_regressor_weighted_depth,
    get_nodes_regressor,
    get_nodes_classifier,
    tree_apply,
)
from ._tree import path_leaf as _path_leaf
from .preprocessing.features_bitarray import FeaturesBitArray, spec_features_bit_array


# TODO: the categorical_features parameter used in TreeClassifier and TreeRegressor
#  is not used. What is used instead is is_categorical, so we can remove safely
#  categorical_features from these classes


class TreeBase(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        is_classifier,
        # max_bins,
        criterion,
        loss,
        step,
        aggregation,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        categorical_features,
        is_categorical,
        max_features,
        random_state,
        verbose=0,
    ):
        self.is_classifier = is_classifier

        if is_classifier:
            self._tree_classifier = None
            self._tree_classifier_context = None
        else:
            self._tree_regressor = None
            self._tree_regressor_context = None

        self._train_indices = None
        self._valid_indices = None

        # self.max_bins = max_bins
        self.criterion = criterion
        self.loss = loss
        self._step = step
        self.aggregation = aggregation
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_features = categorical_features
        self.is_categorical = is_categorical
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
        if self.is_classifier:
            return self._tree_classifier.max_depth
        else:
            return self._tree_regressor.max_depth

    def get_actual_depth(self):
        nd = self.get_nodes()
        return max(nd.depth)

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        if self.is_classifier:
            return self._tree_classifier.n_leaves
        else:
            return self._tree_regressor.n_leaves

    def apply(self, X):
        if self.is_classifier:
            return tree_apply(self._tree_classifier, X)
        else:
            return tree_apply(self._tree_regressor, X)

    def __getstate__(self):
        return {k: serialize(v) for k, v in self.__dict__.items()}

    def __setstate__(self, state):
        self.__dict__ = {k: unserialize(k, v) for k, v in state.items()}

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, val):
        # We skip the checks and assume they are done by the forest
        if self._step != val:
            self._step = val
            if self.is_classifier:
                if hasattr(self, "_tree_classifier_context"):
                    self._tree_classifier_context.step = val
                    if self._tree_classifier_context.aggregation:
                        compute_tree_weights(
                            self._tree_classifier.nodes,
                            self._tree_classifier.node_count,
                            val,
                        )
            else:
                if hasattr(self, "_tree_regressor_context"):
                    self._tree_regressor_context.step = val
                    if self._tree_regressor_context.aggregation:
                        compute_tree_weights(
                            self._tree_regressor.nodes,
                            self._tree_regressor.node_count,
                            val,
                        )

    def lighten(self):
        if hasattr(self, "_train_indices"):
            self._train_indices = None

        if hasattr(self, "_valid_indices"):
            self._valid_indices = None

        if self.is_classifier:
            if hasattr(self, "_tree_classifier_context"):
                self._tree_classifier_context = None
            if hasattr(self, "_tree_classifier"):
                self._tree_classifier.nodes = self._tree_classifier.nodes[:self._tree_classifier.node_count]
                self._tree_classifier.bin_partitions = self._tree_classifier.bin_partitions[:self._tree_classifier.bin_partitions_end]
        else:
            if hasattr(self, "_tree_regressor_context"):
                self._tree_regressor_context = None
            if hasattr(self, "_tree_regressor"):
                self._tree_regressor.nodes = self._tree_regressor.nodes[:self._tree_regressor.node_count]
                self._tree_regressor.bin_partitions = self._tree_regressor.bin_partitions[:self._tree_regressor.bin_partitions_end]


class TreeClassifier(ClassifierMixin, TreeBase):
    def __init__(
        self,
        *,
        # max_bins,
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
        is_categorical,
        max_features,
        random_state,
        cat_split_strategy,
        verbose=0,
    ):
        super().__init__(
            is_classifier=True,
            # max_bins=max_bins,
            criterion=criterion,
            loss=loss,
            step=step,
            aggregation=aggregation,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            categorical_features=categorical_features,
            is_categorical=is_categorical,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_classes = n_classes
        # We set dirichlet like this at init to avoid launching what the property does
        self._dirichlet = dirichlet
        self._step = step
        self.cat_split_strategy = cat_split_strategy

    def fit(self, features_bitarray, y, train_indices, valid_indices, sample_weights):
        n_classes = self.n_classes
        random_state = self.random_state
        n_features = features_bitarray.n_features

        # Create the tree object, which is mostly a data container for the nodes
        node_count = 0
        capacity = 0
        bin_partitions_capacity = 0
        bin_partitions_end = 0
        tree_classifier = _TreeClassifier(
            n_features,
            n_classes,
            random_state,
            node_count,
            capacity,
            bin_partitions_capacity,
            bin_partitions_end,
        )

        # We build a tree context, that contains global information about
        # the data, in particular the way we'll organize data into contiguous
        # node indexes both for training and validation samples
        tree_classifier_context = TreeClassifierContext(
            features_bitarray,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            self.n_classes,
            self.max_features,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.aggregation,
            self.dirichlet,
            self._step,
            self.is_categorical,
            self.cat_split_strategy,
            self.criterion,
        )

        node_context = NodeClassifierContext(tree_classifier_context)
        best_split = SplitClassifier(
            tree_classifier_context.n_classes, tree_classifier_context.max_n_bins
        )
        compute_node_context = compute_node_classifier_context
        grow(
            tree_classifier,
            tree_classifier_context,
            node_context,
            compute_node_context,
            find_best_split_classifier_along_feature,
            best_split,
        )
        self._train_indices = train_indices
        self._valid_indices = valid_indices
        self._tree_classifier = tree_classifier
        self._tree_classifier_context = tree_classifier_context
        return self

    def predict_proba(self, features_bitarray):
        proba = tree_classifier_predict_proba(
            self._tree_classifier,
            features_bitarray,
            self.aggregation,
            self.step,
            # self._tree_classifier_context.aggregation,
            # self._tree_classifier_context.step,
        )
        return proba

    def get_nodes(self):
        return get_nodes_classifier(self._tree_classifier)

    def path_leaf(self, X):
        return _path_leaf(self._tree_classifier, X)

    @property
    def dirichlet(self):
        return self._dirichlet

    @dirichlet.setter
    def dirichlet(self, val):
        # We skip the checks and assume they are done by the forest
        if val != self.dirichlet:
            self._dirichlet = val
            # If the attribute _tree_classifier_context is there, then the tree has
            # already been trained and its predictions needs to be recomputed
            if hasattr(self, "_tree_classifier_context"):
                self._tree_classifier_context.dirichlet = val
                recompute_node_predictions(
                    self._tree_classifier, self._tree_classifier_context, val
                )


class TreeRegressor(TreeBase, RegressorMixin):
    def __init__(
        self,
        *,
        criterion,
        loss,
        step=1.0,
        aggregation=True,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        categorical_features=None,
        is_categorical=None,
        max_features="auto",
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            is_classifier=False,
            criterion=criterion,
            loss=loss,
            step=step,
            aggregation=aggregation,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            categorical_features=categorical_features,
            is_categorical=is_categorical,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, features_bitarray, y, train_indices, valid_indices, sample_weights):
        random_state = self.random_state
        # TODO: on obtiendra cette info via le binner qui est dans la foret
        n_features = features_bitarray.n_features
        # Create the tree object, which is mostly a data container for the nodes
        node_count = 0
        capacity = 0
        bin_partitions_capacity = 0
        bin_partitions_end = 0
        tree_regressor = _TreeRegressor(
            n_features,
            random_state,
            node_count,
            capacity,
            bin_partitions_capacity,
            bin_partitions_end,
        )

        # We build a tree context, that contains global information about
        # the data, in particular the way we'll organize data into contiguous
        # node indexes both for training and validation samples
        tree_regressor_context = TreeRegressorContext(
            features_bitarray,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            self.max_features,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.aggregation,
            self._step,
            self.is_categorical,
            self.criterion,
        )

        node_context = NodeRegressorContext(tree_regressor_context)
        best_split = SplitRegressor(tree_regressor_context.max_n_bins)
        compute_node_context = compute_node_regressor_context

        grow(
            tree_regressor,
            tree_regressor_context,
            node_context,
            compute_node_context,
            find_best_split_regressor_along_feature,
            best_split,
        )
        self._train_indices = train_indices
        self._valid_indices = valid_indices
        self._tree_regressor = tree_regressor
        self._tree_regressor_context = tree_regressor_context
        return self

    def predict(self, X):
        y_pred = tree_regressor_predict(
            self._tree_regressor,
            X,
            self.aggregation,
            self.step,
            # self._tree_regressor_context.aggregation,
            # self._tree_regressor_context.step,
        )
        return y_pred

    def weighted_depth(self, X):
        return tree_regressor_weighted_depth(self._tree_regressor, X, self.step)

    def get_nodes(self):
        return get_nodes_regressor(self._tree_regressor)


def serialize(obj):
    if isinstance(obj, _TreeClassifier):
        return {attr: getattr(obj, attr) for attr, _ in tree_classifier_type}
    elif isinstance(obj, _TreeRegressor):
        return {attr: getattr(obj, attr) for attr, _ in tree_regressor_type}
    elif isinstance(obj, TreeClassifierContext):
        dd = {}
        for attr, _ in tree_classifier_context_type:
            if attr == "features_bitarray":
                features_bitarray = getattr(obj, attr)
                dd[attr] = {
                    key: getattr(features_bitarray, key)
                    for key, _ in spec_features_bit_array
                }
            else:
                dd[attr] = getattr(obj, attr)
        return dd
        # return {attr: getattr(obj, attr) for attr, _ in tree_classifier_context_type}
    elif isinstance(obj, TreeRegressorContext):
        dd = {}
        for attr, _ in tree_regressor_context_type:
            if attr == "features_bitarray":
                features_bitarray = getattr(obj, attr)
                dd[attr] = {
                    key: getattr(features_bitarray, key)
                    for key, _ in spec_features_bit_array
                }
            else:
                dd[attr] = getattr(obj, attr)
        return dd
    elif isinstance(obj, FeaturesBitArray):
        return {attr: getattr(obj, attr) for attr, _ in spec_features_bit_array}
    else:
        return obj


def unserialize(key, val):
    if key == "_tree_classifier":
        n_features = val["n_features"]
        n_classes = val["n_classes"]
        random_state = val["random_state"]
        node_count = val["node_count"]
        capacity = val["capacity"]
        bin_partitions_capacity = val["bin_partitions_capacity"]
        bin_partitions_end = val["bin_partitions_end"]
        tree = _TreeClassifier(
            n_features,
            n_classes,
            random_state,
            node_count,
            capacity,
            bin_partitions_capacity,
            bin_partitions_end,
        )
        tree.nodes[:] = val["nodes"]
        tree.y_pred[:] = val["y_pred"]
        tree.bin_partitions[:] = val["bin_partitions"]
        return tree
    elif key == "_tree_regressor":
        n_features = val["n_features"]
        random_state = val["random_state"]
        node_count = val["node_count"]
        capacity = val["capacity"]
        bin_partitions_capacity = val["bin_partitions_capacity"]
        bin_partitions_end = val["bin_partitions_end"]
        tree = _TreeRegressor(
            n_features,
            random_state,
            node_count,
            capacity,
            bin_partitions_capacity,
            bin_partitions_end,
        )
        tree.nodes[:] = val["nodes"]
        tree.y_pred[:] = val["y_pred"]
        tree.bin_partitions[:] = val["bin_partitions"]
        return tree
    elif key == "_tree_classifier_context":
        dict_features_bitarray = val["features_bitarray"]
        n_samples = dict_features_bitarray["n_samples"]
        max_values = dict_features_bitarray["max_values"]
        features_bitarray = FeaturesBitArray(n_samples, max_values)
        features_bitarray.n_features = dict_features_bitarray["n_features"]
        features_bitarray.n_bits[:] = dict_features_bitarray["n_bits"]
        features_bitarray.offsets[:] = dict_features_bitarray["offsets"]
        features_bitarray.n_values_in_words[:] = dict_features_bitarray[
            "n_values_in_words"
        ]
        features_bitarray.bitarray[:] = dict_features_bitarray["bitarray"]
        features_bitarray.bitmasks[:] = dict_features_bitarray["bitmasks"]
        y = val["y"]
        sample_weights = val["sample_weights"]
        train_indices = val["train_indices"]
        valid_indices = val["valid_indices"]
        n_classes = val["n_classes"]
        max_depth = val["max_depth"]
        max_features = val["max_features"]
        min_samples_split = val["min_samples_split"]
        min_samples_leaf = val["min_samples_leaf"]
        aggregation = val["aggregation"]
        dirichlet = val["dirichlet"]
        step = val["step"]
        is_categorical = val["is_categorical"]
        cat_split_strategy = val["cat_split_strategy"]
        criterion = val["criterion"]
        tree_context = TreeClassifierContext(
            features_bitarray,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            n_classes,
            max_features,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            aggregation,
            dirichlet,
            step,
            is_categorical,
            cat_split_strategy,
            criterion,
        )
        tree_context.partition_train[:] = val["partition_train"]
        tree_context.partition_valid[:] = val["partition_valid"]
        tree_context.left_buffer[:] = val["left_buffer"]
        tree_context.right_buffer[:] = val["right_buffer"]
        return tree_context
    elif key == "_tree_regressor_context":
        dict_features_bitarray = val["features_bitarray"]
        n_samples = dict_features_bitarray["n_samples"]
        max_values = dict_features_bitarray["max_values"]
        features_bitarray = FeaturesBitArray(n_samples, max_values)
        features_bitarray.n_features = dict_features_bitarray["n_features"]
        features_bitarray.n_bits[:] = dict_features_bitarray["n_bits"]
        features_bitarray.offsets[:] = dict_features_bitarray["offsets"]
        features_bitarray.n_values_in_words[:] = dict_features_bitarray[
            "n_values_in_words"
        ]
        features_bitarray.bitarray[:] = dict_features_bitarray["bitarray"]
        features_bitarray.bitmasks[:] = dict_features_bitarray["bitmasks"]
        y = val["y"]
        sample_weights = val["sample_weights"]
        train_indices = val["train_indices"]
        valid_indices = val["valid_indices"]
        max_features = val["max_features"]
        max_depth = val["max_depth"]
        min_samples_split = val["min_samples_split"]
        min_samples_leaf = val["min_samples_leaf"]
        aggregation = val["aggregation"]
        step = val["step"]
        is_categorical = val["is_categorical"]
        criterion = val["criterion"]
        tree_context = TreeRegressorContext(
            features_bitarray,
            y,
            sample_weights,
            train_indices,
            valid_indices,
            max_features,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            aggregation,
            step,
            is_categorical,
            criterion,
        )
        tree_context.partition_train[:] = val["partition_train"]
        tree_context.partition_valid[:] = val["partition_valid"]
        tree_context.left_buffer[:] = val["left_buffer"]
        tree_context.right_buffer[:] = val["right_buffer"]
        return tree_context
    else:
        return val
