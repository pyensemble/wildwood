"""

"""
# License: BSD 3 clause

from typing import Union
from warnings import warn
import threading
import numpy as np
from joblib import Parallel, effective_n_jobs, parallel_config, delayed

from tqdm import tqdm
from math import exp

from scipy.sparse import issparse

from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state, compute_sample_weight, check_array
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_consistent_length, _check_sample_weight
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

#from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

# from ._binning import Binner
from ._utils import split_strategy_mapping, criteria_mapping
from .preprocessing._checks import get_is_categorical
from .preprocessing import Encoder


eps = np.finfo("float32").eps

__all__ = ["ForestClassifier", "ForestRegressor"]


from wildwood.tree import TreeClassifier, TreeRegressor, serialize, unserialize
from ._tree import tree_regressor_weighted_depth

import numba

def _generate_train_valid_samples(random_state, n_samples):
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    indices = np.arange(n_samples)
    valid_mask = sample_counts == 0
    if valid_mask.sum() == 0:
        return _generate_train_valid_samples(
            (random_state + 1) % np.iinfo(np.uint32).max, n_samples
        )
    train_mask = np.logical_not(valid_mask)
    train_indices = indices[train_mask].astype(np.uintp)
    valid_indices = indices[valid_mask].astype(np.uintp)
    train_indices_count = sample_counts[train_mask].astype(np.uintp)
    return train_indices, valid_indices, train_indices_count

def _parallel_build_trees(tree, features_bitarray, y, sample_weight, random_state_bootstrap):
    n_samples = features_bitarray.n_samples
    sample_weight = sample_weight.copy()
    train_indices, valid_indices, train_indices_count = _generate_train_valid_samples(
        random_state_bootstrap, n_samples
    )
    sample_weight[train_indices] *= train_indices_count
    tree.fit(features_bitarray, y, train_indices, valid_indices, sample_weight)
    return tree

def _accumulate_prediction(predict, features_bitarray, out, lock, label_class_col=None):
    prediction = predict(features_bitarray)
    with lock:
        if label_class_col is None:
            out += prediction
        else:
            out[:, label_class_col] += prediction[:, 1]

def _compute_weighted_depth(weighted_depth, X, out, lock, tree_idx):
    tree_weighted_depth = weighted_depth(X)
    with lock:
        out[tree_idx] = tree_weighted_depth

def _get_tree_prediction(predict, X, out, lock, tree_idx):
    prediction = predict(X)
    with lock:
        out[tree_idx] = prediction

def _parallel_tree_apply(apply, X, out, lock, idx_tree):
    leaves = apply(X)
    with lock:
        out[idx_tree] = leaves

class ForestBase(BaseEstimator):
    def __init__(self, *, n_estimators, criterion, loss, step=1.0, aggregation=True, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_bins=256, categorical_features=None,
                 max_features="auto", subsample=int(2e5), cat_min_categories="log",
                 handle_unknown="consider_missing", n_jobs=1, random_state=None, verbose=False):
        self._fitted = False
        self._n_samples_ = None
        self._n_features_ = None
        self._class_weight = None
        self.max_features_ = None
        self.n_jobs_ = None
        self._random_states = None
        self.trees = None
        self._encoder = None
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.loss = loss
        self.step = step
        self.aggregation = aggregation
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.categorical_features = categorical_features
        self.is_categorical_ = None
        self.max_features = max_features
        self.subsample = subsample
        self.cat_min_categories = cat_min_categories
        self.handle_unknown = handle_unknown
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _generate_random_states(self):
        random_instance = check_random_state(self.random_state)
        _random_states = random_instance.randint(np.iinfo(np.uint32).max, size=self.n_estimators)
        if hasattr(self, "multiclass") and self.multiclass == "ovr":
            _random_states = np.repeat(_random_states, repeats=self._n_classes_)
        self._random_states_bootstrap = _random_states
        self._random_states_trees = _random_states

    def __getstate__(self):
        return {k: serialize(v) for k, v in self.__dict__.items()}

    def __setstate__(self, state):
        self.__dict__ = {k: unserialize(k, v) for k, v in state.items()}

    def fit(self, X, y, sample_weight=None, categorical_features=None, randomized_depth=False):
        if categorical_features is not None:
            self.categorical_features = categorical_features
        check_consistent_length(X, y)
        sample_weight_ = _check_sample_weight(sample_weight, X, dtype=np.float32)
        is_classifier = isinstance(self, ForestClassifier)
        n_estimators = self.n_estimators
        if is_classifier:
            y = self._encode_y(y)
            class_weight = self.class_weight
            multiclass = self.multiclass
            n_classes = self._n_classes_
            if self.aggregation and self.dirichlet == 0.0:
                raise ValueError("dirichlet must be > 0 when aggregation=True")
            if multiclass == "ovr":
                n_classes_per_tree = 2
                n_trees = n_classes * n_estimators
            else:
                n_classes_per_tree = n_classes
                n_trees = n_estimators
            self.n_trees_ = n_trees
            self.n_classes_per_tree_ = n_classes_per_tree
            if multiclass == "multinomial" and class_weight is not None:
                expanded_class_weight = compute_sample_weight(class_weight, y)
                sample_weight_ *= expanded_class_weight
            elif class_weight is not None:
                sample_weight_ = [
                    np.ascontiguousarray(sample_weight_ * compute_sample_weight(class_weight, yy), dtype=np.float32)
                    for yy in y
                ]
            else:
                sample_weight_ = [sample_weight_] * n_classes
        else:
            y = np.ascontiguousarray(y, dtype=np.float32)
        n_samples, n_features = X.shape
        max_depth_ = self._get_max_depth_(self.max_depth)
        max_features_ = self._get_max_features_(self.max_features, n_features)
        self.max_features_ = max_features_
        n_jobs_ = self._get_n_jobs_(self.n_jobs, self.n_estimators)
        self.n_jobs_ = n_jobs_
        self._generate_random_states()
        is_categorical = get_is_categorical(categorical_features, n_features)
        encoder = Encoder(
            max_bins=self.max_bins,
            subsample=self.subsample,
            is_categorical=is_categorical,
            cat_min_categories=self.cat_min_categories,
            handle_unknown=self.handle_unknown,
        )
        encoder.fit(X)
        is_categorical_ = encoder.is_categorical_
        self.is_categorical_ = is_categorical_
        features_bitarray = encoder.transform(X)
        self._encoder = encoder
        if is_classifier:
            trees = [
                TreeClassifier(
                    n_classes=n_classes_per_tree,
                    criterion=criteria_mapping[self.criterion],
                    loss=self.loss,
                    step=self.step,
                    aggregation=self.aggregation,
                    dirichlet=self.dirichlet,
                    max_depth=max_depth_ if not randomized_depth else np.random.randint(3, 51),
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    categorical_features=self.categorical_features,
                    is_categorical=is_categorical_,
                    max_features=max_features_,
                    random_state=random_state,
                    cat_split_strategy=split_strategy_mapping[self.cat_split_strategy],
                    verbose=self.verbose,
                )
                for random_state in self._random_states_trees
            ]
        else:
            trees = [
                TreeRegressor(
                    criterion=criteria_mapping[self.criterion],
                    loss=self.loss,
                    step=self.step,
                    aggregation=self.aggregation,
                    max_depth=max_depth_ if not randomized_depth else np.random.randint(3, 51),
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    categorical_features=self.categorical_features,
                    is_categorical=is_categorical_,
                    max_features=max_features_,
                    random_state=random_state,
                    verbose=self.verbose,
                )
                for _, random_state in zip(range(self.n_estimators), self._random_states_trees)
            ]
        ovr = is_classifier and self.multiclass == "ovr"
        with parallel_config(prefer="threads"):
            trees = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_build_trees)(
                    tree,
                    features_bitarray,
                    y if not ovr else y[tree_idx // n_estimators],
                    sample_weight_ if not ovr else sample_weight_[tree_idx // n_estimators],
                    self._random_states_bootstrap[tree_idx],
                )
                for tree_idx, tree in enumerate(trees)
            )
        self.trees = trees
        self._fitted = True
        self._n_samples_ = n_samples
        self._n_features_ = n_features
        return self

    def apply(self, X):
        features_bitarray, n_jobs, lock = self._predict_helper(X)
        n_samples = features_bitarray.n_samples
        out = np.empty((len(self.trees), n_samples), dtype=np.uintp)
        with parallel_config(prefer="threads"):
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_parallel_tree_apply)(tree.apply, features_bitarray, out, lock, idx_tree)
                for idx_tree, tree in enumerate(self.trees)
            )
        return out

    def _validate_y_class_weight(self, y):
        return y, None

    def _validate_X_predict(self, X, check_input=False):
        if check_input:
            X = check_array(X, accept_sparse="csr", dtype=np.float32)
            if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based sparse matrices")
        else:
            self._check_n_features(X, reset=False)
        return X

    def _predict_helper(self, X):
        check_is_fitted(self)
        features_bitarray = self._encoder.transform(X)
        n_jobs, _, _ = _partition_estimators(len(self.trees), self.n_jobs)
        lock = threading.Lock()
        return features_bitarray, n_jobs, lock

    def lighten(self):
        if hasattr(self, "trees"):
            for tree in self.trees:
                tree.lighten()

    def get_nodes(self, tree_idx):
        return self.trees[tree_idx].get_nodes()

    def path_leaf(self, X, tree_idx=0):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input=True)
        X_binned = self._bin_data(X, is_training_data=False)
        return self.trees[tree_idx].path_leaf(X_binned)

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, val):
        if self._fitted:
            raise ValueError("You cannot change n_estimators after calling fit")
        if not isinstance(val, int):
            raise ValueError("n_estimators must be an integer number")
        if val < 1:
            raise ValueError("n_estimators must be >= 1")
        self._n_estimators = val

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, val):
        if not isinstance(val, float):
            raise ValueError("step must be a float")
        if val <= 0:
            raise ValueError("step must be positive")
        self._step = val
        if self._fitted:
            for tree in self.trees:
                tree.step = val

    @property
    def aggregation(self):
        return self._aggregation

    @aggregation.setter
    def aggregation(self, val):
        if not isinstance(val, bool):
            raise ValueError("aggregation must be boolean")
        self._aggregation = val

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, val):
        if val is None:
            self._max_depth = val
        elif not isinstance(val, int):
            raise ValueError("max_depth must be None or an integer number")
        elif val < 2:
            raise ValueError("max_depth must be >= 2")
        else:
            self._max_depth = val

    @staticmethod
    def _get_max_depth_(max_depth):
        return np.iinfo(np.uintp).max if max_depth is None else max_depth

    @property
    def min_samples_split(self):
        return self._min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, val):
        if not isinstance(val, int):
            raise ValueError("min_samples_split must be an integer number")
        if val < 2:
            raise ValueError("min_samples_split must be >= 2")
        self._min_samples_split = val

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf

    @min_samples_leaf.setter
    def min_samples_leaf(self, val):
        if not isinstance(val, int):
            raise ValueError("min_samples_leaf must be an integer number")
        if val < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        self._min_samples_leaf = val

    @property
    def max_bins(self):
        return self._max_bins

    @max_bins.setter
    def max_bins(self, val):
        if not isinstance(val, int):
            raise ValueError("max_bins must be an integer number")
        if val < 3:
            raise ValueError("max_bins must be >= 3")
        self._max_bins = val

    @property
    def max_features(self):
        return self._max_features

    @max_features.setter
    def max_features(self, val):
        if val is None:
            self._max_features = None
        elif isinstance(val, str):
            if val in {"sqrt", "log2", "auto"}:
                self._max_features = val
            else:
                raise ValueError("max_features can be either None, an integer value or 'sqrt', 'log2' or 'auto'")
        elif isinstance(val, int):
            if val < 1:
                raise ValueError("max_features must be >= 1")
            self._max_features = val
        elif isinstance(val, float):
            if val > 1.0 or val <= 0.0:
                raise ValueError("max_features must be <= 1.0 and > 0.0")
            self._max_features = val
        else:
            raise ValueError("max_features can be either None, an integer value or 'sqrt', 'log2' or 'auto'")

    @staticmethod
    def _get_max_features_(max_features, n_features):
        if isinstance(max_features, str):
            if max_features == "auto":
                return max(1, int(np.sqrt(n_features)))
            if max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if max_features == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError("max_features can be either None, an integer value, a float between 0 and 1 or 'sqrt', 'log2' or 'auto'")
        if max_features is None:
            return n_features
        if isinstance(max_features, int):
            if max_features > n_features:
                raise ValueError("max_features must be <= n_features")
            return max_features
        if isinstance(max_features, float):
            if max_features > 1.0 or max_features <= 0.0:
                raise ValueError("max_features must be <= 1.0 and > 0.0")
            return max(1, int(max_features * n_features))
        raise ValueError("max_features can be either None, an integer value, a float between 0 and 1 or 'sqrt', 'log2' or 'auto'")

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, val):
        if not isinstance(val, int):
            raise ValueError("n_jobs must be an integer number")
        if val < -1 or val == 0:
            raise ValueError("n_jobs must be >= 1 or equal to -1")
        self._n_jobs = val

    @staticmethod
    def _get_n_jobs_(n_jobs, n_estimators):
        return min(effective_n_jobs(n_jobs), n_estimators)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, val):
        if self._fitted:
            raise ValueError("You cannot modify random_state after calling fit")
        if val is None:
            self._random_state = val
        elif isinstance(val, int):
            if val >= 0:
                self._random_state = val
            else:
                raise ValueError("random_state must be >= 0")
        else:
            raise ValueError("random_state must be either None or an integer number")

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        if not isinstance(val, bool):
            raise ValueError("verbose must be boolean")
        self._verbose = val

    @property
    def n_samples_in_(self):
        if self._fitted:
            return self._n_samples_
        raise ValueError("You must call fit before asking for n_samples_in_")

    @n_samples_in_.setter
    def n_samples_in_(self, val):
        if isinstance(val, int) and val >= 1:
            self._n_samples_ = val
        else:
            raise ValueError("n_samples_in_ must be an int >= 1")

    @property
    def n_features_in_(self):
        if self._fitted:
            return self._n_features_
        raise ValueError("You must call fit before asking for n_features_in_")

    @n_features_in_.setter
    def n_features_in_(self, val):
        if isinstance(val, int) and val >= 1:
            self._n_features_ = val
        else:
            raise ValueError("n_features_int_ must be an int >= 1")

class ForestClassifier(ForestBase, ClassifierMixin):
    def __init__(self, *, n_estimators=10, criterion="gini", loss="log", step=1.0, aggregation=True,
                 dirichlet=0.5, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_bins=256,
                 categorical_features=None, max_features="auto", handle_unknown="consider_missing",
                 cat_min_categories="log", subsample=int(2e5), n_jobs=1, random_state=None, verbose=False,
                 class_weight=None, multiclass="multinomial", cat_split_strategy="binary"):
        super(ForestClassifier, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            loss=loss,
            step=step,
            aggregation=aggregation,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            categorical_features=categorical_features,
            max_features=max_features,
            subsample=subsample,
            cat_min_categories=cat_min_categories,
            handle_unknown=handle_unknown,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self._classes_ = None
        self._n_classes_ = None
        self.n_trees_ = None
        self.n_classes_per_tree_ = None
        self._multiclass = None
        self.dirichlet = dirichlet
        self.class_weight = class_weight
        self.multiclass = multiclass
        self.cat_split_strategy = cat_split_strategy

    def _encode_y(self, y):
        multiclass = self.multiclass
        check_classification_targets(y)
        label_encoder = LabelBinarizer() if multiclass == "ovr" else LabelEncoder()
        pre_encoded_y = label_encoder.fit_transform(y)
        self._classes_ = label_encoder.classes_
        n_classes_ = self._classes_.shape[0]
        self._n_classes_ = n_classes_
        if multiclass == "ovr":
            if n_classes_ <= 2:
                if self.verbose:
                    warn(
                        "Only two classes were detected: switching to "
                        'multiclass="multinomial" instead of "ovr"'
                    )
                self.multiclass = "multinomial"
                encoded_y = pre_encoded_y.ravel().astype(np.float32)
            else:
                encoded_y = [
                    np.ascontiguousarray(pre_encoded_y[:, i], dtype=np.float32)
                    for i in range(n_classes_)
                ]
        else:
            encoded_y = np.ascontiguousarray(pre_encoded_y, dtype=np.float32)
        return encoded_y

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)
        y = np.copy(y)
        expanded_class_weight = None
        if self.class_weight is not None:
            y_original = np.copy(y)
        y_store_unique_indices = np.zeros(y.shape, dtype=int)
        classes_, y_store_unique_indices[:] = np.unique(y, return_inverse=True)
        self._classes_ = classes_
        self._n_classes_ = classes_.shape[0]
        y = y_store_unique_indices
        if self.class_weight is not None:
            expanded_class_weight = compute_sample_weight(self.class_weight, y_original)
        return y, expanded_class_weight

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        features_bitarray, n_jobs, lock = self._predict_helper(X)
        all_proba = np.zeros((features_bitarray.n_samples, self.n_classes_))
        n_estimators = self.n_estimators
        ovr = self.multiclass == "ovr"
        with parallel_config(require="sharedmem"):
            Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_accumulate_prediction)(
                    tree.predict_proba,
                    features_bitarray,
                    all_proba,
                    lock,
                    label_class_col=tree_idx // n_estimators if ovr else None,
                )
                for tree_idx, tree in enumerate(self.trees)
            )
        if ovr:
            all_proba_sum = all_proba.sum(axis=1)
            mask = all_proba_sum <= eps
            all_proba[mask, :] = 1 / self.n_classes_
            all_proba[~mask, :] /= np.expand_dims(all_proba_sum[~mask], axis=1)
        else:
            all_proba /= len(self.trees)
        return all_proba

    def predict_proba_trees(self, X):
        check_is_fitted(self)
        X = self._validate_X_predict(X)
        features_bitarray, n_jobs, lock = self._predict_helper(X)
        n_samples = len(X)
        n_estimators = len(self.trees)
        n_classes = self.n_classes_
        probas = np.empty(
            (
                n_estimators,
                n_samples,
                2 if n_classes > 2 and self.multiclass == "ovr" else n_classes,
            )
        )
        with parallel_config(require="sharedmem"):
            Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_get_tree_prediction)(
                    e.predict_proba, features_bitarray, probas, lock, tree_idx
                )
                for tree_idx, e in enumerate(self.trees)
            )
        return probas

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, val):
        if not isinstance(val, str):
            raise ValueError("loss must be a string")
        if val != "log":
            raise ValueError("Only loss='log' is supported for now")
        self._loss = val

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, val):
        if not isinstance(val, str):
            raise ValueError("criterion must be a string")
        if val in {"gini", "entropy"}:
            self._criterion = val
        else:
            raise ValueError("Unknown criterion: " + val)

    @property
    def dirichlet(self):
        return self._dirichlet

    @dirichlet.setter
    def dirichlet(self, val):
        if not isinstance(val, float):
            raise ValueError("dirichlet must be a float")
        if val < 0.0:
            raise ValueError("dirichlet must be positive")
        recompute = hasattr(self, "_dirichlet") and self._dirichlet != val
        self._dirichlet = val
        if self._fitted and recompute:
            for tree in self.trees:
                tree.dirichlet = val

    @property
    def classes_(self):
        if self._fitted:
            return self._classes_
        raise ValueError("You must call fit before asking for classes_")

    @classes_.setter
    def classes_(self, _):
        raise ValueError("classes_ is a readonly attribute")

    @property
    def n_classes_(self):
        if self._fitted:
            return self._n_classes_
        raise ValueError("You must call fit before asking for n_classes_")

    @n_classes_.setter
    def n_classes_(self, _):
        raise ValueError("n_classes_ is a readonly attribute")

    @property
    def class_weight(self):
        return self._class_weight

    @class_weight.setter
    def class_weight(self, val):
        if val is None:
            self._class_weight = None
        elif isinstance(val, str):
            if val != "balanced":
                raise ValueError('class_weight can only be None or "balanced"')
            self._class_weight = val
        else:
            raise ValueError('class_weight can only be None or "balanced"')

    @property
    def multiclass(self):
        return self._multiclass

    @multiclass.setter
    def multiclass(self, val):
        if self._fitted:
            raise AttributeError(
                "You cannot change the multiclass option after calling fit"
            )
        if not isinstance(val, str):
            raise ValueError("multiclass must be a str")
        if val not in {"multinomial", "ovr"}:
            raise ValueError('multiclass must be either "multinomial" or "ovr"')
        self._multiclass = val

    @property
    def cat_split_strategy(self):
        return self._cat_split_strategy

    @cat_split_strategy.setter
    def cat_split_strategy(self, val):
        if self._fitted:
            raise AttributeError(
                "You cannot change the cat_split_strategy option after calling fit"
            )
        if not isinstance(val, str):
            raise ValueError("cat_split_strategy must be a str")
        if val not in {"binary", "all", "random"}:
            raise ValueError(
                'cat_split_strategy must be either "binary", "all" or "random"'
            )
        self._cat_split_strategy = val


class ForestRegressor(ForestBase, RegressorMixin):
    def __init__(self, *, n_estimators=10, criterion="mse", loss="mse", step=1.0, aggregation=True,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, max_bins=256,
                 categorical_features=None, max_features="auto", handle_unknown="consider_missing",
                 cat_min_categories="log", subsample=int(2e5), n_jobs=1, random_state=None, verbose=False):
        super(ForestRegressor, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            loss=loss,
            step=step,
            aggregation=aggregation,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            categorical_features=categorical_features,
            max_features=max_features,
            subsample=subsample,
            cat_min_categories=cat_min_categories,
            handle_unknown=handle_unknown,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def predict(self, X):
        features_bitarray, n_jobs, lock = self._predict_helper(X)
        all_preds = np.zeros(features_bitarray.n_samples)
        with parallel_config(require="sharedmem"):
            Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_accumulate_prediction)(tree.predict, features_bitarray, all_preds, lock)
                for tree in self.trees
            )
        all_preds /= len(self.trees)
        return all_preds

    def _weighted_depth(self, X):
        features_bitarray, n_jobs, lock = self._predict_helper(X)
        all_weighted_depths = np.zeros(
            (self.n_estimators, features_bitarray.n_samples), dtype=np.float32
        )
        with parallel_config(require="sharedmem"):
            Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_compute_weighted_depth)(
                    e.weighted_depth, features_bitarray, all_weighted_depths, lock, tree_idx
                )
                for tree_idx, e in enumerate(self.trees)
            )
        return all_weighted_depths

    def predict_trees(self, X):
        check_is_fitted(self)
        X = self._validate_X_predict(X)
        features_bitarray, n_jobs, lock = self._predict_helper(X)
        n_samples = len(X)
        n_estimators = len(self.trees)
        probas = np.empty((n_estimators, n_samples))
        with parallel_config(require="sharedmem"):
            Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_get_tree_prediction)(
                    e.predict, features_bitarray, probas, lock, tree_idx
                )
                for tree_idx, e in enumerate(self.trees)
            )
        return probas

    @property
    def criterion(self):
        return self._criterion

    @property
    def loss(self):
        return self._loss

    @criterion.setter
    def criterion(self, val):
        if not isinstance(val, str):
            raise ValueError("criterion must be a string")
        if val != "mse":
            raise ValueError("Only criterion='mse' is supported for now")
        self._criterion = val

    @loss.setter
    def loss(self, val):
        if not isinstance(val, str):
            raise ValueError("loss must be a string")
        if val != "mse":
            raise ValueError("Only loss='mse' is supported for now")
        self._loss = val
