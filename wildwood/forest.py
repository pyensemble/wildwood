"""

"""
# License: BSD 3 clause

from typing import Union
from warnings import warn
import threading
import numpy as np
from joblib import Parallel, effective_n_jobs

from tqdm import tqdm

from scipy.sparse import issparse

from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state, compute_sample_weight, check_array
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_consistent_length, _check_sample_weight
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from ._binning import Binner


__all__ = ["ForestClassifier", "ForestRegressor"]


from wildwood.tree import TreeClassifier, TreeRegressor


# TODO: bootstrap with stratification for unbalanced data ?
# TODO: bon il faut gerer: 1. le bootstrap 2. la stratification 3. les
#  sample_weight 4. le binning. On va pour l'instant aller au plus simple: 1.
#  on binne les donnees des de debut et on ne s'emmerde pas avec les
#  sample_weight. Le bootstrap et la stratification devront etre geres dans la
#  fonction _generate_train_valid_samples au dessus (qu'on appelle avec un
#  Parallel comme dans l'implem originelle des forets)


def _generate_train_valid_samples(random_state, n_samples):
    """
    This functions generates "in-the-bag" (train) and "out-of-the-bag" samples

    Parameters
    ----------
    random_state : None or int or RandomState
        Allows to specify the RandomState used for random splitting

    n_samples : int
        Total number of samples

    Returns
    -------
    output : tuple of theer numpy arrays
        * output[0] contains the indices of the training samples
        * output[1] contains the indices of the validation samples
        * output[2] contains the counts of the training samples
    """
    random_instance = check_random_state(random_state)
    # Sample the bootstrap samples (uniform sampling with replacement)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    indices = np.arange(n_samples)
    valid_mask = sample_counts == 0
    # For very small samples, we might end up with empty validation...
    if valid_mask.sum() == 0:
        return _generate_train_valid_samples(
            (random_state + 1) % np.iinfo(np.uint32).max, n_samples
        )
    else:
        train_mask = np.logical_not(valid_mask)
        train_indices = indices[train_mask].astype(np.uintp)
        valid_indices = indices[valid_mask].astype(np.uintp)
        train_indices_count = sample_counts[train_mask].astype(np.uintp)
        return train_indices, valid_indices, train_indices_count


def _parallel_build_trees(tree, X, y, sample_weight, random_state_bootstrap):
    """Private function used to fit a single tree in parallel.

    Parameters
    ----------
    tree : TreeClassifier or TreeRegressor
        The tree to fit

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples. Internally, it will be binned into a uint8
        dtype following LightGBM's histogram strategy. If a sparse matrix is
        provided, it will be converted into a sparse ``csc_matrix``.

    y : array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in
        regression).

    sample_weight : array-like of shape (n_samples,)
        Sample weights. If no weighting is used then it is a vector of ones.

    random_state_bootstrap : int
        The seed used to instantiate the random number generator

    Returns
    -------
    output : TreeClassifier or TreeRegressor
        The fitted tree
    """

    n_samples = X.shape[0]
    # A copy is required, since we'll modify it inplace for the bootstrap
    sample_weight = sample_weight.copy()

    train_indices, valid_indices, train_indices_count = _generate_train_valid_samples(
        random_state_bootstrap, n_samples
    )
    # We use bootstrap: sample repetition is achieved by multiplying the sample
    # weights by the sample counts. By construction, no repetition is possible in
    # validation data
    sample_weight[train_indices] *= train_indices_count

    # Fit the tree
    tree.fit(X, y, train_indices, valid_indices, sample_weight)
    return tree


def _accumulate_prediction(predict, X, out, lock, ovr_index=None):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X)

    with lock:
        if ovr_index is None:
            out += prediction
        else:
            out[:,ovr_index] += prediction[:,1]


def _compute_weighted_depth(weighted_depth, X, out, lock, tree_idx):
    tree_weighted_depth = weighted_depth(X)
    with lock:
        out[tree_idx] = tree_weighted_depth


def _get_tree_prediction(predict, X, out, lock, tree_idx):
    prediction = predict(X, check_input=False)
    with lock:
        out[tree_idx] = prediction


def _parallel_tree_apply(apply, X, out, lock, idx_tree):
    leaves = apply(X)
    with lock:
        out[idx_tree] = leaves


class ForestBase(BaseEstimator):
    """
    BLABLA
    """

    def __init__(
        self,
        *,
        n_estimators,
        criterion,
        loss,
        step=1.0,
        aggregation=True,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_bins=255,
        categorical_features=None,
        max_features="auto",
        n_jobs=1,
        random_state=None,
        verbose=False,
    ):
        self._fitted = False
        self._n_samples_ = None
        self._n_features_ = None
        self._class_weight = None
        self.max_features_ = None
        self.n_jobs_ = None
        self._random_states = None
        self.trees = None

        # Set the parameters. This calls the properties defined below
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
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _generate_random_states(self, n_states=None):
        """This helper method generates one seed (random_state) for each tree,
        that is used to generate at random bootstrap samples and columns subsampling.

        For testing purposes (see tests/test_forest.py::test_random_state) we save
        those into two separate arrays self._random_states_bootstrap and
        self._random_states_trees even if these are identical in non-testing settings
        """
        # Get the random instance
        random_instance = check_random_state(self.random_state)
        # Generate seeds for each tree and same them for testing
        _random_states = random_instance.randint(
            np.iinfo(np.uint32).max, size=n_states or self.n_estimators
        )
        self._random_states_bootstrap = _random_states
        self._random_states_trees = _random_states

    def fit(self, X, y, sample_weight=None):
        """
        Trains WildWood's forest predictor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be binned into a uint8
            dtype following LightGBM's histogram strategy. If a sparse matrix is
            provided, it will be converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            If None, then samples are equally weighted. Otherwise, samples are
            weighted. If sample_weight[42] = 3.0 then all computations do "as if"
            there were 3 lines with the same contents as X[42] in all computations
            (for split finding, node predictions and for the aggregation algorithm
            (computation of validation losses).

        Returns
        -------
        self : object
        """

        # TODO: Why only float64 ? What if the data is already binned ?
        X, y = self._validate_data(
            X, y, dtype=[np.float32], force_all_finite=False, order="F"
        )
        check_consistent_length(X, y)

        # In all cases we have a sample_weight_ vector: it contains only ones if
        # sample_weight=None.
        sample_weight_ = _check_sample_weight(sample_weight, X, dtype=np.float32)

        is_classifier = isinstance(self, ForestClassifier)
        n_estimators = self.n_estimators

        if is_classifier:
            y = self._encode_y(y)
            class_weight = self.class_weight
            multiclass = self.multiclass
            if class_weight is not None:
                if multiclass == "multinomial":
                    expanded_class_weight = compute_sample_weight(class_weight, y)
                    sample_weight_ *= expanded_class_weight
                else:
                    sample_weight_ = [np.ascontiguousarray(sample_weight_ * compute_sample_weight(class_weight, yy), dtype=np.float32) for yy in y]

            n_classes = self._n_classes_
            if multiclass == "ovr":
                n_classes_trees = 2
                n_trees = n_classes*n_estimators
            else:
                n_classes_trees = n_classes
                n_trees = n_estimators

        else:
            y = np.ascontiguousarray(y, dtype=np.float32)

        # TODO: deal properly with categorical features. What if these are specified ?
        self.is_categorical_, known_categories = self._check_categories(X)

        n_samples, n_features = X.shape
        # Let's get actual parameters based on the parameters passed by the user and
        # the data
        max_depth_ = self._get_max_depth_(self.max_depth)
        max_features_ = self._get_max_features_(self.max_features, n_features)
        self.max_features_ = max_features_
        n_jobs_ = self._get_n_jobs_(self.n_jobs, self.n_estimators)
        self.n_jobs_ = n_jobs_

        self._generate_random_states(n_trees)

        # Everywhere in the code, the convention is that n_bins == max_bins + 1,
        # since max_bins is the maximum number of bins, without the eventual bin for
        # missing values (+ 1 is for the missing values bin)
        n_bins = self.max_bins + 1  # + 1 for missing values

        # TODO: ici ici ici faut que je reprenne le code de scikit et pas de pygbm
        #  car ils gerent les features categorielles dans le mapper

        # TODO: Deal more intelligently with this. Do not bin if the data is already
        #  binned by test for dtype=='uint8' for instance
        self._bin_mapper = Binner(
            n_bins=n_bins,
            is_categorical=self.is_categorical_,
            known_categories=known_categories,
        )
        X_binned = self._bin_data(X, is_training_data=True)

        # TODO: Deal with categorical data
        # Uses binned data to check for missing values
        has_missing_values = (
            (X_binned == self._bin_mapper.missing_values_bin_idx_)
            .any(axis=0)
            .astype(np.uint8)
        )

        if is_classifier:
            # We are training a classifier
            trees = [
                TreeClassifier(
                    n_bins=n_bins,
                    n_classes=n_classes_trees,
                    criterion=self.criterion,
                    loss=self.loss,
                    step=self.step,
                    aggregation=self.aggregation,
                    dirichlet=self.dirichlet,
                    max_depth=max_depth_,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    categorical_features=self.categorical_features,
                    max_features=max_features_,
                    random_state=random_state,
                    verbose=self.verbose,
                )
                for _, random_state in zip(
                    range(n_trees), self._random_states_trees
                )

            ]

        else:
            # We are training a regressor
            trees = [
                TreeRegressor(
                    n_bins=n_bins,
                    criterion=self.criterion,
                    loss=self.loss,
                    step=self.step,
                    aggregation=self.aggregation,
                    max_depth=max_depth_,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    categorical_features=self.categorical_features,
                    max_features=max_features_,
                    random_state=random_state,
                    verbose=self.verbose,
                )
                for _, random_state in zip(
                    range(self.n_estimators), self._random_states_trees
                )
            ]
        # TODO : Question : do we use different sample weights for multiclass with ovr ?
        # Parallel loop: use threading since all the numba code releases the GIL

        ovr = is_classifier and self.multiclass == "ovr"
        if self.verbose:
            trees = Parallel(
                n_jobs=self.n_jobs, **_joblib_parallel_args(prefer="threads"),
            )(
                delayed(_parallel_build_trees)(
                    trees[ind],
                    X_binned,
                    y if not ovr else y[ind // n_estimators],
                    sample_weight_ if not ovr else sample_weight_[ind // n_estimators],
                    self._random_states_bootstrap[ind],
                )
                for ind in range(len(trees))
            )
        else:
            trees = Parallel(
                n_jobs=self.n_jobs, **_joblib_parallel_args(prefer="threads"),
            )(
                delayed(_parallel_build_trees)(
                    trees[ind],
                    X_binned,
                    y if not ovr else y[ind // n_estimators],
                    sample_weight_ if not ovr else sample_weight_[ind // n_estimators],
                    self._random_states_bootstrap[ind],
                )
                for ind in range(len(trees))
            )
        self.trees = trees
        self._fitted = True
        self._n_samples_ = n_samples
        self._n_features_ = n_features
        return self

    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X_binned, n_jobs, lock = self.predict_helper(X)
        n_samples = X_binned.shape[0]
        out = np.empty((len(self.trees), n_samples), dtype=np.uintp)
        Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer="threads"),
        )(
            delayed(_parallel_tree_apply)(tree.apply, X_binned, out, lock, idx_tree)
            for idx_tree, tree in enumerate(self.trees)
        )

        return out

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None

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

    def _check_categories(self, X):
        """Check and validate categorical features in X

        Return
        ------
        is_categorical : ndarray of shape (n_features,) or None, dtype=bool
            Indicates whether a feature is categorical. If no feature is
            categorical, this is None.
        known_categories : list of size n_features or None
            The list contains, for each feature:
                - an array of shape (n_categories,) with the unique cat values
                - None if the feature is not categorical
            None if no feature is categorical.
        """
        if self.categorical_features is None:
            return None, None

        categorical_features = np.asarray(self.categorical_features)

        if categorical_features.size == 0:
            return None, None

        if categorical_features.dtype.kind not in ("i", "b"):
            raise ValueError(
                "categorical_features must be an array-like of "
                "bools or array-like of ints."
            )

        n_features = X.shape[1]

        # check for categorical features as indices
        if categorical_features.dtype.kind == "i":
            if (
                np.max(categorical_features) >= n_features
                or np.min(categorical_features) < 0
            ):
                raise ValueError(
                    "categorical_features set as integer "
                    "indices must be in [0, n_features - 1]"
                )
            is_categorical = np.zeros(n_features, dtype=bool)
            is_categorical[categorical_features] = True
        else:
            if categorical_features.shape[0] != n_features:
                raise ValueError(
                    "categorical_features set as a boolean mask "
                    "must have shape (n_features,), got: "
                    f"{categorical_features.shape}"
                )
            is_categorical = categorical_features

        if not np.any(is_categorical):
            return None, None

        # compute the known categories in the training data. We need to do
        # that here instead of in the BinMapper because in case of early
        # stopping, the mapper only gets a fraction of the training data.
        known_categories = []

        for f_idx in range(n_features):
            if is_categorical[f_idx]:
                categories = np.unique(X[:, f_idx])
                missing = np.isnan(categories)
                if missing.any():
                    categories = categories[~missing]

                if categories.size > self.max_bins:
                    raise ValueError(
                        f"Categorical feature at index {f_idx} is "
                        f"expected to have a "
                        f"cardinality <= {self.max_bins}"
                    )

                if (categories >= self.max_bins).any():
                    raise ValueError(
                        f"Categorical feature at index {f_idx} is "
                        f"expected to be encoded with "
                        f"values < {self.max_bins}"
                    )
            else:
                categories = None
            known_categories.append(categories)

        return is_categorical, known_categories

    def _bin_data(self, X, is_training_data):
        """Bin data X.

        If is_training_data, then fit the _bin_mapper attribute.
        Else, the binned data is converted to a C-contiguous array.
        """

        description = "training" if is_training_data else "validation"
        if self.verbose:
            print(
                "Binning {:.3f} GB of {} data: ".format(X.nbytes / 1e9, description),
                end="",
                flush=True,
            )
        # tic = time()
        if is_training_data:
            X_binned = self._bin_mapper.fit_transform(X)  # F-aligned array
        else:
            X_binned = self._bin_mapper.transform(X)  # F-aligned array
            # We convert the array to C-contiguous since predicting is faster
            # with this layout (training is faster on F-arrays though)
            X_binned = np.ascontiguousarray(X_binned)
        # toc = time()
        # if self.verbose:
        #     duration = toc - tic
        #     print("{:.3f} s".format(duration))

        return X_binned

    def predict_helper(self, X):
        """A method used in all predict functions to avoid code duplication
        """
        # Is the forest fitted ?
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X, check_input=True)
        # TODO: we can also avoid data binning for predictions...
        X_binned = self._bin_data(X, is_training_data=False)
        n_jobs, _, _ = _partition_estimators(len(self.trees), self.n_jobs)
        lock = threading.Lock()
        return X_binned, n_jobs, lock

    def get_nodes(self, tree_idx):
        return self.trees[tree_idx].get_nodes()

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, val):
        if self._fitted:
            raise ValueError("You cannot change n_estimators after calling fit")
        else:
            if not isinstance(val, int):
                raise ValueError("n_estimators must be an integer number")
            elif val < 1:
                raise ValueError("n_estimators must be >= 1")
            else:
                self._n_estimators = val

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, val):
        if not isinstance(val, float):
            raise ValueError("step must be a float")
        elif val <= 0:
            raise ValueError("step must be positive")
        else:
            self._step = val

    @property
    def aggregation(self):
        return self._aggregation

    @aggregation.setter
    def aggregation(self, val):
        if not isinstance(val, bool):
            raise ValueError("aggregation must be boolean")
        else:
            self._aggregation = val

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, val):
        if val is None:
            self._max_depth = val
        else:
            if not isinstance(val, int):
                raise ValueError("max_depth must be None or an integer number")
            else:
                if val < 2:
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
        else:
            if val < 2:
                raise ValueError("min_samples_split must be >= 2")
            else:
                self._min_samples_split = val

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf

    @min_samples_leaf.setter
    def min_samples_leaf(self, val):
        if not isinstance(val, int):
            raise ValueError("min_samples_leaf must be an integer number")
        else:
            if val < 1:
                raise ValueError("min_samples_leaf must be >= 1")
            else:
                self._min_samples_leaf = val

    @property
    def max_bins(self):
        return self._max_bins

    @max_bins.setter
    def max_bins(self, val):
        if not isinstance(val, int):
            raise ValueError("max_bins must be an integer number")
        else:
            if (val < 2) or (val > 256):
                raise ValueError("max_bins must be between 2 and 256")
            else:
                self._max_bins = val

    # TODO: property for categorical_features here

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
                raise ValueError(
                    "max_features can be either None, an integer value "
                    "or 'sqrt', 'log2' or 'auto'"
                )
        elif isinstance(val, int):
            if val < 1:
                raise ValueError("max_features must be >= 1")
            else:
                self._max_features = val
        else:
            raise ValueError(
                "max_features can be either None, an integer value "
                "or 'sqrt', 'log2' or 'auto'"
            )

    @staticmethod
    def _get_max_features_(max_features, n_features):
        if isinstance(max_features, str):
            if max_features == "auto":
                return max(1, int(np.sqrt(n_features)))
            elif max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            elif max_features == "log2":
                return max(1, int(np.log2(n_features)))
            else:
                raise ValueError(
                    "max_features can be either None, an integer "
                    "value or 'sqrt', 'log2' or 'auto'"
                )
        elif max_features is None:
            return n_features
        elif isinstance(max_features, int):
            if max_features > n_features:
                raise ValueError("max_features must be <= n_features")
            else:
                return max_features
        else:
            raise ValueError(
                "max_features can be either None, an integer "
                "value or 'sqrt', 'log2' or 'auto'"
            )

    @max_bins.setter
    def max_bins(self, val):
        if not isinstance(val, int):
            raise ValueError("max_bins must be an integer number")
        else:
            if not 2 <= val <= 255:
                raise ValueError("max_bins must be between 2 and 256")
            else:
                self._max_bins = val

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, val):
        if not isinstance(val, int):
            raise ValueError("n_jobs must be an integer number")
        elif val < -1 or val == 0:
            raise ValueError("n_jobs must be >= 1 or equal to -1")
        else:
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
        else:
            if val is None:
                self._random_state = val
            elif isinstance(val, int):
                if val >= 0:
                    self._random_state = val
                else:
                    ValueError("random_state must be >= 0")
            else:
                ValueError("random_state must be either None or an integer number")

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        if not isinstance(val, bool):
            raise ValueError("verbose must be boolean")
        else:
            self._verbose = val

    # TODO: property for class_weight here

    @property
    def n_samples_(self):
        if self._fitted:
            return self._n_samples_
        else:
            raise ValueError("You must call fit before asking for n_features_")

    @n_samples_.setter
    def n_samples_(self, _):
        raise ValueError("n_samples_ is a readonly attribute")

    @property
    def n_features_(self):
        if self._fitted:
            return self._n_features_
        else:
            raise ValueError("You must call fit before asking for n_features_")

    @n_features_.setter
    def n_features_(self, _):
        raise ValueError("n_features_ is a readonly attribute")


class ForestClassifier(ForestBase, ClassifierMixin):
    """
    WildWood forest for classification.

    It grows in parallel `n_estimator` trees using bootstrap samples and aggregates
    their predictions (bagging). Each tree uses "in-the-bag" samples to grow itself
    and "out-of-the-bag" samples to compute aggregation weights for all possible
    subtrees of the whole tree.

    The prediction function of each tree in WildWood is very different from the one
    of a standard decision trees. Indeed, the predictions of a tree are computed here
    as an aggregation with exponential weights of all the predictions given by all
    possible subtrees (prunings) of the full tree. The required computations are
    performed efficiently thanks to a variant of the context tree weighting algorithm.

    Also, features are all binned with a maximum of 255 bins.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"gini", "entropy"}, default="gini"
        The impurity criterion used to measure the quality of a split. The supported
        impurity criteria are "gini" for the Gini impurity and "entropy" for the
        entropy impurity.

    loss : {"log"}, default="log"
        The loss used for the computation of the aggregation weights. Only "log"
        is supported for now, namely the log-loss for classification.

    step : float, default=1
        Step-size for the aggregation weights. Default is 1 for classification with
        the log-loss, which is usually the best choice. A larger value will lead to
        aggregation weights with the best validation loss.

    aggregation : bool, default=True
        Controls if aggregation is used in the trees. It is highly recommended to
        leave it as `True`.

    dirichlet : float, default=0.5
        Regularization level of the class frequencies used for predictions in each
        node. A good default is dirichlet=0.5 for binary classification.

    max_depth : int, default=None
        The maximum depth of a tree. If None, then nodes from the tree are split until
        they are "pure" (impurity is small enough) or until they contain
        min_samples_split samples.

    min_samples_split : int, default=2
        The minimum number of training samples and out-of-the-bag samples required to
        split a node. This must be >= 2.

    min_samples_leaf : int, default=1
        A split point is considered if it leaves at least ``min_samples_leaf``
        training samples and out-of-the-bag samples in the left and right childs.
        This must be >= 1.

    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.

    categorical_features : array-like of {bool, int} of shape (n_features) \
            or shape (n_categorical_features,), default=None.
        Indicates the categorical features.

        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.

        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be in [0, max_bins -1].

        Read more in the :ref:`User Guide <categorical_support_gbdt>`.

        .. versionadded:: 0.24

    max_features : {"auto", "sqrt", "log2"} or int, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        TODO: this is not true for now...

    n_jobs : int, default=1
        The number of jobs to run in parallel for :meth:`fit`, :meth:`predict`,
        :meth:`predict_proba`, :meth:`decision_path` and :meth:`apply`. All
        these methods are parallelized over the trees in the forets. ``n_jobs=-1``
        means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    class_weight : "balanced" or None, default=None
        Weights associated with classes. If None, all classes are supposed to have
        weight one. The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    multiclass : "multinomial" or "ovr", default="multinomial"
        Strategy to adopt in the multiclass situation. If "multinomial", n_estimators
        trees will be trained to make multiclass predictions. In the "ovr" mode the
        one versus all strategy is adopted, labels are binarized and n_classes*n_estimators
        trees are trained to make binary predictions and the final predictions are obtained
        as normalized scores. This parameter is ignored for binary classification.

    References
    ----------
    TODO: insert references

    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        criterion: str = "gini",
        loss: str = "log",
        step: float = 1.0,
        aggregation: bool = True,
        dirichlet: bool = 0.5,
        max_depth: Union[None, int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_bins: int = 255,
        categorical_features=None,
        max_features: Union[str, int] = "auto",
        n_jobs: int = 1,
        random_state=None,
        verbose: bool = False,
        class_weight=None,
        multiclass="multinomial",
    ):
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
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self._classes_ = None
        self._n_classes_ = None
        self.dirichlet = dirichlet
        self.class_weight = class_weight
        self.multiclass = multiclass

    def _encode_y(self, y):
        """
        Encode classes into {0, ..., n_classes - 1} and sets attributes classes_,
        n_classes_ and n_trees_per_iteration_

        Parameters
        ----------
        y : ndarrray
            Array of input labels

        Returns
        -------
        output : ndarray
            Encoded array of labels
        """
        #
        # and n_trees_per_iteration_
        multiclass = self.multiclass
        check_classification_targets(y)
        label_encoder = LabelBinarizer() if multiclass == "ovr" else LabelEncoder()
        pre_encoded_y = label_encoder.fit_transform(y)
        self._classes_ = label_encoder.classes_
        n_classes_ = self._classes_.shape[0]
        self._n_classes_ = n_classes_
        # only 1 tree for binary classification.
        # TODO: For multiclass classification, we build 1 tree per class.
        # self.n_trees_per_iteration_ = 1 if n_classes_ <= 2 else n_classes_
        if multiclass == "ovr" and n_classes_ <= 2:
            if self.verbose:
                print("WARNING : no more than two classes detected, one versus all strategy will NOT be used")
            self.multiclass = "multinomial"
        if self.multiclass == "ovr":
            #encoded_y = np.zeros((len(pre_encoded_y), n_classes_), dtype=np.float32, order="F")
            #encoded_y[np.arange(len(pre_encoded_y)),pre_encoded_y] = 1
            encoded_y = [np.ascontiguousarray(pre_encoded_y[:,i], dtype=np.float32) for i in range(n_classes_)]
        else:
            if multiclass == "multinomial":
                encoded_y = np.ascontiguousarray(pre_encoded_y, dtype=np.float32)
            else:
                encoded_y = np.ascontiguousarray(pre_encoded_y.flatten(), dtype=np.float32)
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
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        X_binned, n_jobs, lock = self.predict_helper(X)
        all_proba = np.zeros((X_binned.shape[0], self.n_classes_))
        n_estimators = self.n_estimators

        if self.multiclass == "ovr":
            ovr_index = lambda ind : ind // n_estimators
        else:
            ovr_index = lambda _ : None
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem"),
        )(
            delayed(_accumulate_prediction)(
                self.trees[ind].predict_proba, X_binned, all_proba, lock, ovr_index=ovr_index(ind)
            )
            for ind in range(len(self.trees))
        )
        if ovr_index(0) is None: # quick check if we are doing ovr
            all_proba /= len(self.trees)
        else: # if yes, the normalization is not known in advance
            all_proba /= all_proba.sum(axis=1, keepdims=True)
        return all_proba

    def predict_proba_trees(self, X):
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)
        # TODO: we can also avoid data binning for predictions...
        X_binned = self._bin_data(X, is_training_data=False)
        n_samples, _ = X.shape
        n_estimators = len(self.trees)
        n_jobs, _, _ = _partition_estimators(n_estimators, self.n_jobs)
        n_classes = self.n_classes_
        probas = np.empty((n_estimators, n_samples, 2 if n_classes > 2 and self.multiclass == "ovr" else n_classes))

        lock = threading.Lock()
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem"),
        )(
            delayed(_get_tree_prediction)(
                e.predict_proba, X_binned, probas, lock, tree_idx
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

    @loss.setter
    def loss(self, val):
        if not isinstance(val, str):
            raise ValueError("loss must be a string")
        else:
            if val != "log":
                raise ValueError("Only loss='log' is supported for now")
            else:
                self._loss = val

    @criterion.setter
    def criterion(self, val):
        if not isinstance(val, str):
            raise ValueError("criterion must be a string")
        else:
            if val != "gini":
                raise ValueError("Only criterion='gini' is supported for now")
            else:
                self._criterion = val

    @property
    def dirichlet(self):
        return self._dirichlet

    @dirichlet.setter
    def dirichlet(self, val):
        if not isinstance(val, float):
            raise ValueError("dirichlet must be a float")
        else:
            if val < 0.0:
                raise ValueError("dirichlet must be positive")
            else:
                self._dirichlet = val

    @property
    def classes_(self):
        if self._fitted:
            return self._classes_
        else:
            raise ValueError("You must call fit before asking for classes_")

    @classes_.setter
    def classes_(self, _):
        raise ValueError("classes_ is a readonly attribute")

    @property
    def n_classes_(self):
        if self._fitted:
            return self._n_classes_
        else:
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
            else:
                self._class_weight = val
        else:
            raise ValueError('class_weight can only be None or "balanced"')

    @property
    def multiclass(self):
        return self._multiclass

    @multiclass.setter
    def multiclass(self, val):
        if self._fitted:
            raise AttributeError("You cannot change the multiclass option after calling fit")
        if not isinstance(val, str):
            raise ValueError("multiclass must be a str")
        else:
            if val not in ["multinomial", "ovr"]:
                raise ValueError("multiclass must be either 'multinomial' or 'ovr'")
            else:
                self._multiclass = val


class ForestRegressor(ForestBase, RegressorMixin):
    """
    WildWood forest for regression.

    It grows in parallel `n_estimator` trees using bootstrap samples and aggregates
    their predictions (bagging). Each tree uses "in-the-bag" samples to grow itself
    and "out-of-the-bag" samples to compute aggregation weights for all possible
    subtrees of the whole tree.

    The prediction function of each tree in WildWood is very different from the one
    of a standard decision trees. Indeed, the predictions of a tree are computed here
    as an aggregation with exponential weights of all the predictions given by all
    possible subtrees (prunings) of the full tree. The required computations are
    performed efficiently thanks to a variant of the context tree weighting algorithm.

    Also, features are all binned with a maximum of 255 bins.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"mse"}, default="mse"
        The impurity criterion used to measure the quality of a split. The only
        supported impurity criterion is "mse" for the least-squares impurity

    loss : {"mse"}, default="mse"
        The loss used for the computation of the aggregation weights. Only "mse"
        is supported for now, namely the least-squares loss for regression

    step : float, default=1
        Step-size for the aggregation weights. Default is 1 for classification with
        the log-loss, which is usually the best choice. A larger value will lead to
        aggregation weights with the best validation loss.

    aggregation : bool, default=True
        Controls if aggregation is used in the trees. It is highly recommended to
        leave it as `True`.

    max_depth : int, default=None
        The maximum depth of a tree. If None, then nodes from the tree are split until
        they are "pure" (impurity is small enough) or until they contain
        min_samples_split samples.

    min_samples_split : int, default=2
        The minimum number of training samples and out-of-the-bag samples required to
        split a node. This must be >= 2.

    min_samples_leaf : int, default=1
        A split point is considered if it leaves at least ``min_samples_leaf``
        training samples and out-of-the-bag samples in the left and right childs.
        This must be >= 1.

    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.

    categorical_features : array-like of {bool, int} of shape (n_features) \
            or shape (n_categorical_features,), default=None.
        Indicates the categorical features.

        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.

        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be in [0, max_bins -1].

        Read more in the :ref:`User Guide <categorical_support_gbdt>`.

        .. versionadded:: 0.24

    max_features : {"auto", "sqrt", "log2"} or int, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        TODO: this is not true for now...

    n_jobs : int, default=1
        The number of jobs to run in parallel for :meth:`fit`, :meth:`predict`,
        :meth:`predict_proba`, :meth:`decision_path` and :meth:`apply`. All
        these methods are parallelized over the trees in the forets. ``n_jobs=-1``
        means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    References
    ----------
    TODO: insert references

    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        criterion: str = "mse",
        loss: str = "mse",
        step: float = 1.0,
        aggregation: bool = True,
        max_depth: Union[None, int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_bins: int = 255,
        categorical_features=None,
        max_features: Union[str, int] = "auto",
        n_jobs: int = 1,
        random_state=None,
        verbose: bool = False,
    ):
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
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def predict(self, X):
        X_binned, n_jobs, lock = self.predict_helper(X)
        all_preds = np.zeros(X_binned.shape[0])

        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem"),
        )(
            delayed(_accumulate_prediction)(tree.predict, X_binned, all_preds, lock)
            for tree in self.trees
        )
        all_preds /= len(self.trees)
        return all_preds

    def weighted_depth(self, X):
        X_binned, n_jobs, lock = self.predict_helper(X)
        all_weighted_depths = np.zeros(
            (self.n_estimators, X_binned.shape[0]), dtype=np.float32
        )
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem"),
        )(
            delayed(_compute_weighted_depth)(
                e.weighted_depth, X_binned, all_weighted_depths, lock, tree_idx
            )
            for tree_idx, e in enumerate(self.trees)
        )
        return all_weighted_depths

    def predict_trees(self, X):
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)
        # TODO: we can also avoid data binning for predictions...
        X_binned = self._bin_data(X, is_training_data=False)
        n_samples, _ = X.shape
        n_estimators = len(self.trees)
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        probas = np.empty((n_estimators, n_samples, n_features))

        lock = threading.Lock()
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem"),
        )(
            delayed(_get_tree_prediction)(
                e.predict_proba, X_binned, probas, lock, tree_idx
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
        else:
            if val != "mse":
                raise ValueError("Only criterion='mse' is supported for now")
            else:
                self._criterion = val

    @loss.setter
    def loss(self, val):
        if not isinstance(val, str):
            raise ValueError("loss must be a string")
        else:
            if val != "mse":
                raise ValueError("Only loss='mse' is supported for now")
            else:
                self._loss = val
