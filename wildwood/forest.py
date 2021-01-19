"""

"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# License: BSD 3 clause


import numbers
from warnings import catch_warnings, simplefilter, warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np

# from scipy.sparse import issparse
# from scipy.sparse import hstack as sparse_hstack
from joblib import Parallel

from sklearn.base import ClassifierMixin, RegressorMixin, MultiOutputMixin

# from sklearn.metrics import r2_score
# from sklearn.preprocessing import OneHotEncoder
# from ..tree import (
#     DecisionTreeClassifier,
#     DecisionTreeRegressor,
#     ExtraTreeClassifier,
#     ExtraTreeRegressor,
# )

# from ..tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array, compute_sample_weight

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_is_fitted,
    check_consistent_length,
    _check_sample_weight,
)
from sklearn.preprocessing import LabelEncoder


# from ..exceptions import DataConversionWarning
# from ._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

from ._binning import Binner

from ._utils import np_size_t, max_size_t

__all__ = ["BinaryClassifier"]

# from wildwood._utils import MAX_INT

from wildwood.tree import TreeBinaryClassifier


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
        return _generate_train_valid_samples((random_state + 1) % max_size_t, n_samples)
    else:
        # print("sample_indices: ", sample_indices)
        # print("sample_counts: ", sample_counts)
        train_mask = np.logical_not(valid_mask)
        train_indices = indices[train_mask].astype(np_size_t)
        valid_indices = indices[valid_mask].astype(np_size_t)
        train_indices_count = sample_counts[train_mask].astype(np_size_t)
        return train_indices, valid_indices, train_indices_count


# TODO: faudrait que la parallelisation marche des de le debut...
def _parallel_build_trees(tree, X, y, sample_weight, tree_idx, n_trees, verbose=0):
    """
    Private function used to fit a single tree in parallel.
    """
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    n_samples = X.shape[0]
    if sample_weight is None:
        sample_weight = np.ones((n_samples,), dtype=np.float32)
    else:
        # Do a copy with float32 dtype
        sample_weight = sample_weight.astype(np.float32)

    train_indices, valid_indices, train_indices_count = _generate_train_valid_samples(
        tree.random_state, n_samples
    )

    # sample_weight_train = sample_weight_full[train_indices]
    # sample_weight_valid = sample_weight_full[valid_indices]
    # We use bootstrap: sample repetition is achieved by multiplying the sample
    # weights by the sample counts. By construction, no repetition is possible in
    # validation data
    sample_weight[train_indices] *= train_indices_count

    # TODO: je ne comprends pas les lignes commentees suivantes. Pour l'instant,
    #  il faut que sample weight contienne deja les poids si
    #  class_weighted="balanced" dans la foret. NB : je crois que ca vient du fait
    #  qu'on peut appeler plusieurs fois fit avec des sample_weight differents
    # if class_weight == "subsample":
    #     with catch_warnings():
    #         simplefilter("ignore", DeprecationWarning)
    #         curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
    # elif class_weight == "balanced_subsample":
    #     curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

    tree.fit(
        X,
        y,
        train_indices,
        valid_indices,
        sample_weight,
        check_input=False,
    )

    return tree


# class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
#     """
#     Base class for forests of trees.
#
#     Warning: This class should not be used directly. Use derived classes
#     instead.
#     """
#
#     @abstractmethod
#     def __init__(
#         self,
#         base_estimator,
#         n_estimators=100,
#         *,
#         estimator_params=tuple(),
#         bootstrap=False,
#         oob_score=False,
#         n_jobs=None,
#         random_state=None,
#         verbose=0,
#         warm_start=False,
#         class_weight=None,
#         max_samples=None
#     ):
#         super().__init__(
#             base_estimator=base_estimator,
#             n_estimators=n_estimators,
#             estimator_params=estimator_params,
#         )
#
#         self.bootstrap = bootstrap
#         self.oob_score = oob_score
#         self.n_jobs = n_jobs
#         self.random_state = random_state
#         self.verbose = verbose
#         self.warm_start = warm_start
#         self.class_weight = class_weight
#         self.max_samples = max_samples


# @property
# def feature_importances_(self):
#     """
#     The impurity-based feature importances.
#
#     The higher, the more important the feature.
#     The importance of a feature is computed as the (normalized)
#     total reduction of the criterion brought by that feature.  It is also
#     known as the Gini importance.
#
#     Warning: impurity-based feature importances can be misleading for
#     high cardinality features (many unique values). See
#     :func:`sklearn.inspection.permutation_importance` as an alternative.
#
#     Returns
#     -------
#     feature_importances_ : ndarray of shape (n_features,)
#         The values of this array sum to 1, unless all trees are single node
#         trees consisting of only the root node, in which case it will be an
#         array of zeros.
#     """
#     check_is_fitted(self)
#
#     all_importances = Parallel(n_jobs=self.n_jobs,
#                                **_joblib_parallel_args(prefer='threads'))(
#         delayed(getattr)(tree, 'feature_importances_')
#         for tree in self.estimators_ if tree.tree_.node_count > 1)
#
#     if not all_importances:
#         return np.zeros(self.n_features_, dtype=np.float64)
#
#     all_importances = np.mean(all_importances,
#                               axis=0, dtype=np.float64)
#     return all_importances / np.sum(all_importances)


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


# class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
#     """
#     Base class for forest of trees-based classifiers.
#
#     Warning: This class should not be used directly. Use derived classes
#     instead.
#     """
#
#     @abstractmethod
#     def __init__(
#         self,
#         base_estimator,
#         n_estimators=100,
#         *,
#         estimator_params=tuple(),
#         bootstrap=False,
#         oob_score=False,
#         n_jobs=None,
#         random_state=None,
#         verbose=0,
#         warm_start=False,
#         class_weight=None,
#         max_samples=None
#     ):
#         super().__init__(
#             base_estimator,
#             n_estimators=n_estimators,
#             estimator_params=estimator_params,
#             bootstrap=bootstrap,
#             oob_score=oob_score,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose,
#             warm_start=warm_start,
#             class_weight=class_weight,
#             max_samples=max_samples,
#         )


class ForestBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    WildWood for binary classification.

    It grows in parallel `n_estimator` trees using bootstrap samples and aggregates
    their predictions (bagging). Each tree uses "in-the-bag" samples to grow itself
    and "out-of-the-bag" samples to compute aggregation weights for all possible
    subtrees of the whole tree.

    The prediction function of each tree in WildWood is very different from the one
    of a standard decision trees. Indeed, the predictions of a tree are computed here
    as an aggregation with exponential weights of all the predictions given by all
    possible subtrees (prunings) of the full tree. The required computations are
    performed efficiently thanks to a variant of the context tree weighting algorithm.

    TODO: discuss histogram strategy

    TODO: insert references

    TODO: update doc link
    Read more in the :ref:`User Guide <forest>`.

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
        the log-loss, which is usually the best choice.

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

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node (not a leaf)

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be in a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` samples in the left and right childs.

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

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    """

    def __init__(
        self,
        *,
        n_estimators=100,
        criterion="gini",
        loss="log",
        step=1.0,
        aggregation=True,
        dirichlet=0.5,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_bins=255,
        categorical_features=None,
        max_features="auto",
        # TODO: default for n_jobs must be integer ?
        n_jobs=1,
        random_state=None,
        verbose=False,
        class_weight=None,
    ):
        self._fitted = False
        self._n_features = None
        # TODO: only a single output for now...
        self.n_outputs_ = 1

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.loss = loss
        self.step = step
        self.aggregation = aggregation
        self.dirichlet = dirichlet
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.categorical_features = categorical_features
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight

    def _encode_y(self, y):
        # encode classes into 0 ... n_classes - 1 and sets attributes classes_
        # and n_trees_per_iteration_
        check_classification_targets(y)

        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        # only 1 tree for binary classification. For multiclass classification,
        # we build 1 tree per class.
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y.astype(np.float64, copy=False)
        return encoded_y

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
        """
        # Validate or convert input data

        # TODO: reprendre ce que j'avais dans _classes.py et gerer la creation des bins
        # TODO: tout simplifier au cas classification binaire
        # TODO : on a fait un copier/coller depuis scikit-learn ici...

        # TODO: Why only float64 ? What is the data is already binned ?
        X, y = self._validate_data(X, y, dtype=[np.float64], force_all_finite=False)
        y = self._encode_y(y)
        check_consistent_length(X, y)

        # Do not create unit sample weights by default to later skip some
        # computation
        # if sample_weight is not None:
        #     sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
        #     # TODO: remove when PDP suports sample weights
        #     self._fitted_with_sw = True

        rng = check_random_state(self.random_state)

        # # When warm starting, we want to re-use the same seed that was used
        # # the first time fit was called (e.g. for subsampling or for the
        # # train/val split).
        # if not (self.warm_start and self._is_fitted()):
        #     self._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")

        # TODO: mettre un max de trucs dans les properties
        # self._validate_parameters()

        # used for validation in predict
        n_samples, self._n_features = X.shape

        self.is_categorical_, known_categories = self._check_categories(X)

        # we need this stateful variable to tell raw_predict() that it was
        # called from fit() (this current method), and that the data it has
        # received is pre-binned.
        # predicting is faster on pre-binned data, so we want early stopping
        # predictions to be made on pre-binned data. Unfortunately the _scorer
        # can only call predict() or predict_proba(), not raw_predict(), and
        # there's no way to tell the scorer that it needs to predict binned
        # data.
        self._in_fit = True

        # if isinstance(self.loss, str):
        #     self._loss = self._get_loss(sample_weight=sample_weight)
        # elif isinstance(self.loss, BaseLoss):
        #     self._loss = self.loss

        # if self.early_stopping == "auto":
        #     self.do_early_stopping_ = n_samples > 10000
        # else:
        #     self.do_early_stopping_ = self.early_stopping

        # self._use_validation_data = self.validation_fraction is not None
        # if self.do_early_stopping_ and self._use_validation_data:
        #     # stratify for classification
        #     stratify = y if hasattr(self._loss, "predict_proba") else None

        # TODO: bootstrap avec stratification pour jeux de donnees mal balances ?
        #  Quelle est la bonne facon de gerer ca ?

        # Save the state of the RNG for the training and validation split.
        # This is needed in order to have the same split when using
        # warm starting.
        #     if sample_weight is None:
        #         X_train, X_val, y_train, y_val = train_test_split(
        #             X,
        #             y,
        #             test_size=self.validation_fraction,
        #             stratify=stratify,
        #             random_state=self._random_seed,
        #         )
        #         sample_weight_train = sample_weight_val = None
        #     else:
        #         # TODO: incorporate sample_weight in sampling here, as well as
        #         # stratify
        #         (
        #             X_train,
        #             X_val,
        #             y_train,
        #             y_val,
        #             sample_weight_train,
        #             sample_weight_val,
        #         ) = train_test_split(
        #             X,
        #             y,
        #             sample_weight,
        #             test_size=self.validation_fraction,
        #             stratify=stratify,
        #             random_state=self._random_seed,
        #         )
        # else:
        #     X_train, y_train, sample_weight_train = X, y, sample_weight
        #     X_val = y_val = sample_weight_val = None

        # TODO: bon il faut gerer: 1. le bootstrap 2. la stratification 3. les
        #  sample_weight 4. le binning. On va pour l'instant aller au plus simple: 1.
        #  on binne les donnees des de debut et on ne s'emmerde pas avec les
        #  sample_weight. Le bootstrap et la stratification devront etre geres dans la
        #  fonction _generate_train_valid_samples au dessus (qu'on appelle avec un
        #  Parallel comme dans l'implem originelle des forets)

        # Bin the data
        # For ease of use of the API, the user-facing GBDT classes accept the
        # parameter max_bins, which doesn't take into account the bin for
        # missing values (which is always allocated). However, since max_bins
        # isn't the true maximal number of bins, all other private classes
        # (binmapper, histbuilder...) accept n_bins instead, which is the
        # actual total number of bins. Everywhere in the code, the
        # convention is that n_bins == max_bins + 1
        n_bins = self.max_bins + 1  # + 1 for missing values

        # TODO: ici ici ici faut que je reprenne le code de scikit et pas de pygbm
        #  car ils gerent les features categorielles dans le mapper

        self._bin_mapper = Binner(
            n_bins=n_bins,
            is_categorical=self.is_categorical_,
            known_categories=known_categories,
            # TODO: what to do with this ?
            # random_state=self._random_seed,
        )
        X_binned = self._bin_data(X, is_training_data=True)

        # X_binned_train = self._bin_data(X_train, is_training_data=True)
        # if X_val is not None:
        #     X_binned_val = self._bin_data(X_val, is_training_data=False)
        # else:
        #     X_binned_val = None

        # Uses binned data to check for missing values
        has_missing_values = (
            (X_binned == self._bin_mapper.missing_values_bin_idx_)
            .any(axis=0)
            .astype(np.uint8)
        )

        if self.verbose:
            print("Fitting gradient boosted rounds:")

        n_samples = X_binned.shape[0]

        # if not self.warm_start or not hasattr(self, "estimators_"):
        #     # Free allocated memory, if any

        # TODO: ici on initialise les estimateurs. Gerer le warm-start plus tard
        self.estimators_ = []

        trees = [
            TreeBinaryClassifier(
                criterion=self.criterion,
                loss=self.loss,
                step=self.step,
                aggregation=self.aggregation,
                dirichlet=self.dirichlet,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                categorical_features=self.categorical_features,
                max_features=self.max_features,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            for _ in range(self.n_estimators)
        ]

        # trees = [
        #     self._make_estimator(
        #         append=False,
        #         # TODO: correct random_state here ?
        #         random_state=rng,
        #     )
        #     for i in range(self.n_estimators)
        # ]

        # Parallel loop: we prefer the threading backend as the Cython code
        # for fitting the trees is internally releasing the Python GIL
        # making threading more efficient than multiprocessing in
        # that case. However, for joblib 0.12+ we respect any
        # parallel_backend contexts set at a higher level,
        # since correctness does not rely on using threads.
        trees = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer="threads"),
        )(
            delayed(_parallel_build_trees)(
                # tree, X, y, sample_weight, i, len(trees),
                tree,
                X_binned,
                y,
                # TODO: deal with sample_weight
                None,
                i,
                len(trees),
                verbose=self.verbose,
            )
            for i, tree in enumerate(trees)
        )

        self.trees = trees

        # tree, X, y, sample_weight, tree_idx, n_trees, verbose=0

        # Collect newly grown trees
        # self.estimators_.extend(trees)

        # if self.oob_score:
        #     self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        # if hasattr(self, "classes_") and self.n_outputs_ == 1:
        # if hasattr(self, "classes_"):
        #     self.n_classes_ = self.n_classes_[0]
        #     self.classes_ = self.classes_[0]

        return self

        # # First time calling fit, or no warm start
        # if not (self._is_fitted() and self.warm_start):
        #     # Clear random state and score attributes
        #     self._clear_state()
        #
        #     # initialize raw_predictions: those are the accumulated values
        #     # predicted by the trees for the training data. raw_predictions has
        #     # shape (n_trees_per_iteration, n_samples) where
        #     # n_trees_per_iterations is n_classes in multiclass classification,
        #     # else 1.
        #     self._baseline_prediction = self._loss.get_baseline_prediction(
        #         y_train, sample_weight_train, self.n_trees_per_iteration_
        #     )
        #     raw_predictions = np.zeros(
        #         shape=(self.n_trees_per_iteration_, n_samples),
        #         dtype=self._baseline_prediction.dtype,
        #     )
        #     raw_predictions += self._baseline_prediction
        #
        #     # predictors is a matrix (list of lists) of TreePredictor objects
        #     # with shape (n_iter_, n_trees_per_iteration)
        #     self._predictors = predictors = []
        #
        #     # Initialize structures and attributes related to early stopping
        #     self._scorer = None  # set if scoring != loss
        #     raw_predictions_val = None  # set if scoring == loss and use val
        #     self.train_score_ = []
        #     self.validation_score_ = []
        #
        #     if self.do_early_stopping_:
        #         # populate train_score and validation_score with the
        #         # predictions of the initial model (before the first tree)
        #
        #         if self.scoring == "loss":
        #             # we're going to compute scoring w.r.t the loss. As losses
        #             # take raw predictions as input (unlike the scorers), we
        #             # can optimize a bit and avoid repeating computing the
        #             # predictions of the previous trees. We'll re-use
        #             # raw_predictions (as it's needed for training anyway) for
        #             # evaluating the training loss, and create
        #             # raw_predictions_val for storing the raw predictions of
        #             # the validation data.
        #
        #             if self._use_validation_data:
        #                 raw_predictions_val = np.zeros(
        #                     shape=(self.n_trees_per_iteration_, X_binned_val.shape[0]),
        #                     dtype=self._baseline_prediction.dtype,
        #                 )
        #
        #                 raw_predictions_val += self._baseline_prediction
        #
        #             self._check_early_stopping_loss(
        #                 raw_predictions,
        #                 y_train,
        #                 sample_weight_train,
        #                 raw_predictions_val,
        #                 y_val,
        #                 sample_weight_val,
        #             )
        #         else:
        #             self._scorer = check_scoring(self, self.scoring)
        #             # _scorer is a callable with signature (est, X, y) and
        #             # calls est.predict() or est.predict_proba() depending on
        #             # its nature.
        #             # Unfortunately, each call to _scorer() will compute
        #             # the predictions of all the trees. So we use a subset of
        #             # the training set to compute train scores.
        #
        #             # Compute the subsample set
        #             (
        #                 X_binned_small_train,
        #                 y_small_train,
        #                 sample_weight_small_train,
        #             ) = self._get_small_trainset(
        #                 X_binned_train, y_train, sample_weight_train, self._random_seed
        #             )
        #
        #             self._check_early_stopping_scorer(
        #                 X_binned_small_train,
        #                 y_small_train,
        #                 sample_weight_small_train,
        #                 X_binned_val,
        #                 y_val,
        #                 sample_weight_val,
        #             )
        #     begin_at_stage = 0
        #
        # # warm start: this is not the first time fit was called
        # else:
        #     # Check that the maximum number of iterations is not smaller
        #     # than the number of iterations from the previous fit
        #     if self.max_iter < self.n_iter_:
        #         raise ValueError(
        #             "max_iter=%d must be larger than or equal to "
        #             "n_iter_=%d when warm_start==True" % (self.max_iter, self.n_iter_)
        #         )
        #
        #     # Convert array attributes to lists
        #     self.train_score_ = self.train_score_.tolist()
        #     self.validation_score_ = self.validation_score_.tolist()
        #
        #     # Compute raw predictions
        #     raw_predictions = self._raw_predict(X_binned_train)
        #     if self.do_early_stopping_ and self._use_validation_data:
        #         raw_predictions_val = self._raw_predict(X_binned_val)
        #     else:
        #         raw_predictions_val = None
        #
        #     if self.do_early_stopping_ and self.scoring != "loss":
        #         # Compute the subsample set
        #         (
        #             X_binned_small_train,
        #             y_small_train,
        #             sample_weight_small_train,
        #         ) = self._get_small_trainset(
        #             X_binned_train, y_train, sample_weight_train, self._random_seed
        #         )
        #
        #     # Get the predictors from the previous fit
        #     predictors = self._predictors
        #
        #     begin_at_stage = self.n_iter_
        #
        # # initialize gradients and hessians (empty arrays).
        # # shape = (n_trees_per_iteration, n_samples).
        # gradients, hessians = self._loss.init_gradients_and_hessians(
        #     n_samples=n_samples,
        #     prediction_dim=self.n_trees_per_iteration_,
        #     sample_weight=sample_weight_train,
        # )
        #
        # for iteration in range(begin_at_stage, self.max_iter):
        #
        #     if self.verbose:
        #         iteration_start_time = time()
        #         print(
        #             "[{}/{}] ".format(iteration + 1, self.max_iter), end="", flush=True
        #         )
        #
        #     # Update gradients and hessians, inplace
        #     self._loss.update_gradients_and_hessians(
        #         gradients, hessians, y_train, raw_predictions, sample_weight_train
        #     )
        #
        #     # Append a list since there may be more than 1 predictor per iter
        #     predictors.append([])
        #
        #     # Build `n_trees_per_iteration` trees.
        #     for k in range(self.n_trees_per_iteration_):
        #         grower = TreeGrower(
        #             X_binned_train,
        #             gradients[k, :],
        #             hessians[k, :],
        #             n_bins=n_bins,
        #             n_bins_non_missing=self._bin_mapper.n_bins_non_missing_,
        #             has_missing_values=has_missing_values,
        #             is_categorical=self.is_categorical_,
        #             monotonic_cst=self.monotonic_cst,
        #             max_leaf_nodes=self.max_leaf_nodes,
        #             max_depth=self.max_depth,
        #             min_samples_leaf=self.min_samples_leaf,
        #             l2_regularization=self.l2_regularization,
        #             shrinkage=self.learning_rate,
        #         )
        #         grower.grow()
        #
        #         acc_apply_split_time += grower.total_apply_split_time
        #         acc_find_split_time += grower.total_find_split_time
        #         acc_compute_hist_time += grower.total_compute_hist_time
        #
        #         if self._loss.need_update_leaves_values:
        #             self._loss.update_leaves_values(
        #                 grower, y_train, raw_predictions[k, :], sample_weight_train
        #             )
        #
        #         predictor = grower.make_predictor(
        #             binning_thresholds=self._bin_mapper.bin_thresholds_
        #         )
        #         predictors[-1].append(predictor)
        #
        #         # Update raw_predictions with the predictions of the newly
        #         # created tree.
        #         tic_pred = time()
        #         _update_raw_predictions(raw_predictions[k, :], grower)
        #         toc_pred = time()
        #         acc_prediction_time += toc_pred - tic_pred
        #
        #     should_early_stop = False
        #     if self.do_early_stopping_:
        #         if self.scoring == "loss":
        #             # Update raw_predictions_val with the newest tree(s)
        #             if self._use_validation_data:
        #                 for k, pred in enumerate(self._predictors[-1]):
        #                     raw_predictions_val[k, :] += pred.predict_binned(
        #                         X_binned_val, self._bin_mapper.missing_values_bin_idx_
        #                     )
        #
        #             should_early_stop = self._check_early_stopping_loss(
        #                 raw_predictions,
        #                 y_train,
        #                 sample_weight_train,
        #                 raw_predictions_val,
        #                 y_val,
        #                 sample_weight_val,
        #             )
        #
        #         else:
        #             should_early_stop = self._check_early_stopping_scorer(
        #                 X_binned_small_train,
        #                 y_small_train,
        #                 sample_weight_small_train,
        #                 X_binned_val,
        #                 y_val,
        #                 sample_weight_val,
        #             )
        #
        #     if self.verbose:
        #         self._print_iteration_stats(iteration_start_time)
        #
        #     # maybe we could also early stop if all the trees are stumps?
        #     if should_early_stop:
        #         break
        #
        # if self.verbose:
        #     duration = time() - fit_start_time
        #     n_total_leaves = sum(
        #         predictor.get_n_leaf_nodes()
        #         for predictors_at_ith_iteration in self._predictors
        #         for predictor in predictors_at_ith_iteration
        #     )
        #     n_predictors = sum(
        #         len(predictors_at_ith_iteration)
        #         for predictors_at_ith_iteration in self._predictors
        #     )
        #     print(
        #         "Fit {} trees in {:.3f} s, ({} total leaves)".format(
        #             n_predictors, duration, n_total_leaves
        #         )
        #     )
        #     print(
        #         "{:<32} {:.3f}s".format(
        #             "Time spent computing histograms:", acc_compute_hist_time
        #         )
        #     )
        #     print(
        #         "{:<32} {:.3f}s".format(
        #             "Time spent finding best splits:", acc_find_split_time
        #         )
        #     )
        #     print(
        #         "{:<32} {:.3f}s".format(
        #             "Time spent applying splits:", acc_apply_split_time
        #         )
        #     )
        #     print(
        #         "{:<32} {:.3f}s".format("Time spent predicting:", acc_prediction_time)
        #     )
        #
        # self.train_score_ = np.asarray(self.train_score_)
        # self.validation_score_ = np.asarray(self.validation_score_)
        # del self._in_fit  # hard delete so we're sure it can't be used anymore
        # return self


    def get_nodes(self, tree_idx):
        return self.trees[tree_idx].get_nodes()


    def _validate_y_class_weight(self, y):
        # TODO: c'est un copier / coller de scikit. Simplifier au cas classif binaire
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(
                y[:, k], return_inverse=True
            )
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ("balanced", "balanced_subsample")
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError(
                        "Valid presets for class_weight include "
                        '"balanced" and "balanced_subsample".'
                        'Given "%s".' % self.class_weight
                    )
                if self.warm_start:
                    warn(
                        'class_weight presets "balanced" or '
                        '"balanced_subsample" are '
                        "not recommended for warm_start if the fitted data "
                        "differs from the full dataset. In order to use "
                        '"balanced" weights, use compute_class_weight '
                        '("balanced", classes, y). In place of y you can use '
                        "a large enough sample of the full training set "
                        "target to properly estimate the class frequency "
                        "distributions. Pass the resulting weights as the "
                        "class_weight parameter."
                    )

            if self.class_weight != "balanced_subsample" or not self.bootstrap:
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight, y_original)

        return y, expanded_class_weight

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

        # TODO: c'est un copier / coller de scikit-learn. Simplifier au cas
        #  classification binaire. Et il faut binner les features avant de predire

        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_), dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                    np.argmax(proba[k], axis=1), axis=0
                )

            return predictions

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
        # TODO: c'est un copier / coller de scikit-learn. Simplifier au cas
        #  classification binaire. Et il faut binner les features avant de predire

        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem"),
        )(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_
        )

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

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
        # TODO: c'est un copier / coller de scikit-learn. Simplifier au cas
        #  classification binaire. Et il faut binner les features avant de predire

        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

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
        X = self._validate_X_predict(X)
        results = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer="threads"),
        )(delayed(tree.apply)(X, check_input=False) for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, X):
        """
        Return the decision path in the forest.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.

        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.

        """
        X = self._validate_X_predict(X)
        indicators = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer="threads"),
        )(
            delayed(tree.decision_path)(X, check_input=False)
            for tree in self.estimators_
        )

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, val):
        raise ValueError("`n_features` is a readonly attribute")

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, val):
        if self._fitted:
            raise ValueError("You cannot modify `n_estimators` after calling `fit`")
        else:
            if not isinstance(val, int):
                raise ValueError("`n_estimators` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_estimators` must be >= 1")
            else:
                self._n_estimators = val

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, val):
        if self._fitted:
            raise ValueError("You cannot modify `n_jobs` after calling `fit`")
        else:
            if not isinstance(val, int):
                raise ValueError("`n_jobs` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_jobs` must be >= 1")
            else:
                self._n_jobs = val

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, val):
        if self._fitted:
            raise ValueError("You cannot modify `step` after calling `fit`")
        else:
            if not isinstance(val, float):
                raise ValueError("`step` must be of type `float`")
            elif val <= 0:
                raise ValueError("`step` must be > 0")
            else:
                self._step = val

    @property
    def use_aggregation(self):
        return self._use_aggregation

    @use_aggregation.setter
    def use_aggregation(self, val):
        if self._fitted:
            raise ValueError("You cannot modify `use_aggregation` after calling `fit`")
        else:
            if not isinstance(val, bool):
                raise ValueError("`use_aggregation` must be of type `bool`")
            else:
                self._use_aggregation = val

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        if self._fitted:
            raise ValueError("You cannot modify `verbose` after calling `fit`")
        else:
            if not isinstance(val, bool):
                raise ValueError("`verbose` must be of type `bool`")
            else:
                self._verbose = val

    @property
    def loss(self):
        return "log"

    @loss.setter
    def loss(self, val):
        pass

    @property
    def random_state(self):
        """:obj:`int` or :obj:`None`: Controls the randomness involved in the trees."""
        if self._random_state == -1:
            return None
        else:
            return self._random_state

    @random_state.setter
    def random_state(self, val):
        if self._fitted:
            raise ValueError("You cannot modify `random_state` after calling `fit`")
        else:
            if val is None:
                self._random_state = -1
            elif not isinstance(val, int):
                raise ValueError("`random_state` must be of type `int`")
            elif val < 0:
                raise ValueError("`random_state` must be >= 0")
            else:
                self._random_state = val

    # TODO: code properties for all arguments with scikit-learn checks
