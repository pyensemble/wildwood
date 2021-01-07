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

# from ..exceptions import DataConversionWarning
# from ._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, _check_sample_weight


__all__ = ["BinaryClassifier"]

from wildwood._utils import MAX_INT


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
        return _generate_train_valid_samples((random_state + 1) % MAX_INT, n_samples)
    else:
        # print("sample_indices: ", sample_indices)
        # print("sample_counts: ", sample_counts)
        train_mask = np.logical_not(valid_mask)
        train_indices = indices[train_mask]
        valid_indices = indices[valid_mask]
        train_indices_count = sample_counts[train_mask]
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
        sample_weight_full = np.ones((n_samples,), dtype=np.float32)
    else:
        # Do a copy with float32 dtype
        sample_weight_full = sample_weight.astype(np.float32)

    train_indices, valid_indices, train_indices_count = _generate_train_valid_samples(
        tree.random_state, n_samples
    )

    sample_weight_train = sample_weight_full[train_indices]
    sample_weight_valid = sample_weight_full[valid_indices]
    # We use bootstrap: sample repetition is achieved by multiplying the sample
    # weights by the sample counts. By construction, no repetition is possible in
    # validation data
    sample_weight_train *= train_indices_count

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
        train_indices=train_indices,
        valid_indices=valid_indices,
        sample_weight_train=sample_weight_train,
        sample_weight_valid=sample_weight_valid,
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


class ForestBinaryClassifier(object):
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
        max_features="auto",
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.loss = loss
        self.step = step
        self.aggregation = aggregation
        self.dirichlet = dirichlet
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight

        self._fitted = False
        self._n_features = None

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
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # if expanded_class_weight is not None:
        #     if sample_weight is not None:
        #         sample_weight = sample_weight * expanded_class_weight
        #     else:
        #         sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        # Check parameters
        # TODO: prendre les tests la dedans et les mettre dans les properties,
        #  et remettre ici explicitement ce qui manque
        # self._validate_estimator()

        # if not self.bootstrap and self.oob_score:
        #     raise ValueError(
        #         "Out of bag estimation only available" " if bootstrap=True"
        #     )

        random_state = check_random_state(self.random_state)

        # TODO: regerer ce test
        # if not self.warm_start or not hasattr(self, "estimators_"):
        #     # Free allocated memory, if any
        self.estimators_ = []

        # n_more_estimators = self.n_estimators - len(self.estimators_)
        #
        # if n_more_estimators < 0:
        #     raise ValueError(
        #         "n_estimators=%d must be larger or equal to "
        #         "len(estimators_)=%d when warm_start==True"
        #         % (self.n_estimators, len(self.estimators_))
        #     )
        #
        # elif n_more_estimators == 0:
        #     warn(
        #         "Warm-start fitting without increasing n_estimators does not "
        #         "fit new trees."
        #     )
        # else:
        #     if self.warm_start and len(self.estimators_) > 0:
        #         # We draw from the random state to get the random state we
        #         # would have got if we hadn't used a warm_start.
        #         random_state.randint(MAX_INT, size=len(self.estimators_))

        # TODO: instancier ici les arbres a la main sans utiliser les classes de base
        trees = [
            self._make_estimator(append=False, random_state=random_state)
            for i in range(n_estimators())
        ]

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
                t,
                self,
                X,
                y,
                sample_weight,
                i,
                len(trees),
                verbose=self.verbose,
                class_weight=self.class_weight,
                n_samples_bootstrap=n_samples_bootstrap,
            )
            for i, t in enumerate(trees)
        )

        # Collect newly grown trees
        self.estimators_.extend(trees)

        # if self.oob_score:
        #     self._set_oob_score(X, y)

        # TODO: Gerer ces histoires de classes
        # # Decapsulate classes_ attributes
        # if hasattr(self, "classes_") and self.n_outputs_ == 1:
        #     self.n_classes_ = self.n_classes_[0]
        #     self.classes_ = self.classes_[0]

        return self

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
