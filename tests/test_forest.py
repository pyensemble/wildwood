# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from wildwood import ForestBinaryClassifier

from .utils import parameter_test_with_min, parameter_test_with_type, approx

#     def test_online_forest_n_features_differs(self):
#         n_samples = 1000
#         n_classes = 2
#         n_trees = 20
#
#         X, y = make_classification(n_samples=n_samples, n_features=10,
#                                    n_redundant=0,
#                                    n_informative=2, random_state=1,
#                                    n_clusters_per_class=1)
#         rng = np.random.RandomState(2)
#         X += 2 * rng.uniform(size=X.shape)
#
#         of = OnlineForestClassifier(n_classes=2, n_trees=n_trees, seed=123,
#                                     step=1.,
#                                     use_aggregation=True)
#
#         of.fit(X, y)
#
#         X, y = make_classification(n_samples=n_samples, n_features=10,
#                                    n_redundant=0,
#                                    n_informative=2, random_state=1,
#                                    n_clusters_per_class=1)
#
#         of.fit(X, y)
#
#         X, y = make_classification(n_samples=n_samples, n_features=3,
#                                    n_redundant=0,
#                                    n_informative=2, random_state=1,
#                                    n_clusters_per_class=1)
#
#     def test_online_forest_n_classes_differs(self):
#         pass

# TODO: parameter_test_with_type does nothing !!!


class TestForestBinaryClassifier(object):

    def test_n_features(self):
        clf = ForestBinaryClassifier(n_estimators=10)
        X = np.random.randn(2, 2)
        y = np.array([0.0, 1.0])
        clf.fit(X, y)
        assert clf.n_features == 2
        with pytest.raises(ValueError, match="`n_features` is a readonly attribute"):
            clf.n_features = 3

    # def test_n_estimators(self):
    #     parameter_test_with_min(
    #         AMFClassifier,
    #         parameter="n_estimators",
    #         valid_val=3,
    #         invalid_type_val=2.0,
    #         invalid_val=0,
    #         min_value=1,
    #         min_value_str="1",
    #         mandatory=False,
    #         fixed_type=int,
    #         required_args={"n_classes": 2},
    #     )
    #
    # def test_step(self):
    #     parameter_test_with_min(
    #         AMFClassifier,
    #         parameter="step",
    #         valid_val=2.0,
    #         invalid_type_val=0,
    #         invalid_val=0.0,
    #         min_value_strict=0.0,
    #         min_value_str="0",
    #         mandatory=False,
    #         fixed_type=float,
    #         required_args={"n_classes": 2},
    #     )
    #
    # def test_loss(self):
    #     amf = AMFClassifier(n_classes=2)
    #     assert amf.loss == "log"
    #     amf.loss = "other loss"
    #     assert amf.loss == "log"
    #
    # def test_use_aggregation(self):
    #     parameter_test_with_type(
    #         AMFClassifier,
    #         parameter="step",
    #         valid_val=False,
    #         invalid_type_val=0,
    #         mandatory=False,
    #         fixed_type=bool,
    #     )
    #
    # def test_dirichlet(self):
    #     parameter_test_with_min(
    #         AMFClassifier,
    #         parameter="dirichlet",
    #         valid_val=0.1,
    #         invalid_type_val=0,
    #         invalid_val=0.0,
    #         min_value_strict=0.0,
    #         min_value_str="0",
    #         mandatory=False,
    #         fixed_type=float,
    #         required_args={"n_classes": 2},
    #     )
    #
    # def test_split_pure(self):
    #     parameter_test_with_type(
    #         AMFClassifier,
    #         parameter="split_pure",
    #         valid_val=False,
    #         invalid_type_val=0,
    #         mandatory=False,
    #         fixed_type=bool,
    #     )
    #
    # def test_random_state(self):
    #     parameter_test_with_min(
    #         AMFClassifier,
    #         parameter="random_state",
    #         valid_val=4,
    #         invalid_type_val=2.0,
    #         invalid_val=-1,
    #         min_value=0,
    #         min_value_str="0",
    #         mandatory=False,
    #         fixed_type=int,
    #         required_args={"n_classes": 2},
    #     )
    #     amf = AMFClassifier(n_classes=2)
    #     assert amf.random_state is None
    #     assert amf._random_state == -1
    #     amf.random_state = 1
    #     amf.random_state = None
    #     assert amf._random_state == -1
    #
    # def test_n_jobs(self):
    #     parameter_test_with_min(
    #         AMFClassifier,
    #         parameter="n_jobs",
    #         valid_val=4,
    #         invalid_type_val=2.0,
    #         invalid_val=0,
    #         min_value=1,
    #         min_value_str="1",
    #         mandatory=False,
    #         fixed_type=int,
    #         required_args={"n_classes": 2},
    #     )
    #
    # def test_n_samples_increment(self):
    #     parameter_test_with_min(
    #         AMFClassifier,
    #         parameter="n_samples_increment",
    #         valid_val=128,
    #         invalid_type_val=2.0,
    #         invalid_val=0,
    #         min_value=1,
    #         min_value_str="1",
    #         mandatory=False,
    #         fixed_type=int,
    #         required_args={"n_classes": 2},
    #     )
    #
    # def test_verbose(self):
    #     parameter_test_with_type(
    #         AMFClassifier,
    #         parameter="verbose",
    #         valid_val=False,
    #         invalid_type_val=0,
    #         mandatory=False,
    #         fixed_type=bool,
    #     )
    #
    # def test_repr(self):
    #     amf = AMFClassifier(n_classes=3)
    #     assert (
    #         repr(amf) == "AMFClassifier(n_classes=3, n_estimators=10, "
    #         "step=1.0, loss='log', use_aggregation=True, "
    #         "dirichlet=0.01, split_pure=False, n_jobs=1, "
    #         "random_state=None, verbose=False)"
    #     )
    #
    #     amf.n_estimators = 42
    #     assert (
    #         repr(amf) == "AMFClassifier(n_classes=3, n_estimators=42, "
    #         "step=1.0, loss='log', use_aggregation=True, "
    #         "dirichlet=0.01, split_pure=False, n_jobs=1, "
    #         "random_state=None, verbose=False)"
    #     )
    #
    #     amf.verbose = False
    #     assert (
    #         repr(amf) == "AMFClassifier(n_classes=3, n_estimators=42, "
    #         "step=1.0, loss='log', use_aggregation=True, "
    #         "dirichlet=0.01, split_pure=False, n_jobs=1, "
    #         "random_state=None, verbose=False)"
    #     )
    #
    # def test_partial_fit(self):
    #     clf = AMFClassifier(n_classes=2)
    #     n_features = 4
    #     X = np.random.randn(2, n_features)
    #     y = np.array([0.0, 1.0])
    #     clf.partial_fit(X, y)
    #     assert clf.n_features == n_features
    #     assert clf.no_python.iteration == 2
    #     assert clf.no_python.samples.n_samples == 2
    #     assert clf.no_python.n_features == n_features
    #
    #     with pytest.raises(ValueError) as exc_info:
    #         X = np.random.randn(2, 3)
    #         y = np.array([0.0, 1.0])
    #         clf.partial_fit(X, y)
    #     assert exc_info.type is ValueError
    #     assert (
    #         exc_info.value.args[0] == "`partial_fit` was first called with "
    #         "n_features=4 while n_features=3 in this call"
    #     )
    #
    #     with pytest.raises(
    #         ValueError, match="All the values in `y` must be non-negative",
    #     ):
    #         clf = AMFClassifier(n_classes=2)
    #         X = np.random.randn(2, n_features)
    #         y = np.array([0.0, -1.0])
    #         clf.partial_fit(X, y)
    #
    #     with pytest.raises(ValueError) as exc_info:
    #         clf = AMFClassifier(n_classes=2)
    #         X = np.random.randn(2, 3)
    #         y = np.array([0.0, 2.0])
    #         clf.partial_fit(X, y)
    #     assert exc_info.type is ValueError
    #     assert exc_info.value.args[0] == "n_classes=2 while y.max()=2"
    #
    # def test_predict_proba(self):
    #     clf = AMFClassifier(n_classes=2)
    #     with pytest.raises(
    #         RuntimeError,
    #         match="You must call `partial_fit` before asking for predictions",
    #     ):
    #         X_test = np.random.randn(2, 3)
    #         clf.predict_proba(X_test)
    #
    #     with pytest.raises(ValueError) as exc_info:
    #         X = np.random.randn(2, 2)
    #         y = np.array([0.0, 1.0])
    #         clf.partial_fit(X, y)
    #         X_test = np.random.randn(2, 3)
    #         clf.predict_proba(X_test)
    #     assert exc_info.type is ValueError
    #     assert exc_info.value.args[
    #         0
    #     ] == "`partial_fit` was called with n_features=%d while predictions are asked with n_features=%d" % (
    #         clf.n_features,
    #         3,
    #     )
    #
    # def test_performance_on_moons(self):
    #     n_samples = 300
    #     random_state = 42
    #     X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.5, random_state=random_state
    #     )
    #     clf = AMFClassifier(n_classes=2, random_state=random_state)
    #     clf.partial_fit(X_train, y_train)
    #     y_pred = clf.predict_proba(X_test)
    #     score = roc_auc_score(y_test, y_pred[:, 1])
    #     # With this random_state, the score should be exactly 0.9709821428571429
    #     assert score > 0.97
    #
    # def test_predict_proba_tree_match_predict_proba(self):
    #     n_samples = 300
    #     n_classes = 2
    #     n_estimators = 10
    #     random_state = 42
    #     X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.5, random_state=random_state
    #     )
    #     clf = AMFClassifier(
    #         n_classes=2, n_estimators=n_estimators, random_state=random_state
    #     )
    #     clf.partial_fit(X_train, y_train)
    #     y_pred = clf.predict_proba(X_test)
    #     y_pred_tree = np.empty((y_pred.shape[0], n_classes, n_estimators))
    #     for idx_tree in range(n_estimators):
    #         y_pred_tree[:, :, idx_tree] = clf.predict_proba_tree(X_test, idx_tree)
    #
    #     assert y_pred == approx(y_pred_tree.mean(axis=2), 1e-6)
    #
    # def test_random_state_is_consistant(self):
    #     n_samples = 300
    #     random_state = 42
    #     X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=random_state)
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.5, random_state=random_state
    #     )
    #
    #     clf = AMFClassifier(n_classes=2, random_state=random_state)
    #     clf.partial_fit(X_train, y_train)
    #     y_pred_1 = clf.predict_proba(X_test)
    #
    #     clf = AMFClassifier(n_classes=2, random_state=random_state)
    #     clf.partial_fit(X_train, y_train)
    #     y_pred_2 = clf.predict_proba(X_test)
    #
    #     assert y_pred_1 == approx(y_pred_2)
