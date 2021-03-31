# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import numpy as np
import pytest


def approx(v, abs=1e-15):
    return pytest.approx(v, abs)


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from wildwood import ForestClassifier


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
    def test_min_samples_split(self):
        clf = ForestClassifier()
        assert clf.min_samples_split == 2
        clf = ForestClassifier(min_samples_split=17)
        assert clf.min_samples_split == 17
        clf.min_samples_split = 5
        assert clf.min_samples_split == 5
        with pytest.raises(
            ValueError, match="min_samples_split must be an integer number"
        ):
            clf.min_samples_split = 0.42
        with pytest.raises(
            ValueError, match="min_samples_split must be an integer number"
        ):
            clf.min_samples_split = None
        with pytest.raises(
            ValueError, match="min_samples_split must be an integer number"
        ):
            clf.min_samples_split = "4"

        with pytest.raises(ValueError, match="min_samples_split must be >= 2"):
            clf.min_samples_split = 1
        with pytest.raises(ValueError, match="min_samples_split must be >= 2"):
            clf.min_samples_split = -3

    def test_min_samples_leaf(self):
        clf = ForestClassifier()
        assert clf.min_samples_leaf == 1
        clf = ForestClassifier(min_samples_leaf=17)
        assert clf.min_samples_leaf == 17
        clf.min_samples_leaf = 5
        assert clf.min_samples_leaf == 5
        with pytest.raises(
                ValueError, match="min_samples_leaf must be an integer number"
        ):
            clf.min_samples_leaf = 0.42
        with pytest.raises(
                ValueError, match="min_samples_leaf must be an integer number"
        ):
            clf.min_samples_leaf = None
        with pytest.raises(
                ValueError, match="min_samples_leaf must be an integer number"
        ):
            clf.min_samples_leaf = "4"
        with pytest.raises(ValueError, match="min_samples_leaf must be >= 1"):
            clf.min_samples_leaf = 0
        with pytest.raises(ValueError, match="min_samples_leaf must be >= 1"):
            clf.min_samples_leaf = -3

    def test_n_features_(self):
        clf = ForestClassifier(n_estimators=2)
        with pytest.raises(
                ValueError, match="You must call fit before asking for n_features_"
        ):
            _ = clf.n_features_
        np.random.seed(42)
        X = np.random.randn(10, 3)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        clf.fit(X, y)
        assert clf.n_features_ == 3

    def test_n_estimators(self):
        clf = ForestClassifier()
        assert clf.n_estimators == 100
        clf = ForestClassifier(n_estimators=17)
        assert clf.n_estimators == 17
        clf.n_estimators = 42
        assert clf.n_estimators == 42
        with pytest.raises(
                ValueError, match="n_estimators must be an integer number"
        ):
            clf.n_estimators = 0.42
        with pytest.raises(
                ValueError, match="n_estimators must be an integer number"
        ):
            clf.n_estimators = None
        with pytest.raises(
                ValueError, match="n_estimators must be an integer number"
        ):
            clf.n_estimators = "4"
        with pytest.raises(ValueError, match="n_estimators must be >= 1"):
            clf.n_estimators = 0
        with pytest.raises(ValueError, match="n_estimators must be >= 1"):
            clf.n_estimators = -3

        clf = ForestClassifier()
        # Check that  the properties checks the _fitted flag
        clf._fitted = True
        with pytest.raises(
                ValueError, match="You cannot change n_estimators after calling fit"
        ):
            clf.n_estimators = 0.42

    def test_n_jobs(self):
        # TODO: test that n_jobs=1 is slower than n_jobs=4 for instance ? indeed
        # TODO: test that n_jobs=-1 indeed uses all physical cores
        clf = ForestClassifier()
        assert clf.n_jobs == 1
        clf = ForestClassifier(n_jobs=17)
        assert clf.n_jobs == 17
        clf.n_jobs = 42
        assert clf.n_jobs == 42
        with pytest.raises(
                ValueError, match="n_jobs must be an integer number"
        ):
            clf.n_jobs = 0.42
        with pytest.raises(
                ValueError, match="n_jobs must be an integer number"
        ):
            clf.n_jobs = None
        with pytest.raises(
                ValueError, match="n_jobs must be an integer number"
        ):
            clf.n_jobs = "4"
        with pytest.raises(
                ValueError, match="n_jobs must be an integer number"
        ):
            clf.n_jobs = 4.0

        with pytest.raises(ValueError, match="n_jobs must be >= 1 or equal to -1"):
            clf.n_jobs = -2
        with pytest.raises(ValueError, match="n_jobs must be >= 1 or equal to -1"):
            clf.n_jobs = 0

    def test_step(self):
        clf = ForestClassifier()
        assert clf.step == 1.0
        clf = ForestClassifier(step=0.17)
        assert clf.step == 0.17
        clf.step = 0.42
        assert clf.step == 0.42
        # "step must be a real number"
        # "step must be positive"
        with pytest.raises(
                ValueError, match="step must be a float"
        ):
            clf.step = "1"
        with pytest.raises(
                ValueError, match="step must be a float"
        ):
            clf.step = None
        with pytest.raises(
                ValueError, match="step must be positive"
        ):
            clf.step = -1
        with pytest.raises(
                ValueError, match="step must be positive"
        ):
            clf.step = -0.42

    def test_aggregation(self):
        clf = ForestClassifier()
        assert clf.aggregation
        clf = ForestClassifier(aggregation=False)
        assert not clf.aggregation
        clf.aggregation = True
        assert clf.aggregation
        with pytest.raises(
                ValueError, match="aggregation must be boolean"
        ):
            clf.aggregation = "true"
        with pytest.raises(
                ValueError, match="aggregation must be boolean"
        ):
            clf.aggregation = 1

    def test_verbose(self):
        clf = ForestClassifier()
        assert not clf.verbose
        clf = ForestClassifier(verbose=True)
        assert clf.verbose
        clf.verbose = False
        assert not clf.verbose
        with pytest.raises(
                ValueError, match="verbose must be boolean"
        ):
            clf.verbose = "true"
        with pytest.raises(
                ValueError, match="verbose must be boolean"
        ):
            clf.verbose = 1

    def test_loss(self):
        clf = ForestClassifier()
        assert clf.loss == "log"
        with pytest.raises(
                ValueError, match="loss must be a string"
        ):
            clf.loss = 3.14
        with pytest.raises(
                ValueError, match="Only loss='log' is supported for now"
        ):
            clf.loss = "other"

        with pytest.raises(
                ValueError, match="Only loss='log' is supported for now"
        ):
            _ = ForestClassifier(loss="other")

    # TODO: test for random_state



        # @n_estimators.setter
        # def n_estimators(self, val):
        #     if self._fitted:
        #         raise ValueError("You cannot change n_estimators after calling fit")
        #     else:
        #         if not isinstance(val, numbers.Integral):
        #             raise ValueError("n_estimators must an integer number")
        #         elif val < 1:
        #             raise ValueError("n_estimators must be >= 1")
        #         else:
        #             self._n_estimators = val
        # with pytest.raises(ValueError) as exc_info:
        #     clf.min_samples_split = 0.42
        # assert exc_info.type is ValueError
        # assert exc_info.value.args[0] == "min_samples_split must be an integer number"

    # def test_n_features(self):
    #     clf = ForestBinaryClassifier(n_estimators=10)
    #     X = np.random.randn(2, 2)
    #     y = np.array([0.0, 1.0])
    #     clf.fit(X, y)
    #     assert clf.n_features == 2
    #     with pytest.raises(ValueError, match="`n_features` is a readonly attribute"):
    #         clf.n_features = 3

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
