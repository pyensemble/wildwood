# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import types
from itertools import product

import numpy as np
import pytest


def approx(v, abs=1e-15):
    return pytest.approx(v, abs)


from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from wildwood import ForestClassifier


# TODO: parameter_test_with_type does nothing !!!


class TestForestBinaryClassifier(object):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.iris = load_iris()

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
        with pytest.raises(ValueError, match="n_estimators must be an integer number"):
            clf.n_estimators = 0.42
        with pytest.raises(ValueError, match="n_estimators must be an integer number"):
            clf.n_estimators = None
        with pytest.raises(ValueError, match="n_estimators must be an integer number"):
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
        with pytest.raises(ValueError, match="n_jobs must be an integer number"):
            clf.n_jobs = 0.42
        with pytest.raises(ValueError, match="n_jobs must be an integer number"):
            clf.n_jobs = None
        with pytest.raises(ValueError, match="n_jobs must be an integer number"):
            clf.n_jobs = "4"
        with pytest.raises(ValueError, match="n_jobs must be an integer number"):
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
        with pytest.raises(ValueError, match="step must be a float"):
            clf.step = "1"
        with pytest.raises(ValueError, match="step must be a float"):
            clf.step = 1
        with pytest.raises(ValueError, match="step must be a float"):
            clf.step = None
        with pytest.raises(ValueError, match="step must be positive"):
            clf.step = -1.0
        with pytest.raises(ValueError, match="step must be positive"):
            clf.step = -0.42

    def test_aggregation(self):
        clf = ForestClassifier()
        assert clf.aggregation
        clf = ForestClassifier(aggregation=False)
        assert not clf.aggregation
        clf.aggregation = True
        assert clf.aggregation
        with pytest.raises(ValueError, match="aggregation must be boolean"):
            clf.aggregation = "true"
        with pytest.raises(ValueError, match="aggregation must be boolean"):
            clf.aggregation = 1

    def test_verbose(self):
        clf = ForestClassifier()
        assert not clf.verbose
        clf = ForestClassifier(verbose=True)
        assert clf.verbose
        clf.verbose = False
        assert not clf.verbose
        with pytest.raises(ValueError, match="verbose must be boolean"):
            clf.verbose = "true"
        with pytest.raises(ValueError, match="verbose must be boolean"):
            clf.verbose = 1

    def test_loss(self):
        clf = ForestClassifier()
        assert clf.loss == "log"
        with pytest.raises(ValueError, match="loss must be a string"):
            clf.loss = 3.14
        with pytest.raises(ValueError, match="Only loss='log' is supported for now"):
            clf.loss = "other"

        with pytest.raises(ValueError, match="Only loss='log' is supported for now"):
            _ = ForestClassifier(loss="other")

    # TODO: test for random_state

    def test_random_state(self):
        iris = self.iris
        X, y = iris["data"], iris["target"]

        def do_test_bootstrap(n_estimators, n_jobs, random_state):
            # 1. Test that all bootstrap samples are different
            clf = ForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state
            )
            clf.fit(X, y)

            for n_estimator1, n_estimator2 in product(
                range(n_estimators), range(n_estimators)
            ):
                if n_estimator1 < n_estimator2:
                    assert clf.trees[n_estimator1]._train_indices != approx(
                        clf.trees[n_estimator2]._train_indices
                    )
                    assert clf.trees[n_estimator1]._valid_indices != approx(
                        clf.trees[n_estimator2]._valid_indices
                    )

            # 2. Test that random_seed makes bootstrap samples identical and that
            #    when no random_seed is used bootstrap samples are different
            clf1 = ForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state
            )
            clf1.fit(X, y)
            clf2 = ForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state
            )
            clf2.fit(X, y)
            for n_estimator in range(n_estimators):
                if random_state is None:
                    assert clf1.trees[n_estimator]._train_indices != approx(
                        clf2.trees[n_estimator]._train_indices
                    )
                    assert clf1.trees[n_estimator]._valid_indices != approx(
                        clf2.trees[n_estimator]._valid_indices
                    )
                else:
                    assert clf1.trees[n_estimator]._train_indices == approx(
                        clf2.trees[n_estimator]._train_indices
                    )
                    assert clf1.trees[n_estimator]._valid_indices == approx(
                        clf2.trees[n_estimator]._valid_indices
                    )

            # 3. Test that the apply() method gives the exact same leaves (this allows
            #    to check that the trees are the same, namely that random columns
            #    subsampling is indeed correctly seeded) and that predictions are the
            #    same (or not)
            clf1 = ForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state
            )
            clf1.fit(X, y)
            clf2 = ForestClassifier(
                n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state
            )
            clf2.fit(X, y)
            if random_state is None:
                assert clf1.apply(X) != approx(clf2.apply(X))
                assert clf1.predict_proba(X) != approx(clf2.predict_proba(X))
            else:
                assert clf1.apply(X) == approx(clf2.apply(X))
                assert clf1.predict_proba(X) == approx(clf2.predict_proba(X))

        # Let's try out tests 1. 2. and 3. on a combination of parameters
        n_estimatorss = [1, 3, 9]
        n_jobss = [1, 4, -1]
        random_states = [None, 42, 0]
        for n_estimators, n_jobs, random_state in product(
            n_estimatorss, n_jobss, random_states
        ):
            do_test_bootstrap(n_estimators, n_jobs, random_state)

        def do_test_bootstrap_again(n_estimators, n_jobs):
            # 4. When bootstrap seeds and column subsampling seeds are the same,
            #    the trees are all the same
            clf = ForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

            def _my_generate_random_states(self):
                self._random_states_bootstrap = np.ones(
                    (clf.n_estimators), dtype=np.int32
                )
                self._random_states_trees = np.ones((clf.n_estimators), dtype=np.int32)

            # Monkey patch the classifier
            clf._generate_random_states = types.MethodType(
                _my_generate_random_states, clf
            )
            clf.fit(X, y)
            leaves = clf.apply(X)
            for n_estimator1, n_estimator2 in product(
                range(n_estimators), range(n_estimators)
            ):
                if n_estimator1 < n_estimator2:
                    assert clf.trees[n_estimator1]._train_indices == approx(
                        clf.trees[n_estimator2]._train_indices
                    )
                    assert clf.trees[n_estimator1]._valid_indices == approx(
                        clf.trees[n_estimator2]._valid_indices
                    )
                    assert leaves[n_estimator1] == approx(leaves[n_estimator2])

            # 5. When bootstrap seeds are the same but column subsampling seeds are
            #    different, all the trees are different
            clf = ForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

            def _my_generate_random_states(self):
                # All bootstrap seeds are the same
                self._random_states_bootstrap = np.ones(
                    (clf.n_estimators), dtype=np.int32
                )
                # But column subsampling seeds are different
                self._random_states_trees = np.arange(clf.n_estimators, dtype=np.int32)

            # Monkey patch the classifier
            clf._generate_random_states = types.MethodType(
                _my_generate_random_states, clf
            )
            clf.fit(X, y)
            leaves = clf.apply(X)
            for n_estimator1, n_estimator2 in product(
                range(n_estimators), range(n_estimators)
            ):
                if n_estimator1 < n_estimator2:
                    assert clf.trees[n_estimator1]._train_indices == approx(
                        clf.trees[n_estimator2]._train_indices
                    )
                    assert clf.trees[n_estimator1]._valid_indices == approx(
                        clf.trees[n_estimator2]._valid_indices
                    )
                    assert leaves[n_estimator1] != approx(leaves[n_estimator2])

            # 6. When bootstrap seeds are different but column subsampling seeds are
            #    identical, all the trees are different
            clf = ForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

            def _my_generate_random_states(self):
                # All bootstrap seeds are the same
                self._random_states_bootstrap = np.arange(
                    clf.n_estimators, dtype=np.int32
                )
                # But column subsampling seeds are different
                self._random_states_trees = np.ones((clf.n_estimators,), dtype=np.int32)

            # Monkey patch the classifier
            clf._generate_random_states = types.MethodType(
                _my_generate_random_states, clf
            )
            clf.fit(X, y)
            leaves = clf.apply(X)
            for n_estimator1, n_estimator2 in product(
                range(n_estimators), range(n_estimators)
            ):
                if n_estimator1 < n_estimator2:
                    assert clf.trees[n_estimator1]._train_indices != approx(
                        clf.trees[n_estimator2]._train_indices
                    )
                    assert clf.trees[n_estimator1]._valid_indices != approx(
                        clf.trees[n_estimator2]._valid_indices
                    )
                    assert leaves[n_estimator1] != approx(leaves[n_estimator2])

        # Now we test 4. 5. and 6.
        n_estimatorss = [1, 3]
        n_jobss = [1, 4, -1]
        for n_estimators, n_jobs in product(
            n_estimatorss, n_jobss
        ):
            do_test_bootstrap_again(n_estimators, n_jobs)
