# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# py.test -rA

import types
from itertools import product
from time import time

import numpy as np
import pytest

from sklearn.utils import compute_sample_weight
from sklearn.metrics import classification_report, average_precision_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelBinarizer

from joblib import effective_n_jobs
from time import time
import logging


def approx(v, abs=1e-15):
    return pytest.approx(v, abs=abs)


from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from wildwood import ForestClassifier, ForestRegressor
from wildwood.dataset import load_car


# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
# )

# TODO: parameter_test_with_type does nothing !!!


class TestForestClassifier(object):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.iris = load_iris()
        self.breast_cancer = load_breast_cancer()
        self.effective_n_jobs = effective_n_jobs()

        # logging.info(
        #     "%d jobs can be effectively be run in parallel on this machine"
        #     % self.effective_n_jobs
        # )

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

            def _my_generate_random_states(self, n_states=None):
                self._random_states_bootstrap = np.ones(
                    (n_states or clf.n_estimators), dtype=np.int32
                )
                self._random_states_trees = np.ones(
                    (n_states or clf.n_estimators), dtype=np.int32
                )

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

            def _my_generate_random_states(self, n_states=None):
                # All bootstrap seeds are the same
                self._random_states_bootstrap = np.ones(
                    (n_states or clf.n_estimators), dtype=np.int32
                )
                # But column subsampling seeds are different
                self._random_states_trees = np.arange(
                    n_states or clf.n_estimators, dtype=np.int32
                )

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

            def _my_generate_random_states(self, n_states=None):
                # All bootstrap seeds are the same
                self._random_states_bootstrap = np.arange(
                    n_states or clf.n_estimators, dtype=np.int32
                )
                # But column subsampling seeds are different
                self._random_states_trees = np.ones(
                    (n_states or clf.n_estimators,), dtype=np.int32
                )

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
        for n_estimators, n_jobs in product(n_estimatorss, n_jobss):
            do_test_bootstrap_again(n_estimators, n_jobs)

    def test_class_weight(self):
        # Test that default if None
        clf = ForestClassifier()
        assert clf.class_weight is None
        clf.class_weight = "balanced"
        assert clf.class_weight == "balanced"
        with pytest.raises(
            ValueError, match='class_weight can only be None or "balanced"'
        ):
            clf.class_weight = "truc"
        with pytest.raises(
            ValueError, match='class_weight can only be None or "balanced"'
        ):
            clf.class_weight = 1

    def test_n_classes_classes_n_features_n_samples(self):
        y = ["one", "two", "three", "one", "one", "two"]
        X = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        clf = ForestClassifier()
        clf.fit(X, y)

        assert tuple(clf.classes_) == ("one", "three", "two")
        assert clf.n_classes_ == 3
        assert clf.n_features_ == 2
        assert clf.n_samples_ == 6

    # TODO: test that the label is 1D

    def test_class_weight_sample_weights(self):
        iris = self.iris
        X, y = iris["data"], iris["target"]
        # Check that no sample_weight and all sample weights equal to 1. is the same
        clf1 = ForestClassifier(class_weight=None, random_state=42)
        clf1.fit(X, y)
        clf2 = ForestClassifier(class_weight=None, random_state=42)
        clf2.fit(X, y, sample_weight=np.ones(y.shape[0]))
        assert clf1.apply(X) == approx(clf2.apply(X))
        assert clf1.predict_proba(X) == approx(clf2.predict_proba(X))

        clf1 = ForestClassifier(class_weight="balanced", random_state=42)
        clf1.fit(X, y)
        clf2 = ForestClassifier(class_weight=None, random_state=42)
        sample_weight = compute_sample_weight("balanced", y)
        clf2.fit(X, y, sample_weight=sample_weight)
        assert clf1.apply(X) == approx(clf2.apply(X))
        assert clf1.predict_proba(X) == approx(clf2.predict_proba(X))

        # Simulate unbalanced data from the iris dataset
        X_unb = np.concatenate((X[0:50], X[50:56], X[100:106]), axis=0)
        y_unb = np.concatenate((y[0:50], y[50:56], y[100:106]), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X_unb, y_unb, shuffle=True, stratify=y_unb, random_state=42, test_size=0.5
        )

        clf = ForestClassifier(class_weight=None, random_state=42, aggregation=True)
        clf.fit(X_train, y_train)
        y_scores = clf.predict(X_test)
        report1 = classification_report(y_test, y_scores, output_dict=True)

        clf = ForestClassifier(
            class_weight="balanced", random_state=42, aggregation=True
        )
        clf.fit(X_train, y_train)
        y_scores = clf.predict(X_test)
        report2 = classification_report(y_test, y_scores, output_dict=True)

        # In the considered case, class_weight should improve all metrics
        for label in ["0", "1", "2"]:
            label_report1 = report1[label]
            label_report2 = report2[label]
            assert label_report2["precision"] >= label_report1["precision"]
            assert label_report2["recall"] >= label_report1["recall"]
            assert label_report2["f1-score"] >= label_report1["f1-score"]

        breast_cancer = self.breast_cancer
        X, y = breast_cancer["data"], breast_cancer["target"]
        idx_0 = y == 0
        idx_1 = y == 1

        X_unb = np.concatenate((X[idx_0], X[idx_1][:10]), axis=0)
        y_unb = np.concatenate((y[idx_0], y[idx_1][:10]), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X_unb, y_unb, shuffle=True, stratify=y_unb, random_state=42, test_size=0.5
        )

        clf = ForestClassifier(class_weight=None, random_state=42, aggregation=True)
        clf.fit(X_train, y_train)
        y_scores = clf.predict(X_test)

        y_test_binary = LabelBinarizer().fit_transform(y_test)

        avg_prec1 = average_precision_score(y_test_binary, y_scores, average="weighted")

        clf = ForestClassifier(
            class_weight="balanced", random_state=42, aggregation=True
        )
        clf.fit(X_train, y_train)
        y_scores = clf.predict(X_test)
        avg_prec2 = average_precision_score(y_test_binary, y_scores, average="weighted")

        assert avg_prec2 > avg_prec1

    def test_performance_iris(self):
        iris = self.iris
        X, y = iris["data"], iris["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=True, stratify=y, random_state=42, test_size=0.3
        )
        clf = ForestClassifier(class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        assert roc_auc_score(y_test, y_score, multi_class="ovo") >= 0.985
        clf = ForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        assert roc_auc_score(y_test, y_score, multi_class="ovo") >= 0.985

    def test_performance_breast_cancer(self):
        breast_cancer = self.breast_cancer
        X, y = breast_cancer["data"], breast_cancer["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=True, stratify=y, random_state=42, test_size=0.3
        )
        clf = ForestClassifier(class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        assert roc_auc_score(y_test, y_score[:, 1]) >= 0.98
        clf = ForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        assert roc_auc_score(y_test, y_score[:, 1]) >= 0.98

    def test_performance_moons(self):
        pass

    def test_parallel_fit(self):
        n_samples = 100_000
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)

        # Precompile
        clf = ForestClassifier(n_estimators=1, n_jobs=1, aggregation=True)
        clf.fit(X[:10], y[:10])
        clf = ForestClassifier(n_estimators=1, n_jobs=1, aggregation=False)
        clf.fit(X[:10], y[:10])

        random_state = 42

        effective_n_jobs = self.effective_n_jobs
        print("effective_n_jobs: ", effective_n_jobs)

        def is_parallel_split_faster(n_estimators, aggregation):
            clf = ForestClassifier(
                random_state=random_state,
                n_estimators=n_estimators,
                n_jobs=1,
                aggregation=aggregation,
            )
            tic = time()
            clf.fit(X, y)
            toc = time()
            time_no_parallel = toc - tic

            clf = ForestClassifier(
                random_state=random_state,
                n_estimators=n_estimators,
                n_jobs=effective_n_jobs,
                aggregation=aggregation,
            )

            tic = time()
            clf.fit(X, y)
            toc = time()
            time_parallel = toc - tic

            # We want parallel code to be effective_n_jobs / 3 faster when using
            # effectively effective_n_jobs threads
            assert time_no_parallel >= effective_n_jobs * time_parallel / 4

        # We want each thread to handle 4 trees
        n_estimators = 4 * effective_n_jobs
        is_parallel_split_faster(n_estimators=n_estimators, aggregation=True)
        is_parallel_split_faster(n_estimators=n_estimators, aggregation=False)

    def test_parallel_predict_proba(self):
        # TODO: predict_proba in parallel is sloooooowwwww ????
        pass

    def test_categorical_features(self):
        clf = ForestClassifier()
        assert clf.categorical_features is None
        clf.categorical_features = [1, 3]
        assert clf.categorical_features == [1, 3]

    def test_multiclass_vs_ovr_on_car(self):
        dataset = load_car()
        dataset.one_hot_encode = False
        random_state = 42
        X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
        y_test_binary = LabelBinarizer().fit_transform(y_test)

        clf = ForestClassifier(
            n_estimators=10,
            n_jobs=-1,
            multiclass="multinomial",
            categorical_features=dataset.categorical_features_,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_test)
        avg_prec1 = average_precision_score(y_test_binary, y_scores, average="weighted")

        clf = ForestClassifier(
            n_estimators=10,
            n_jobs=-1,
            multiclass="ovr",
            categorical_features=dataset.categorical_features_,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_test)
        avg_prec2 = average_precision_score(y_test_binary, y_scores, average="weighted")

        clf = ForestClassifier(
            n_estimators=10,
            n_jobs=-1,
            multiclass="ovr",
            categorical_features=dataset.categorical_features_,
            random_state=random_state,
        )

        assert avg_prec2 > avg_prec1 + 1


    def test_categorical_features_performance(self):
        pass

        # inspired from
        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-categorical-py
        #  Ames Housing dataset¶

        # from sklearn.datasets import fetch_openml
        # from sklearn.pipeline import make_pipeline
        # from sklearn.compose import make_column_transformer, make_column_selector
        # from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
        #
        # X, y = fetch_openml(data_id=41211, as_frame=True, return_X_y=True)
        # X_train = X[:100]
        # y_train = y[:100]
        # X_test = X[100:120]
        # y_test = y[100:120]
        #
        # # 1 - drop categorical features
        # dropper = make_column_transformer(
        #     ('drop', make_column_selector(dtype_include='category')),
        #     remainder='passthrough')
        # X_train_drop = dropper.fit_transform(X_train, y_train)
        # X_test_drop = dropper.fit_transform(X_test, y_test)
        # clf_dropped = ForestRegressor(random_state=42)
        # clf_dropped.fit(X_train_drop, y_train)
        # y_pred_drop = clf_dropped.predict(X_test_drop)
        #
        # # 2 - one-hot encoding on categorical features
        # one_hot_encoder = make_column_transformer(
        #     (OneHotEncoder(sparse=False, handle_unknown='ignore'),
        #      make_column_selector(dtype_include='category')),
        #     remainder='passthrough')
        #
        # hist_one_hot = make_pipeline(one_hot_encoder,
        #                              ForestClassifier(random_state=42))
        # # 3 - ordinal encoder
        # ordinal_encoder = make_column_transformer(
        #     (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan),
        #      make_column_selector(dtype_include='category')),
        #     remainder='passthrough')
        #
        # hist_ordinal = make_pipeline(ordinal_encoder,
        #                              ForestClassifier(random_state=42))
        #
        # # 4 - use categorical support of Forest Classifier
        # categorical_features = np.where(X.dtypes == 'category')[0]
        # hist_native = make_pipeline(
        #     ordinal_encoder,
        #     ForestClassifier(random_state=42,
        #                      categorical_features=categorical_features)
        # )

    # TODO: faire un test pour compute_split_partition en s'inspirant de ce qui est
    #  en dessous

    # @jit(
    #     [void(SplitClassifierType), void(SplitRegressorType)], nopython=True, nogil=True,
    # )
    # def init_split(split):
    #     """A common initializer for SplitClassifier and SplitRegressor.
    #     Since numba's jitclass support for inheritance is inexistant, we use this
    #     function both in the constructors of SplitClassifier and SplitRegressor.
    #
    #     Parameters
    #     ----------
    #     split : SplitClassifier or SplitRegressor
    #         The split to be initialized by this function
    #     """
    #     split.found_split = False
    #     split.gain_proxy = -np.inf
    #     split.feature = 0
    #     split.bin_threshold = 0
    #     split.w_samples_train_left = 0.0
    #     split.w_samples_train_right = 0.0
    #     split.w_samples_valid_left = 0.0
    #     split.w_samples_valid_right = 0.0
    #     split.impurity_left = 0.0
    #     split.impurity_right = 0.0
    #     split.is_split_categorical = False
    #     split.order_bins = np.empty(256, dtype=np.uint8)
    #     split.split_partition = np.empty(128, dtype=np.uint8)
    #     split.split_partition_size = 0
    #
    #
    # @jit(
    #     [void(uint8, SplitClassifierType), void(uint8, SplitRegressorType)],
    #     nopython=True,
    #     nogil=True,
    #     locals={
    #         "n_bins_non_missing": uint8,
    #         "bin_threshold": uint8,
    #         "order_bins": uint8[::1],
    #         "split_partition": uint8[::1],
    #         "idx_bin": uint8,
    #         "bin": uint8,
    #         "left_partition_size": uint8,
    #         "right_partition_size": uint8,
    #     },
    # )
    # def compute_split_partition(n_bins_non_missing, best_split):
    #     """
    #
    #     Parameters
    #     ----------
    #     tree_context : TreeClassifierContext or TreeRegressorContext
    #         The tree context which contains all the data about the tree that is useful to
    #         find a split
    #
    #     best_split : SplitClassifier or SplitRegressor
    #         The best split found for which we want to compute the correct split_partition
    #         and  split_partition_size attributes
    #
    #     Example
    #     -------
    #     Let us consider the following examples where we have a categorical features with
    #     bins [0, 1, 2, 3, 4] and let's say that order_bins is [3, 0, 4, 2, 1].
    #
    #     * Example 1. If bin_threshold == 2 then the partition is {[3, 0, 4, 2], [1]}.
    #       In this case split_partition = [1] and split_partition_size = 1
    #
    #     * Example 2. If bin_threshold == 3 then the partition is {[3], [0, 4, 2, 1]}
    #       In this case split_partition = [3] and split_partition_size = 1
    #
    #     * Example 3. If bin_threshold == 4 then the partition is {[3, 0, 4], [2, 1]}
    #       In this case split_partition = [1, 2] and split_partition_size = 2
    #     """
    #     # Bin threshold used by the best split
    #     bin_threshold = best_split.bin_threshold
    #     # Ordering of the bins used in the split
    #     order_bins = best_split.order_bins
    #     # The array in which we save the split partition
    #     split_partition = best_split.split_partition
    #     idx_bin = 0
    #     bin_ = order_bins[idx_bin]
    #     left_partition_size = 1
    #     # TODO: we have a problem if n_bins_non_missing == 1. Test that the Binner raises
    #     #  an error in such a case
    #     right_partition_size = n_bins_non_missing - 1
    #     while bin_ != bin_threshold:
    #         idx_bin += 1
    #         left_partition_size += 1
    #         right_partition_size -= 1
    #         bin_ = order_bins[idx_bin]
    #
    #     if left_partition_size < right_partition_size:
    #         split_partition[:left_partition_size] = np.sort(
    #             order_bins[:left_partition_size]
    #         )
    #         best_split.split_partition_size = left_partition_size
    #     else:
    #         split_partition[:right_partition_size] = np.sort(
    #             order_bins[left_partition_size:n_bins_non_missing]
    #         )
    #         best_split.split_partition_size = right_partition_size
    #
    #
    # @jit(nopython=True, nogil=True)
    # def main():
    #     n_classes = 3
    #     n_bins_non_missing = 5
    #     best_split = SplitClassifier(n_classes)
    #     best_split.order_bins[:n_bins_non_missing] = np.array(
    #         [3, 0, 4, 2, 1], dtype=np.uint8
    #     )
    #     best_split.bin_threshold = 2
    #
    #     compute_split_partition(n_bins_non_missing, best_split)
    #
    #     print("best_split.split_partition:", best_split.split_partition)
    #     print("best_split.split_partition_size:", best_split.split_partition_size)
    #     print(
    #         "split_partition: ",
    #         best_split.split_partition[: best_split.split_partition_size],
    #     )
    #
    #     # Let us consider the following example where we have a categorical features
    #     # with bins [0, 1, 2, 3, 4] and let's say that order_bins is [3, 0, 4, 2, 1].
    #     # If bin_threshold == 2 then the partition is [3, 0, 4, 2], [1]}
    #     # If bin_threshold == 3 then the partition is {[3], [0, 4, 2, 1]}
    #     # If bin_threshold == 4 then the partition is {[3, 0, 4], [2, 1]}
    #
    #
    # main()
