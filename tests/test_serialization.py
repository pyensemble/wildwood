# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module tests the fact that both wildwood.ForestClassifier and
wildwood.ForestRegressor can be serialized using pickle.
"""


import os
import pickle as pkl
import pytest
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from wildwood.datasets import load_adult, load_boston, load_car
from wildwood import ForestClassifier, ForestRegressor


def assert_features_bitarray_equal(fb1, fb2):
    assert fb1.n_samples == fb2.n_samples
    assert fb1.n_features == fb2.n_features
    np.testing.assert_equal(fb1.max_values, fb2.max_values)
    np.testing.assert_equal(fb1.n_bits, fb2.n_bits)
    np.testing.assert_equal(fb1.offsets, fb2.offsets)
    np.testing.assert_equal(fb1.n_values_in_words, fb2.n_values_in_words)
    np.testing.assert_equal(fb1.bitarray, fb2.bitarray)
    np.testing.assert_equal(fb1.bitmasks, fb2.bitmasks)


def assert_encoder_equal(enc1, enc2):
    assert enc1.max_bins == enc2.max_bins
    assert enc1.subsample == enc2.subsample
    assert enc1.cat_min_categories == enc2.cat_min_categories
    if enc1.is_categorical is None:
        assert enc2.is_categorical is None
    else:
        np.testing.assert_equal(enc1.is_categorical, enc2.is_categorical)
    assert enc1.handle_unknown == enc2.handle_unknown
    assert enc1.random_state == enc2.random_state
    assert enc1.verbose == enc2.verbose
    assert enc1._fitted == enc2._fitted
    assert enc1.n_samples_in_ == enc2.n_samples_in_
    assert enc1.n_features_in_ == enc2.n_features_in_

    categories1 = enc1.categories_
    categories2 = enc2.categories_
    if categories1 is None:
        assert categories2 is None
    else:
        assert len(categories1) == len(categories2)
        assert set(categories1.keys()) == set(categories2.keys())
        for key1 in categories1:
            np.testing.assert_equal(categories1[key1], categories2[key1])

    thresholds1 = enc1.binning_thresholds_
    thresholds2 = enc2.binning_thresholds_
    if thresholds1 is None:
        assert thresholds2 is None
    else:
        assert len(thresholds1) == len(thresholds2)
        assert set(thresholds1.keys()) == set(thresholds2.keys())
        for key1 in thresholds1:
            np.testing.assert_equal(thresholds1[key1], thresholds2[key1])

    has_missing_values1 = enc1.has_missing_values_
    has_missing_values2 = enc2.has_missing_values_
    if has_missing_values1 is None:
        assert has_missing_values2 is None
    else:
        np.testing.assert_equal(has_missing_values1, has_missing_values2)

    n_bins_no_missing_values1 = enc1.n_bins_no_missing_values_
    n_bins_no_missing_values2 = enc2.n_bins_no_missing_values_
    np.testing.assert_equal(n_bins_no_missing_values1, n_bins_no_missing_values2)


def assert_forests_equal(clf1, clf2, is_classifier):
    assert clf1._fitted == clf2._fitted
    assert clf1._n_samples_ == clf2._n_samples_
    assert clf1._n_features_ == clf2._n_features_
    if is_classifier:
        assert clf1._class_weight == clf2._class_weight
    assert clf1.max_features_ == clf2.max_features_
    assert clf1.n_jobs_ == clf2.n_jobs_
    assert clf1._random_states == clf2._random_states
    assert clf1._n_estimators == clf2._n_estimators
    assert clf1._criterion == clf2._criterion
    assert clf1._loss == clf2._loss
    assert clf1._step == clf2._step
    assert clf1._aggregation == clf2._aggregation
    assert clf1._max_depth == clf2._max_depth
    assert clf1._min_samples_split == clf2._min_samples_split
    assert clf1._min_samples_leaf == clf2._min_samples_leaf
    assert clf1.max_bins == clf2.max_bins
    assert clf1._max_bins == clf2._max_bins
    np.testing.assert_equal(clf1.categorical_features, clf2.categorical_features)
    np.testing.assert_equal(clf1.is_categorical_, clf2.is_categorical_)
    assert clf1._max_features == clf2._max_features
    assert clf1._n_jobs == clf2._n_jobs
    assert clf1._random_state == clf2._random_state
    assert clf1._verbose == clf2._verbose
    if is_classifier:
        np.testing.assert_equal(clf1._classes_, clf2._classes_)
        assert clf1._n_classes_ == clf2._n_classes_
        assert clf1.n_classes_per_tree_ == clf2.n_classes_per_tree_
        assert clf1._multiclass == clf2._multiclass
        assert clf1._dirichlet == clf2._dirichlet
        assert clf1._cat_split_strategy == clf2._cat_split_strategy
    np.testing.assert_equal(
        clf1._random_states_bootstrap, clf2._random_states_bootstrap
    )
    np.testing.assert_equal(clf1._random_states_trees, clf2._random_states_trees)

    # Test that encoders are the same
    enc1 = clf1._encoder
    enc2 = clf2._encoder
    assert_encoder_equal(enc1, enc2)

    if is_classifier:
        fb1 = clf1.trees[0]._tree_classifier_context.features_bitarray
        fb2 = clf2.trees[0]._tree_classifier_context.features_bitarray
    else:
        fb1 = clf1.trees[0]._tree_regressor_context.features_bitarray
        fb2 = clf2.trees[0]._tree_regressor_context.features_bitarray

    assert_features_bitarray_equal(fb1, fb2)

    trees1 = clf1.trees
    trees2 = clf2.trees
    assert len(trees1) == len(trees2)

    for tree1, tree2 in zip(trees1, trees2):
        np.testing.assert_equal(tree1._train_indices, tree2._train_indices)
        np.testing.assert_equal(tree1._valid_indices, tree2._valid_indices)
        assert tree1.criterion == tree2.criterion
        assert tree1.loss == tree2.loss
        assert tree1._step == tree2._step
        assert tree1.aggregation == tree2.aggregation
        assert tree1.max_depth == tree2.max_depth
        assert tree1.min_samples_split == tree2.min_samples_split
        assert tree1.min_samples_leaf == tree2.min_samples_leaf
        np.testing.assert_equal(tree1.categorical_features, tree2.categorical_features)
        # assert np.all(tree1.categorical_features == tree2.categorical_features)
        np.testing.assert_equal(tree1.is_categorical, tree2.is_categorical)
        # assert np.all(tree1.is_categorical == tree2.is_categorical)
        assert tree1.max_features == tree2.max_features
        assert tree1.random_state == tree2.random_state
        assert tree1.verbose == tree2.verbose

        if is_classifier:
            assert tree1.n_classes == tree2.n_classes
            assert tree1._dirichlet == tree2._dirichlet
            assert tree1.cat_split_strategy == tree2.cat_split_strategy

        if is_classifier:
            _tree1 = tree1._tree_classifier
            _tree2 = tree2._tree_classifier
        else:
            _tree1 = tree1._tree_regressor
            _tree2 = tree2._tree_regressor

        assert _tree1.n_features == _tree2.n_features
        assert _tree1.random_state == _tree2.random_state
        assert _tree1.max_depth == _tree2.max_depth
        assert _tree1.node_count == _tree2.node_count
        assert _tree1.capacity == _tree2.capacity
        np.testing.assert_equal(_tree1.nodes, _tree2.nodes)
        np.testing.assert_equal(_tree1.bin_partitions, _tree2.bin_partitions)
        assert _tree1.bin_partitions_capacity == _tree2.bin_partitions_capacity
        assert _tree1.bin_partitions_end == _tree2.bin_partitions_end

        assert np.all(_tree1.y_pred == _tree2.y_pred)

        if is_classifier:
            assert _tree1.n_classes == _tree2.n_classes
            _tree_context1 = tree1._tree_classifier_context
            _tree_context2 = tree2._tree_classifier_context
        else:
            _tree_context1 = tree1._tree_regressor_context
            _tree_context2 = tree2._tree_regressor_context

        # TODO: Verifier qu'on ne teste pas plein de fois la meme chose
        np.testing.assert_equal(
            _tree_context1.sample_weights, _tree_context2.sample_weights
        )
        # assert np.all(_tree_context1.sample_weights == _tree_context2.sample_weights)
        np.testing.assert_equal(
            _tree_context1.train_indices, _tree_context2.train_indices
        )
        # assert np.all(_tree_context1.train_indices == _tree_context2.train_indices)
        np.testing.assert_equal(
            _tree_context1.valid_indices, _tree_context2.valid_indices
        )
        # assert np.all(_tree_context1.valid_indices == _tree_context2.valid_indices)
        assert _tree_context1.n_samples == _tree_context2.n_samples
        assert _tree_context1.n_samples_train == _tree_context2.n_samples_train
        assert _tree_context1.n_samples_valid == _tree_context2.n_samples_valid
        assert _tree_context1.n_features == _tree_context2.n_features
        assert _tree_context1.max_features == _tree_context2.max_features
        assert _tree_context1.min_samples_split == _tree_context2.min_samples_split
        assert _tree_context1.min_samples_leaf == _tree_context2.min_samples_leaf
        assert _tree_context1.aggregation == _tree_context2.aggregation
        assert _tree_context1.step == _tree_context2.step

        np.testing.assert_equal(
            _tree_context1.is_categorical, _tree_context2.is_categorical
        )
        # assert np.all(_tree_context1.is_categorical == _tree_context2.is_categorical)
        np.testing.assert_equal(
            _tree_context1.partition_train, _tree_context2.partition_train
        )

        np.testing.assert_equal(
            _tree_context1.partition_valid, _tree_context2.partition_valid
        )
        # assert np.all(_tree_context1.partition_train == _tree_context2.partition_train)
        # assert np.all(_tree_context1.partition_valid == _tree_context2.partition_valid)
        np.testing.assert_equal(_tree_context1.left_buffer, _tree_context2.left_buffer)

        # assert np.all(_tree_context1.left_buffer == _tree_context2.left_buffer)
        np.testing.assert_equal(
            _tree_context1.right_buffer, _tree_context2.right_buffer
        )

        # assert np.all(_tree_context1.right_buffer == _tree_context2.right_buffer)
        assert _tree_context1.criterion == _tree_context2.criterion
        if is_classifier:
            assert _tree_context1.n_classes == _tree_context2.n_classes
            assert (
                _tree_context1.cat_split_strategy == _tree_context2.cat_split_strategy
            )
            assert _tree_context1.dirichlet == _tree_context2.dirichlet


@pytest.mark.parametrize("dataset_name", ["adult", "iris"])
@pytest.mark.parametrize("n_estimators", [2])
@pytest.mark.parametrize("aggregation", (True,))
@pytest.mark.parametrize("class_weight", (None, "balanced"))
@pytest.mark.parametrize("dirichlet", (1e-7,))
@pytest.mark.parametrize("n_jobs", (-1,))
@pytest.mark.parametrize("max_features", ("auto",))
@pytest.mark.parametrize("random_state", (42,))
@pytest.mark.parametrize("step", (1.0,))
@pytest.mark.parametrize("multiclass", ("multinomial", "ovr"))
@pytest.mark.parametrize("cat_split_strategy", ("binary",))
def test_forest_classifier_serialization(
    dataset_name,
    n_estimators,
    aggregation,
    class_weight,
    dirichlet,
    n_jobs,
    max_features,
    random_state,
    step,
    multiclass,
    cat_split_strategy,
):
    if dataset_name == "adult":
        X, y = load_adult(raw=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state
        )
    elif dataset_name == "iris":
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 / 5, random_state=random_state
        )

    clf1 = ForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        multiclass=multiclass,
        max_bins=37,
        cat_split_strategy=cat_split_strategy,
        aggregation=aggregation,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        dirichlet=dirichlet,
        step=step,
    )
    clf1.fit(X_train, y_train)

    filename = "forest_classifier_on_iris.pkl"
    with open(filename, "wb") as f:
        pkl.dump(clf1, f)

    with open(filename, "rb") as f:
        clf2 = pkl.load(f)

    os.remove(filename)

    assert_forests_equal(clf1, clf2, is_classifier=True)

    y_pred1 = clf1.predict_proba(X_test)
    y_pred2 = clf2.predict_proba(X_test)
    np.testing.assert_equal(y_pred1, y_pred2)

    y_pred1 = clf1.predict(X_test)
    y_pred2 = clf2.predict(X_test)
    np.testing.assert_equal(y_pred1, y_pred2)

    apply1 = clf1.apply(X_test)
    apply2 = clf2.apply(X_test)
    np.testing.assert_equal(apply1, apply2)


@pytest.mark.parametrize("n_estimators", [2])
@pytest.mark.parametrize("aggregation", (True,))
@pytest.mark.parametrize("class_weight", (None, "balanced"))
@pytest.mark.parametrize("dirichlet", (1e-7,))
@pytest.mark.parametrize("n_jobs", (-1,))
@pytest.mark.parametrize("max_features", ("auto",))
@pytest.mark.parametrize("random_state", (42,))
@pytest.mark.parametrize("step", (1.0,))
@pytest.mark.parametrize("multiclass", ("multinomial", "ovr"))
@pytest.mark.parametrize("cat_split_strategy", ("binary",))
@pytest.mark.parametrize(
    "dataset_name, one_hot_encode, use_categoricals", [("diabetes", False, False),],
)
def test_forest_regressor_serialization(
    n_estimators,
    aggregation,
    class_weight,
    dirichlet,
    n_jobs,
    max_features,
    random_state,
    step,
    multiclass,
    cat_split_strategy,
    dataset_name,
    one_hot_encode,
    use_categoricals,
):
    if dataset_name == "diabetes":
        iris = datasets.load_diabetes()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 / 5, random_state=random_state
        )

    categorical_features = None

    clf1 = ForestRegressor(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        aggregation=aggregation,
        max_features=max_features,
        categorical_features=categorical_features,
        random_state=random_state,
        step=step,
    )
    clf1.fit(X_train, y_train)

    filename = "forest_classifier_on_iris.pkl"
    with open(filename, "wb") as f:
        pkl.dump(clf1, f)

    with open(filename, "rb") as f:
        clf2 = pkl.load(f)

    os.remove(filename)

    assert_forests_equal(clf1, clf2, is_classifier=False)

    y_pred1 = clf1.predict(X_test)
    y_pred2 = clf2.predict(X_test)
    np.testing.assert_equal(y_pred1, y_pred2)

    apply1 = clf1.apply(X_test)
    apply2 = clf2.apply(X_test)
    np.testing.assert_equal(apply1, apply2)
