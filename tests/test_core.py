# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module performs unittests on some core computations involved in WildWood
"""

import pytest
import numpy as np

from wildwood import ForestClassifier, ForestRegressor
from wildwood.datasets import load_adult, load_boston, load_car

from wildwood._split import is_bin_in_partition


@pytest.mark.parametrize(
    "bin_partition",
    [
        np.array([3], dtype=np.uint8),
        np.array([3, 4], dtype=np.uint8),
        np.array([3, 4, 8], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize(
    "bin_", range(13),
)
def test_is_bin_in_partition(bin_partition, bin_):
    in_partition = bin_ in bin_partition
    assert is_bin_in_partition(bin_, bin_partition) == in_partition


def check_nodes(nodes, bin_partitions, aggregation):
    node_count = nodes.size
    assert node_count > 0
    for node_id, node in enumerate(nodes):
        # Check that node_id is indeed its index in the array
        assert node_id == node["node_id"]
        # Check that nodes contain both training and validation samples
        assert node["n_samples_train"] >= 1
        assert node["w_samples_train"] > 0.0
        if aggregation:
            assert node["n_samples_valid"] >= 1
            assert node["w_samples_valid"] > 0.0

        # Check start_train, end_train, start_valid and end_valid
        start_train = node["start_train"]
        end_train = node["end_train"]
        start_valid = node["start_valid"]
        end_valid = node["end_valid"]
        assert start_train < end_train
        if aggregation:
            assert start_valid < end_valid

        parent = node["parent"]
        if node_id != 0:
            # Check that node_id of the node is larger than the one of its parent
            assert node_id > nodes[parent]["node_id"]
            # Check that start_train, end_train, start_valid and end_valid are
            # included in those of the parent
            assert start_train >= nodes[parent]["start_train"]
            assert end_train <= nodes[parent]["end_train"]
            if aggregation:
                assert start_valid >= nodes[parent]["start_valid"]
                assert end_valid <= nodes[parent]["end_valid"]
            # Check that depth of a child is +1 the one of its parent
            assert node["depth"] == nodes[parent]["depth"] + 1
            # TODO: Check that leaves have no child

        # Check that categorical splits have non-empty bin_partition
        if node["is_split_categorical"]:
            bin_partition_start = node["bin_partition_start"]
            bin_partition_end = node["bin_partition_end"]
            assert bin_partition_start < bin_partition_end
            bin_partition = bin_partitions[bin_partition_start:bin_partition_end]
            assert bin_partition.size >= 1

        if_leaf = node["is_leaf"]
        # Check that no non-leaf node is pure
        if not if_leaf:
            assert node["impurity"] > 0.0

        # Check that a pure node is a leaf
        if node["impurity"] == 0:
            assert if_leaf

        if not if_leaf:
            left_child = node["left_child"]
            right_child = node["right_child"]
            # Check that childs and parent information does match
            assert node_id == nodes[left_child]["parent"]
            assert node_id == nodes[right_child]["parent"]
            # TODO: check that left childs are indeed left childs (we don't use
            #  these for now...)
            # assert nodes[left_child]["is_left"] == True
            # assert not nodes[right_child]["is_left"]


@pytest.mark.parametrize("n_estimators", [2])
@pytest.mark.parametrize("aggregation, dirichlet", [(False, 0.0), (True, 1e-7)])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("max_features", [None, "auto"])
@pytest.mark.parametrize("random_state", [42])
@pytest.mark.parametrize("step", [1.0])
@pytest.mark.parametrize("multiclass", ["multinomial"])
@pytest.mark.parametrize("cat_split_strategy", ["binary", "all", "random"])
@pytest.mark.parametrize(
    "one_hot_encode, use_categoricals", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("criterion", ("gini", "entropy"))
def test_nodes_on_adult(
    n_estimators,
    aggregation,
    class_weight,
    n_jobs,
    max_features,
    random_state,
    dirichlet,
    step,
    multiclass,
    cat_split_strategy,
    one_hot_encode,
    use_categoricals,
    criterion,
):
    dataset = load_adult()
    dataset.test_size = 1.0 / 5
    dataset.standardize = False
    dataset.one_hot_encode = one_hot_encode
    X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
    if use_categoricals:
        categorical_features = dataset.categorical_features_
    else:
        categorical_features = None
    clf = ForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        multiclass=multiclass,
        cat_split_strategy=cat_split_strategy,
        aggregation=aggregation,
        criterion=criterion,
        max_features=max_features,
        class_weight=class_weight,
        categorical_features=categorical_features,
        random_state=random_state,
        dirichlet=dirichlet,
        step=step,
    )
    clf.fit(X_train, y_train)

    for tree in clf.trees:
        node_count = tree._tree.node_count
        nodes = tree._tree.nodes[:node_count]
        bin_partitions = tree._tree.bin_partitions
        assert tree._tree.nodes.size >= node_count
        check_nodes(nodes, bin_partitions, aggregation)


@pytest.mark.parametrize("n_estimators", [2])
@pytest.mark.parametrize("aggregation, dirichlet", [(False, 0.0), (True, 1e-7)])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("max_features", [None, "auto"])
@pytest.mark.parametrize("random_state", [42])
@pytest.mark.parametrize("step", [1.0])
@pytest.mark.parametrize("multiclass", ["multinomial", "ovr"])
@pytest.mark.parametrize("cat_split_strategy", ["binary", "all", "random"])
@pytest.mark.parametrize(
    "one_hot_encode, use_categoricals", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("criterion", ("gini", "entropy"))
def test_nodes_on_car(
    n_estimators,
    aggregation,
    class_weight,
    n_jobs,
    max_features,
    random_state,
    dirichlet,
    step,
    multiclass,
    cat_split_strategy,
    one_hot_encode,
    use_categoricals,
    criterion,
):
    dataset = load_car()
    dataset.test_size = 1.0 / 5
    dataset.standardize = False
    dataset.one_hot_encode = one_hot_encode
    X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
    if use_categoricals:
        categorical_features = dataset.categorical_features_
    else:
        categorical_features = None
    clf = ForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        multiclass=multiclass,
        cat_split_strategy=cat_split_strategy,
        aggregation=aggregation,
        criterion=criterion,
        max_features=max_features,
        class_weight=class_weight,
        categorical_features=categorical_features,
        random_state=random_state,
        dirichlet=dirichlet,
        step=step,
    )
    clf.fit(X_train, y_train)

    for tree in clf.trees:
        node_count = tree._tree.node_count
        nodes = tree._tree.nodes[:node_count]
        bin_partitions = tree._tree.bin_partitions
        assert tree._tree.nodes.size >= node_count
        check_nodes(nodes, bin_partitions, aggregation)


@pytest.mark.parametrize("n_estimators", [2])
@pytest.mark.parametrize("aggregation, dirichlet", [(False, 0.0), (True, 1e-7)])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("max_features", [None, "auto"])
@pytest.mark.parametrize("random_state", [42])
@pytest.mark.parametrize("step", [1.0])
@pytest.mark.parametrize("multiclass", ["multinomial"])
@pytest.mark.parametrize("cat_split_strategy", ["binary", "all", "random"])
@pytest.mark.parametrize(
    "one_hot_encode, use_categoricals", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("criterion", ("gini", "entropy"))
def test_nodes_on_churn(
    n_estimators,
    aggregation,
    class_weight,
    n_jobs,
    max_features,
    random_state,
    dirichlet,
    step,
    multiclass,
    cat_split_strategy,
    one_hot_encode,
    use_categoricals,
    criterion,
):
    dataset = load_car()
    dataset.test_size = 1.0 / 5
    dataset.standardize = False
    dataset.one_hot_encode = one_hot_encode
    X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
    if use_categoricals:
        categorical_features = dataset.categorical_features_
    else:
        categorical_features = None
    clf = ForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        multiclass=multiclass,
        cat_split_strategy=cat_split_strategy,
        aggregation=aggregation,
        criterion=criterion,
        max_features=max_features,
        class_weight=class_weight,
        categorical_features=categorical_features,
        random_state=random_state,
        dirichlet=dirichlet,
        step=step,
    )
    clf.fit(X_train, y_train)

    for tree in clf.trees:
        node_count = tree._tree.node_count
        nodes = tree._tree.nodes[:node_count]
        bin_partitions = tree._tree.bin_partitions
        assert tree._tree.nodes.size >= node_count
        check_nodes(nodes, bin_partitions, aggregation)


@pytest.mark.parametrize("n_estimators", [2])
@pytest.mark.parametrize("aggregation", [False, True])
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("max_features", [None, "auto"])
@pytest.mark.parametrize("random_state", [42])
@pytest.mark.parametrize("step", [1.0])
@pytest.mark.parametrize(
    "one_hot_encode, use_categoricals", [(False, False), (False, True), (True, False)]
)
def test_nodes_on_boston(
    n_estimators,
    aggregation,
    n_jobs,
    max_features,
    random_state,
    step,
    one_hot_encode,
    use_categoricals,
):
    dataset = load_boston()
    dataset.test_size = 1.0 / 5
    dataset.standardize = False
    dataset.one_hot_encode = one_hot_encode
    X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)

    if use_categoricals:
        categorical_features = dataset.categorical_features_
    else:
        categorical_features = None
    clf = ForestRegressor(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        aggregation=aggregation,
        max_features=max_features,
        categorical_features=categorical_features,
        random_state=random_state,
        step=step,
    )
    clf.fit(X_train, y_train)

    for tree in clf.trees:
        node_count = tree._tree.node_count
        nodes = tree._tree.nodes[:node_count]
        bin_partitions = tree._tree.bin_partitions
        assert tree._tree.nodes.size >= node_count
        check_nodes(nodes, bin_partitions, aggregation)


@pytest.mark.parametrize("aggregation", [False, True])
@pytest.mark.parametrize("max_features", [None, "auto"])
@pytest.mark.parametrize("random_state", [42])
@pytest.mark.parametrize(
    "one_hot_encode, use_categoricals", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize(
    "min_samples_split, min_samples_leaf", [(2, 1), (13, 7), (3, 5)]
)
@pytest.mark.parametrize("criterion", ("gini", "entropy"))
def test_min_samples_split_min_samples_leaf_on_adult(
    aggregation,
    max_features,
    random_state,
    one_hot_encode,
    use_categoricals,
    min_samples_split,
    min_samples_leaf,
    criterion,
):
    dataset = load_adult()
    dataset.test_size = 1.0 / 5
    dataset.standardize = False
    dataset.one_hot_encode = one_hot_encode

    n_estimators = 3
    n_jobs = -1
    class_weight = "balanced"
    multiclass = "multinomial"
    step = 1.0

    X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
    if use_categoricals:
        categorical_features = dataset.categorical_features_
    else:
        categorical_features = None
    clf = ForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        multiclass=multiclass,
        aggregation=aggregation,
        max_features=max_features,
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        categorical_features=categorical_features,
        random_state=random_state,
        step=step,
    )
    clf.fit(X_train, y_train)
    min_samples = min(min_samples_split, min_samples_leaf)
    for tree in clf.trees:
        node_count = tree._tree.node_count
        nodes = tree._tree.nodes[:node_count]
        for node_id, node in enumerate(nodes):
            # Check that nodes respect the min_samples_split and
            # min_samples_leaf constraints
            assert node["n_samples_train"] >= min_samples
            if aggregation:
                assert node["n_samples_valid"] >= min_samples


@pytest.mark.parametrize("aggregation", [False, True])
@pytest.mark.parametrize("max_features", [None, "auto"])
@pytest.mark.parametrize("random_state", [42])
@pytest.mark.parametrize(
    "one_hot_encode, use_categoricals", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize(
    "min_samples_split, min_samples_leaf", [(2, 1), (13, 7), (3, 5)]
)
def test_min_samples_split_min_samples_leaf_on_boston(
    aggregation,
    max_features,
    random_state,
    one_hot_encode,
    use_categoricals,
    min_samples_split,
    min_samples_leaf,
):
    dataset = load_boston()
    dataset.test_size = 1.0 / 5
    dataset.standardize = False
    dataset.one_hot_encode = one_hot_encode

    n_estimators = 3
    n_jobs = -1
    step = 1.0

    X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
    if use_categoricals:
        categorical_features = dataset.categorical_features_
    else:
        categorical_features = None
    clf = ForestRegressor(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        aggregation=aggregation,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        categorical_features=categorical_features,
        random_state=random_state,
        step=step,
    )
    clf.fit(X_train, y_train)
    min_samples = min(min_samples_split, min_samples_leaf)
    for tree in clf.trees:
        node_count = tree._tree.node_count
        nodes = tree._tree.nodes[:node_count]
        for node_id, node in enumerate(nodes):
            # Check that nodes respect the min_samples_split and
            # min_samples_leaf constraints
            assert node["n_samples_train"] >= min_samples
            if aggregation:
                assert node["n_samples_valid"] >= min_samples
