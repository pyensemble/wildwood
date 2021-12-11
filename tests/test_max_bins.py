# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module contains tests that ensure that the Forest works with a varying number of
bins, and checks it performance.
"""


import numpy as np
import pytest

from sklearn.metrics import (
    log_loss,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

from wildwood import ForestClassifier, ForestRegressor
from wildwood.datasets import load_adult, load_car, load_boston


def approx(v, abs=1e-15):
    return pytest.approx(v, abs=abs)


@pytest.mark.parametrize(
    "loader, is_categorical_, required_log_loss",
    [
        (
            load_adult,
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], dtype=np.bool_),
            0.41,
        ),
        (load_car, np.array([1, 1, 1, 1, 1, 1], dtype=np.bool_), 0.29),
    ],
)
@pytest.mark.parametrize("aggregation", (False, True))
@pytest.mark.parametrize("max_bins", (10, 30, 256, 1234))
def test_several_max_bins_for_classification(
    loader, is_categorical_, required_log_loss, max_bins, aggregation
):
    X, y = loader(raw=True)
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    n_estimators = 10
    class_weight = "balanced"
    n_jobs = -1
    dirichlet = 1e-2

    clf = ForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        aggregation=aggregation,
        max_bins=max_bins,
        dirichlet=dirichlet,
        class_weight=class_weight,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    np.testing.assert_equal(clf.is_categorical_, is_categorical_)
    y_scores_test = clf.predict_proba(X_test)
    assert log_loss(y_test, y_scores_test) < required_log_loss


@pytest.mark.parametrize(
    "loader, is_categorical_, required_mse",
    [
        (
            load_boston,
            np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_),
            17.83,
        )
    ],
)
@pytest.mark.parametrize("aggregation", (False, True))
@pytest.mark.parametrize("max_bins", (10, 30, 256, 1234))
def test_several_max_bins_for_regression(
    loader, is_categorical_, required_mse, max_bins, aggregation
):
    X, y = loader(raw=True)
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    n_estimators = 10
    n_jobs = -1

    clf = ForestRegressor(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        aggregation=aggregation,
        max_bins=max_bins,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    np.testing.assert_equal(clf.is_categorical_, is_categorical_)
    y_pred = clf.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    assert mse_test < required_mse
