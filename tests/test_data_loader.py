# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module performs unittests for the data loaders proposed in wildwood.
"""

import numpy as np
import pandas as pd
import pytest

from wildwood.datasets import (
    load_adult,
    load_bank,
    load_breastcancer,
    load_car,
    load_cardio,
    load_churn,
    load_covtype,
    load_default_cb,
    # load_diabetes,
    # load_epsilon,
    # load_higgs,
    load_internet,
    load_kick,
    # load_kddcup99,
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
    load_amazon,
)


@pytest.mark.parametrize(
    "loader, name, task, n_classes, n_samples, n_features, n_features_categorical",
    [
        (load_adult, "adult", "binary-classification", 2, 48841, 14, 8),
        (load_amazon, "amazon", "binary-classification", 2, 32769, 9, 9),
        (load_bank, "bank", "binary-classification", 2, 45211, 16, 10),
        (load_breastcancer, "breastcancer", "binary-classification", 2, 569, 30, 0),
        (load_car, "car", "multiclass-classification", 4, 1728, 6, 6),
        (load_cardio, "cardio", "multiclass-classification", 10, 2126, 35, 0),
        (load_churn, "churn", "binary-classification", 2, 3333, 19, 4),
        (load_covtype, "covtype", "multiclass-classification", 7, 581012, 54, 0),
        (load_default_cb, "default-cb", "binary-classification", 2, 30000, 23, 3),
        # (load_diabetes, 'diabetes' "regression", None, 442, 10, 0),
        (load_internet, "internet", "multiclass-classification", 46, 10108, 70, 70),
        (load_kick, "kick", "binary-classification", 2, 72983, 32, 18),
        (load_letter, "letter", "multiclass-classification", 26, 20000, 16, 0),
        (load_satimage, "satimage", "multiclass-classification", 6, 5104, 36, 0),
        (load_sensorless, "sensorless", "multiclass-classification", 11, 58509, 48, 0),
        (load_spambase, "spambase", "binary-classification", 2, 4601, 57, 0),
    ],
)
@pytest.mark.parametrize(
    "one_hot_encode, standardize, drop, pd_df_categories",
    [
        (True, False, None, False),
        (False, False, None, False),
        (False, False, None, True),
    ],
)
def test_loaders(
    loader,
    name,
    task,
    n_classes,
    n_samples,
    n_features,
    n_features_categorical,
    one_hot_encode,
    standardize,
    drop,
    pd_df_categories,
):
    dataset = loader()
    dataset.one_hot_encode = one_hot_encode
    dataset.standardize = standardize
    dataset.drop = drop
    dataset.pd_df_categories = pd_df_categories
    X_train, X_test, y_train, y_test = dataset.extract(random_state=42)

    assert dataset.n_classes_ == n_classes
    assert dataset.n_samples_in_ == n_samples
    assert dataset.n_features_in_ == n_features
    assert dataset.n_features_categorical_ == n_features_categorical

    if not dataset.one_hot_encode:
        assert dataset.n_columns_ == dataset.n_features_in_
        assert X_train.shape[1] == dataset.n_features_in_
        assert X_test.shape[1] == dataset.n_features_in_

    # Test also that the raw loading (no preprocessing works as well)
    X, y = loader(raw=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert np.unique(y).shape[0] == n_classes
    assert X.shape[0] == y.shape[0] == n_samples
    assert X.shape[1] == n_features
