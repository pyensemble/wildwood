# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

"""
This module performs unittests for the dataset class
"""

import pytest
import numpy as np

from wildwood.preprocessing.dataset import array_to_dataset, dataset_to_array


np.random.seed(42)


@pytest.mark.parametrize("n_samples", [2, 32, 17, 1000, 1_000_000])
@pytest.mark.parametrize(
    "max_values, dtype",
    [
        (np.array([32], dtype=np.uint64), np.uint8),
        (np.array([17, 2, 64], dtype=np.uint64), np.uint8),
        (np.array([1024, 32, 64], dtype=np.uint64), np.uint16),
        (np.array([123765, 2, 16], dtype=np.uint64), np.uint32),
        (np.array([12323765, 123765, 2, 1024], dtype=np.uint64), np.uint64),
        (np.array([2378, 2, 213, 123765, 7, 64, 1024, 3], dtype=np.uint64), np.uint64),
    ],
)
def test_dataset(n_samples, max_values, dtype):
    n_features = max_values.size
    X_in = np.asfortranarray(
        np.random.randint(max_values + 1, size=(n_samples, n_features)), dtype=dtype
    )
    dataset = array_to_dataset(X_in)
    X_out = dataset_to_array(dataset)
    np.testing.assert_array_equal(X_in, X_out)
