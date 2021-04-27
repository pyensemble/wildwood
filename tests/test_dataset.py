# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

# pytest -v

import os
import pytest
import pandas as pd

from wildwood.datasets import describe_datasets


def approx(v, abs=1e-15):
    return pytest.approx(v, abs=abs)


class TestDataset(object):
    @pytest.fixture(autouse=True)
    def _setup(self):
        module_path = os.path.dirname(__file__)
        filename = os.path.join(module_path, "dataset-small-classification.csv.gz")
        self.df_small_classification = pd.read_csv(filename)
        filename = os.path.join(module_path, "dataset-small-regression.csv.gz")
        self.df_small_regression = pd.read_csv(filename)

    def test_dataset(self):
        df = describe_datasets(include="small-classification")
        pd.testing.assert_frame_equal(df, self.df_small_classification)
        df = describe_datasets(include="small-regression")
        pd.testing.assert_frame_equal(df, self.df_small_regression)
