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
    load_diabetes,
    load_epsilon,
    load_higgs,
    load_internet,
    load_kick,
    load_kddcup99,
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
    load_amazon,
)


class TestDataLoaders(object):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data_extractions = [
            # {  # TODO: not working for log reg actually..
            # "one_hot_encode": True,
            # "standardize": True,
            # "drop": "first",
            # "pd_df_categories": False,
            # },
            {
                "one_hot_encode": True,
                "standardize": False,
                "drop": None,
                "pd_df_categories": False,
            },
            {
                "one_hot_encode": False,
                "standardize": False,
                "drop": None,
                "pd_df_categories": False,
            },
            {
                "one_hot_encode": False,
                "standardize": False,
                "drop": None,
                "pd_df_categories": True,
            },
        ]

    def _helper_test_loader(
        self, dataset, n_classes, n_samples, n_features, n_features_categorical
    ):
        for data_extraction in self.data_extractions:
            for key, val in data_extraction.items():
                setattr(dataset, key, val)

            X_train, X_test, y_train, y_test = dataset.extract(random_state=42)

            assert dataset.n_classes_ == n_classes
            assert dataset.n_samples_ == n_samples
            assert dataset.n_features_ == n_features
            assert dataset.n_features_categorical_ == n_features_categorical

            if not dataset.one_hot_encode:
                assert dataset.n_columns_ == dataset.n_features_
                assert X_train.shape[1] == dataset.n_features_
                assert X_test.shape[1] == dataset.n_features_

    def test_load_adult(self):
        dataset = load_adult()
        assert dataset.name == "adult"
        assert dataset.task == "binary-classification"
        self._helper_test_loader(dataset, 2, 48841, 14, 8)

    def test_load_amazon(self):
        dataset = load_amazon()
        assert dataset.name == "amazon"
        assert dataset.task == "binary-classification"
        self._helper_test_loader(dataset, 2, 32769, 9, 9)

    def test_load_bank(self):
        dataset = load_bank()
        assert dataset.name == "bank"
        assert dataset.task == "binary-classification"
        self._helper_test_loader(dataset, 2, 45211, 16, 10)

    def test_load_breastcancer(self):
        dataset = load_breastcancer()
        assert dataset.name == "breastcancer"
        assert dataset.task == "binary-classification"
        self._helper_test_loader(dataset, 2, 569, 30, 0)

    def test_load_car(self):
        dataset = load_car()
        assert dataset.name == "car"
        assert dataset.task == "multiclass-classification"
        self._helper_test_loader(dataset, 4, 1728, 6, 6)

    def test_load_cardio(self):
        dataset = load_cardio()
        assert dataset.name == "cardio"
        assert dataset.task == "multiclass-classification"
        self._helper_test_loader(dataset, 10, 2126, 35, 0)

    def test_load_churn(self):
        dataset = load_churn()
        assert dataset.name == "churn"
        assert dataset.task == "binary-classification"
        self._helper_test_loader(dataset, 2, 3333, 19, 4)

    def test_load_covtype(self):
        dataset = load_covtype()
        assert dataset.name == "covtype"
        assert dataset.task == "multiclass-classification"
        self._helper_test_loader(dataset, 7, 581012, 54, 0)

    def test_load_default_cb(self):
        dataset = load_default_cb()
        assert dataset.name == "default-cb"
        assert dataset.task == "binary-classification"
        self._helper_test_loader(dataset, 2, 30000, 23, 3)

    # def test_load_diabetes(self):
    #     dataset = load_diabetes()
    #     assert dataset.name == 'diabetes'
    #     assert dataset.task == "regression"
    #     self._helper_test_loader(dataset, None, 442, 10, 0)

    def test_load_epsilon(self):  # 3GB data
        pass

    def test_load_higgs(self):  # too big to test
        pass

    def test_load_kddcup99(self):  # too big to test
        pass

    def test_load_internet(self):
        dataset = load_internet()
        assert dataset.name == "internet"
        assert dataset.task == "multiclass-classification"
        self._helper_test_loader(dataset, 46, 10108, 70, 70)

    def test_load_kick(self):
        dataset = load_kick()
        assert dataset.name == "kick"
        assert dataset.task == "binary-classification"
        self._helper_test_loader(dataset, 2, 72983, 32, 18)

    def test_load_letter(self):
        dataset = load_letter()
        assert dataset.name == "letter"
        assert dataset.task == "multiclass-classification"
        self._helper_test_loader(dataset, 26, 20000, 16, 0)

    def test_load_satimage(self):
        dataset = load_satimage()
        assert dataset.name == "satimage"
        assert dataset.task == "multiclass-classification"
        self._helper_test_loader(dataset, 6, 5104, 36, 0)

    def test_load_sensorless(self):
        dataset = load_sensorless()
        assert dataset.name == "sensorless"
        assert dataset.task == "multiclass-classification"
        self._helper_test_loader(dataset, 11, 58509, 48, 0)

    def test_load_spambase(self):
        dataset = load_spambase()
        assert dataset.name == "spambase"
        assert dataset.task == "binary-classification"
        self._helper_test_loader(dataset, 2, 4601, 57, 0)
