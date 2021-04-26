"""
This modules includes dataset loaders for experiments conducted with WildWood
"""

from .dataset import Dataset

from ._adult import load_adult

from .loaders import (
    load_bank,
    load_boston,
    load_breastcancer,
    load_car,
    load_cardio,
    load_churn,
    load_covtype,
    load_diabetes,
    load_default_cb,
    load_kddcup,
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
    describe_datasets,
)

from .signals import get_signal, make_regression
