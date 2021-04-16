"""
This modules includes dataset loaders for experiments conducted with WildWood
"""

from .dataset import Dataset

from ._adult import load_adult
from ._bank import load_bank
from ._higgs import load_higgs
from ._car import load_car
from ._kick import load_kick
from ._amazon import load_amazon
from ._epsilon import load_epsilon
from ._internet import load_internet

from .loaders import (
    load_boston,
    load_letor,
    load_epsilon_catboost,
    load_breastcancer,
    load_cardio,
    load_churn,
    load_covtype,
    load_diabetes,
    load_default_cb,
    load_kddcup99,
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
    describe_datasets,
)

from .signals import get_signal, make_regression
