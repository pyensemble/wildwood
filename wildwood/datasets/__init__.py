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
    load_diabetes_cl,
    load_heart,
    load_heart2,
    load_gamma_particle,
    load_phoneme,
    load_banknote,
    load_ionosphere,
    load_wilt2,
    load_covtype,
    load_smoke,
    load_cc_default,
    load_hcv,
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


loader_from_name = {
    "adult": load_adult,
    "bank": load_bank,
    "breastcancer": load_breastcancer,
    "car": load_car,
    "cardio": load_cardio,
    "churn": load_churn,
    "default-cb": load_default_cb,
    "letter": load_letter,
    "satimage": load_satimage,
    "sensorless": load_sensorless,
    "spambase": load_spambase,
    "amazon": load_amazon,
    "covtype": load_covtype,
    "internet": load_internet,
    "kick": load_kick,
    "kddcup": load_kddcup99,
    "higgs": load_higgs,
}
