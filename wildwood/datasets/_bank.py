import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
from sklearn.datasets._base import _fetch_remote

from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home
import zipfile

ARCHIVE = RemoteFileMetadata(
    filename="bank.zip",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
    checksum="99d7e8eb12401ed278b793984423915411ea8df099e1795f9fefe254f513fe5e",
)


logger = logging.getLogger(__name__)


def _fetch_bank(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "bank")
    data_path = join(data_dir, "bank.zip")

    if download_if_missing and not exists(data_path):

        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE.url)
        _fetch_remote(ARCHIVE, dirname=data_dir)


def load_bank(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_bank(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "bank")
    data_path = join(data_dir, "bank.zip")

    dtype = {
        "age": np.int,
        "job": "category",
        "marital": "category",
        "education": "category",
        "default": "category",
        "balance": np.int,
        "housing": "category",
        "loan": "category",
        "contact": "category",
        "day": "category",
        "month": "category",
        "duration": np.int,
        "campaign": np.int,
        "pdays": np.int,
        "previous": np.int,
        "poutcome": "category",
    }
    dataset = Dataset.from_dtype(
        name="bank", task="binary-classification", label_column="y", dtype=dtype
    )
    zip_ref = zipfile.ZipFile(data_path, "r")
    zip_ref.extract("bank-full.csv", path=data_dir)
    zip_ref.close()
    unzipped_data_path = join(data_dir, "bank-full.csv")
    dataset_ = dataset.load_from_csv(unzipped_data_path, sep=";", dtype=dtype)
    os.remove(unzipped_data_path)
    return dataset_
