import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
from sklearn.datasets._base import _fetch_remote

from .dataset import Dataset, load_raw_dataset
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


def load_bank(download_if_missing=True, raw=False, verbose=False):
    # Fetch the data is necessary
    _fetch_bank(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "bank")
    data_path = join(data_dir, "bank.zip")

    dtype = {
        "age": np.int64,
        "job": "category",
        "marital": "category",
        "education": "category",
        "default": "category",
        "balance": np.int64,
        "housing": "category",
        "loan": "category",
        "contact": "category",
        "day": "category",
        "month": "category",
        "duration": np.int64,
        "campaign": np.int64,
        "pdays": np.int64,
        "previous": np.int64,
        "poutcome": "category",
    }
    zip_ref = zipfile.ZipFile(data_path, "r")
    zip_ref.extract("bank-full.csv", path=data_dir)
    zip_ref.close()
    unzipped_data_path = join(data_dir, "bank-full.csv")
    label_column = "y"

    if raw:
        X, y = load_raw_dataset(
            data_path=unzipped_data_path,
            label_column=label_column,
            dtype=dtype,
            verbose=verbose,
            sep=";",
        )
        os.remove(unzipped_data_path)
        return X, y
    else:
        dataset = Dataset.from_dtype(
            name="bank",
            task="binary-classification",
            label_column=label_column,
            dtype=dtype,
        )
        dataset_ = dataset.load_from_csv(unzipped_data_path, sep=";", dtype=dtype)
        os.remove(unzipped_data_path)
        return dataset_
