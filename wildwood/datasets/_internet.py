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
    filename="internet_usage.csv",
    url="https://www.openml.org/data/get_csv/52407/internet_usage.arff",
    checksum="6c0fd6f6ce5fe1b6edf7c7872d91230cd9d99b7dd01795e478bdc5a47f7581d8",
)


logger = logging.getLogger(__name__)


def _fetch_internet(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "internet")
    data_path = join(data_dir, "internet.csv.gz")

    if download_if_missing and not exists(data_path):

        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE.url)
        _fetch_remote(ARCHIVE, dirname=data_dir)

        logger.debug("Converting as a single dataframe with the correct schema")
        file = join(data_dir, ARCHIVE.filename)

        df = pd.read_csv(file)
        df = df.fillna("?")
        df.to_csv(data_path, compression="gzip", index=False)
        # Remove temporary files
        os.remove(file)


def load_internet(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_internet(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "internet")
    data_path = join(data_dir, "internet.csv.gz")

    df = pd.read_csv(data_path)

    dtype = {field: "category" for field in list(df.columns)[1:-1]}

    dataset = Dataset.from_dtype(
        name="internet",
        task="multiclass-classification",
        drop_columns=["who"],
        label_column="Actual_Time",
        dtype=dtype,
    )
    return dataset.load_from_csv(data_path, dtype=dtype)
