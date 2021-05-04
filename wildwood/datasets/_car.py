import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
from sklearn.datasets._base import _fetch_remote

from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home


ARCHIVE = RemoteFileMetadata(
    filename="car.data",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    checksum="b703a9ac69f11e64ce8c223c0a40de4d2e9d769f7fb20be5f8f2e8a619893d83",
)

logger = logging.getLogger(__name__)


def _fetch_car(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "car")
    data_path = join(data_dir, "car.csv.gz")

    columns = ["Buying", "Maint", "Doors", "Persons", "LugBoot", "Safety", "Evaluation"]

    if download_if_missing and not exists(data_path):
        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE.url)
        _fetch_remote(ARCHIVE, dirname=data_dir)
        logger.debug("Converting as a single dataframe with the correct schema")
        filepath = join(data_dir, ARCHIVE.filename)

        df = pd.read_csv(filepath, header=None, sep=",")
        df.columns = columns

        df.to_csv(data_path, compression="gzip", index=False)
        # Remove temporary files
        os.remove(filepath)


def load_car(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_car(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "car")
    data_path = join(data_dir, "car.csv.gz")

    dtype = {
        "Buying": "category",
        "Maint": "category",
        "Doors": "category",
        "Persons": "category",
        "LugBoot": "category",
        "Safety": "category",
    }
    dataset = Dataset.from_dtype(
        name="car",
        task="multiclass-classification",
        label_column="Evaluation",
        dtype=dtype,
    )
    return dataset.load_from_csv(data_path, dtype=dtype)
