import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
from sklearn.datasets._base import _fetch_remote

from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home


ARCHIVE_TRAIN = RemoteFileMetadata(
    filename="adult_train",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    checksum="5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
)


ARCHIVE_TEST = RemoteFileMetadata(
    filename="adult_test",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    checksum="a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
)


logger = logging.getLogger(__name__)


def _fetch_adult(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "adult")
    data_path = join(data_dir, "adult.csv.gz")

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        ">50K?",
    ]

    if download_if_missing and not exists(data_path):
        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE_TRAIN.url)
        _fetch_remote(ARCHIVE_TRAIN, dirname=data_dir)
        logger.info("Downloading %s" % ARCHIVE_TEST.url)
        _fetch_remote(ARCHIVE_TEST, dirname=data_dir)

        logger.debug("Converting as a single dataframe with the correct schema")
        archive_path_train = join(data_dir, ARCHIVE_TRAIN.filename)
        # We use skiprows = [0] since there is a weird first line containing :
        #   |1x3 Cross validator
        df_train = pd.read_csv(archive_path_train, header=None, sep=",", skiprows=[0])
        df_train.columns = columns
        archive_path_test = join(data_dir, ARCHIVE_TEST.filename)
        # We use skiprows = [0] since there is a weird first line containing :
        #   |1x3 Cross validator
        df_test = pd.read_csv(archive_path_test, header=None, sep=",", skiprows=[0])
        df_test.columns = columns
        df_test = df_test.replace(" >50K.", " >50K")
        df_test = df_test.replace(" <=50K.", " <=50K")

        df = pd.concat([df_train, df_test], axis="index")
        df.to_csv(data_path, compression="gzip", index=False)
        # Remove temporary files
        os.remove(archive_path_train)
        os.remove(archive_path_test)


def load_adult(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_adult(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "adult")
    data_path = join(data_dir, "adult.csv.gz")

    dtype = {
        "age": np.int,
        "workclass": "category",
        "fnlwgt": np.int,
        "education": "category",
        "education-num": np.int,
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "sex": "category",
        "capital-gain": np.int,
        "capital-loss": np.int,
        "hours-per-week": np.int,
        "native-country": "category",
    }
    dataset = Dataset.from_dtype(
        name="adult", task="binary-classification", label_column=">50K?", dtype=dtype
    )
    return dataset.load_from_csv(data_path, dtype=dtype)
