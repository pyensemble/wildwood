import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
from sklearn.datasets._base import _fetch_remote

from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home
import zipfile

ARCHIVE_TRAIN = RemoteFileMetadata(
    filename="epsilon_normalized.bz2",
    url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2",
    checksum="aff916d4f97f18d286558ca088d2a9f7e1fcee9376539a5aa6ef5b7ef9dfa978",
)


ARCHIVE_TEST = RemoteFileMetadata(
    filename="epsilon_normalized.t.bz2",
    url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2",
    checksum="cb299295ad11e200696eaa3050f5d8cf700eaa9c65e6aa859bda959f8669458b",
)


logger = logging.getLogger(__name__)


def _fetch_epsilon(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "epsilon")
    data_path1 = join(data_dir, "epsilon_normalized.bz2")
    data_path2 = join(data_dir, "epsilon_normalized.t.bz2")

    if download_if_missing and not exists(data_path1):
        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE_TRAIN.url)
        _fetch_remote(ARCHIVE_TRAIN, dirname=data_dir)
        logger.info("Downloading %s" % ARCHIVE_TEST.url)
        _fetch_remote(ARCHIVE_TEST, dirname=data_dir)


def load_epsilon(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_epsilon(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "epsilon")
    data_path1 = join(data_dir, "epsilon_normalized.bz2")
    data_path2 = join(data_dir, "epsilon_normalized.t.bz2")

    from sklearn.datasets import load_svmlight_file

    X, y = load_svmlight_file(
        data_path1, dtype=np.float32  # pylint: disable=unbalanced-tuple-unpacking
    )
    X = X.toarray()
    X_test, y_test = load_svmlight_file(
        data_path2, dtype=np.float32  # pylint: disable=unbalanced-tuple-unpacking
    )
    X_test = X_test.toarray()

    X = np.vstack((X, X_test))
    y = np.append(y, y_test)

    y[y <= 0] = 0
    columns = [str(i) for i in range(X.shape[1] + 1)]

    dataset = Dataset(
        name="epsilon",
        task="binary-classification",
        label_column=columns[0],
        continuous_columns=columns[1:],
        categorical_columns=None,
    )

    dataset.df_raw = pd.DataFrame(np.hstack((y[:, np.newaxis], X)))
    dataset.df_raw.columns = columns
    return dataset
