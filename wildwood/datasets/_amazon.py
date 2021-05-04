import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
from sklearn.datasets._base import _fetch_remote

from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home


ARCHIVE = RemoteFileMetadata(
    filename="amazon.csv",
    url="https://www.openml.org/data/get_csv/1681098/phpmPOD5A",
    checksum="c5289c0aac5d56c3255b4976297527c64e2fcc3afede820c43b2e62a0a564605",
)

logger = logging.getLogger(__name__)


def _fetch_amazon(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "amazon")
    data_path = join(data_dir, "amazon.csv.gz")

    if download_if_missing and not exists(data_path):
        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE.url)
        _fetch_remote(ARCHIVE, dirname=data_dir)
        logger.debug("Converting as a single dataframe with the correct schema")
        filepath = join(data_dir, ARCHIVE.filename)

        df = pd.read_csv(filepath)

        df.to_csv(data_path, compression="gzip", index=False)
        # Remove temporary files
        os.remove(filepath)


def load_amazon(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_amazon(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "amazon")
    data_path = join(data_dir, "amazon.csv.gz")

    fields = [
        "RESOURCE",
        "MGR_ID",
        "ROLE_ROLLUP_1",
        "ROLE_ROLLUP_2",
        "ROLE_DEPTNAME",
        "ROLE_TITLE",
        "ROLE_FAMILY_DESC",
        "ROLE_FAMILY",
        "ROLE_CODE",
    ]

    dtype = {field: "category" for field in fields}
    dataset = Dataset.from_dtype(
        name="amazon", task="binary-classification", label_column="target", dtype=dtype,
    )
    return dataset.load_from_csv(data_path, dtype=dtype)
