import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
from sklearn.datasets._base import _fetch_remote

from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home


ARCHIVE = RemoteFileMetadata(
    filename="kick.csv",
    url="https://www.openml.org/data/get_csv/19335679/kick.arff",
    checksum="32ca4d28e69c6bd56c201025d3aed651f393c1364264a29b512be109006dc7fb",
)

logger = logging.getLogger(__name__)

dtype = {
    "PurchDate": np.int,
    "Auction": "category",
    "VehYear": np.int,
    "VehicleAge": np.int,
    "Make": "category",
    "Model": "category",
    "Trim": "category",
    "SubModel": "category",
    "Color": "category",
    "Transmission": "category",
    "WheelTypeID": "category",
    "WheelType": "category",
    "VehOdo": np.int,
    "Nationality": "category",
    "Size": "category",
    "TopThreeAmericanName": "category",
    "MMRAcquisitionAuctionAveragePrice": np.float,
    "MMRAcquisitionAuctionCleanPrice": np.float,
    "MMRAcquisitionRetailAveragePrice": np.float,
    "MMRAcquisitonRetailCleanPrice": np.float,
    "MMRCurrentAuctionAveragePrice": np.float,
    "MMRCurrentAuctionCleanPrice": np.float,
    "MMRCurrentRetailAveragePrice": np.float,
    "MMRCurrentRetailCleanPrice": np.float,
    "PRIMEUNIT": "category",
    "AUCGUART": "category",
    "BYRNO": "category",
    "VNZIP1": "category",
    "VNST": "category",
    "VehBCost": np.float,
    "IsOnlineSale": "category",
    "WarrantyCost": np.int,
}


def _fetch_kick(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "kick")
    data_path = join(data_dir, "kick.csv.gz")

    if download_if_missing and not exists(data_path):
        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE.url)
        _fetch_remote(ARCHIVE, dirname=data_dir)
        logger.debug("Converting as a single dataframe with the correct schema")
        filepath = join(data_dir, ARCHIVE.filename)

        numerical_columns = [a for a, b in dtype.items() if b != "category"]

        df = pd.read_csv(filepath)
        for nc in numerical_columns:
            df[nc] = df[nc].replace("?", np.nan)

        df.to_csv(data_path, compression="gzip", index=False)
        # Remove temporary files
        os.remove(filepath)


def load_kick(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_kick(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "kick")
    data_path = join(data_dir, "kick.csv.gz")

    dataset = Dataset.from_dtype(
        name="kick",
        drop_columns=None,  # ["PRIMEUNIT" ,"AUCGUART"],
        task="binary-classification",
        label_column="IsBadBuy",
        dtype=dtype,
    )
    return dataset.load_from_csv(data_path, dtype=dtype)
