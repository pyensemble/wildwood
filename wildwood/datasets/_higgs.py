import errno
from gzip import GzipFile
import logging
import os
from os.path import dirname, exists, join

import numpy as np
import joblib

from sklearn.datasets._base import (
    _fetch_remote,
    _convert_data_dataframe,
    RemoteFileMetadata,
    _sha256,
)

# from sklearn.datasets import get_data_home
from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home


from urllib.request import urlretrieve


# from ..utils import Bunch
# from ..utils import check_random_state
# from ..utils import shuffle as shuffle_method
# from ..utils.validation import _deprecate_positional_args
#

# https://archive.ics.uci.edu/ml/datasets/HIGGS#
ARCHIVE = RemoteFileMetadata(
    filename="HIGGS.csv.gz",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
    checksum="ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698",
)


logger = logging.getLogger(__name__)


def _fetch_higgs(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "higgs")
    data_path = join(data_dir, "HIGGS.csv.gz")

    if download_if_missing and not exists(data_path):
        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE.url)
        _fetch_remote(ARCHIVE, dirname=data_dir)


def load_higgs(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_higgs(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "higgs")
    data_path = join(data_dir, "HIGGS.csv.gz")

    dtype = {i + 1: np.float32 for i in range(28)}
    dataset = Dataset.from_dtype(
        name="higgs", task="binary-classification", label_column=0, dtype=dtype
    )
    return dataset.load_from_csv(data_path, dtype=dtype, header=None)


# def _fetch_higgs(download_if_missing=True):
#     data_home = get_data_home()
#     data_dir = join(data_home, "higgs")
#     data_path = join(data_dir, "higgs.csv.gz")
#
#     if download_if_missing and not exists(data_path):
#         _mkdirp(data_dir)
#         logger.info("Downloading %s" % ARCHIVE.url)
#         _fetch_remote(ARCHIVE, dirname=data_dir)
#
#
#
# def load_higgs(download_if_missing=True):
#     # Fetch the data is necessary
#     _fetch_higgs(download_if_missing)
#
#     data_home = get_data_home()
#     data_dir = join(data_home, "higgs")
#     data_path = join(data_dir, "higgs.csv.gz")
#
#     dtype = {i+1:np.float32 for i in range(28)}
#
#     dataset = Dataset.from_dtype(
#         name="higgs", task="binary-classification", label_column=0, dtype=dtype
#     )
#     return dataset.load_from_csv(data_path, dtype=dtype, header=None, dtype=np.float32)


# def load_higgs():
#     # TODO: Prendre le code de KDD de scikit par exemple
#     pass
#
#
# # #
# # class Higgs(Datasets):  # binary
# #
# #     def __init__(
# #         self,
# #         filename=None,
# #         subsample=None,
# #         test_split=0.005,
# #         random_state=0,
# #         normalize_intervals=False,
# #         as_pandas=False,
# #     ):
# #         filename = filename or os.path.dirname(__file__) + "/HIGGS.csv.gz"
# #         URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
# #         if not os.path.exists(filename):
# #             reply = (
# #                 str(
# #                     input(
# #                         "Higgs datasets file not provided, would you like to download it ? (2.6 Go) (y/n): "
# #                     )
# #                 )
# #                 .lower()
# #                 .strip()
# #             )
# #             if reply == "y":
# #                 print(f"Downloading {URL} to {filename} (2.6 GB)...")
# #                 urlretrieve(URL, filename)
# #                 print("done.")
# #             else:
# #                 print("Higgs boson datasets unavailable, exiting")
# #                 exit()
# #
# #         print(f"Loading Higgs boson datasets from {filename}...")
# #         tic = time()
# #         with GzipFile(filename) as f:
# #             self.df = pd.read_csv(f, header=None, dtype=np.float32)
# #         toc = time()
# #         print(f"Loaded {self.df.values.nbytes / 1e9:0.3f} GB in {toc - tic:0.3f}s")
# #
# #         if subsample is not None:
# #             print("Subsampling datasets with subsample={}".format(subsample))
# #
# #         self.data = self.df[:subsample]
# #         self.target = self.data.pop(0).astype(int).values
# #
# #         if normalize_intervals:
# #             mins = self.data.min()
# #             self.data = (self.data - mins) / (self.data.max() - mins)
# #
# #         if not as_pandas:
# #             self.data = np.ascontiguousarray(self.data.values)
# #
# #         self.binary = True
# #         self.task = "classification"
# #         self.n_classes = 2
# #         self.one_hot_categoricals = False
# #         self.size, self.n_features = self.data.shape
# #         self.nb_continuous_features = self.n_features
# #
# #         print("Making train/test split ...")
# #
# #         self.split_train_test(test_split, random_state)
# #         # self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
# #         #    self.data, self.target, test_size=self.get_test_size(test_split), random_state=random_state)
# #
# #         print("Done.")


# def load_higgs(download_if_missing=True):
#
#     data_home = get_data_home()
#     higgs = _fetch_higgs(download_if_missing=download_if_missing)
#
#     print(type(higgs))
#
#     # data = kddcup99.data
#     # target = kddcup99.target
#     # feature_names = kddcup99.feature_names
#     # target_names = kddcup99.target_names
#     #
#     # if subset == "SA":
#     #     s = target == b"normal."
#     #     t = np.logical_not(s)
#     #     normal_samples = data[s, :]
#     #     normal_targets = target[s]
#     #     abnormal_samples = data[t, :]
#     #     abnormal_targets = target[t]
#     #
#     #     n_samples_abnormal = abnormal_samples.shape[0]
#     #     # selected abnormal samples:
#     #     random_state = check_random_state(random_state)
#     #     r = random_state.randint(0, n_samples_abnormal, 3377)
#     #     abnormal_samples = abnormal_samples[r]
#     #     abnormal_targets = abnormal_targets[r]
#     #
#     #     data = np.r_[normal_samples, abnormal_samples]
#     #     target = np.r_[normal_targets, abnormal_targets]
#     #
#     # if subset == "SF" or subset == "http" or subset == "smtp":
#     #     # select all samples with positive logged_in attribute:
#     #     s = data[:, 11] == 1
#     #     data = np.c_[data[s, :11], data[s, 12:]]
#     #     feature_names = feature_names[:11] + feature_names[12:]
#     #     target = target[s]
#     #
#     #     data[:, 0] = np.log((data[:, 0] + 0.1).astype(float, copy=False))
#     #     data[:, 4] = np.log((data[:, 4] + 0.1).astype(float, copy=False))
#     #     data[:, 5] = np.log((data[:, 5] + 0.1).astype(float, copy=False))
#     #
#     #     if subset == "http":
#     #         s = data[:, 2] == b"http"
#     #         data = data[s]
#     #         target = target[s]
#     #         data = np.c_[data[:, 0], data[:, 4], data[:, 5]]
#     #         feature_names = [feature_names[0], feature_names[4], feature_names[5]]
#     #
#     #     if subset == "smtp":
#     #         s = data[:, 2] == b"smtp"
#     #         data = data[s]
#     #         target = target[s]
#     #         data = np.c_[data[:, 0], data[:, 4], data[:, 5]]
#     #         feature_names = [feature_names[0], feature_names[4], feature_names[5]]
#     #
#     #     if subset == "SF":
#     #         data = np.c_[data[:, 0], data[:, 2], data[:, 4], data[:, 5]]
#     #         feature_names = [
#     #             feature_names[0],
#     #             feature_names[2],
#     #             feature_names[4],
#     #             feature_names[5],
#     #         ]
#     #
#     # if shuffle:
#     #     data, target = shuffle_method(data, target, random_state=random_state)
#     #
#     # module_path = dirname(__file__)
#     # with open(join(module_path, "descr", "kddcup99.rst")) as rst_file:
#     #     fdescr = rst_file.read()
#     #
#     # frame = None
#     # if as_frame:
#     #     frame, data, target = _convert_data_dataframe(
#     #         "fetch_kddcup99", data, target, feature_names, target_names
#     #     )
#     #
#     # if return_X_y:
#     #     return data, target
#     #
#     # return Bunch(
#     #     data=data,
#     #     target=target,
#     #     frame=frame,
#     #     target_names=target_names,
#     #     feature_names=feature_names,
#     #     DESCR=fdescr,
#     # )
#
#
# def _fetch_higgs(download_if_missing=True):
#     data_home = get_data_home()
#     higgs_dir = join(data_home, "higgs")
#     archive = ARCHIVE
#     data_path = join(higgs_dir, "higgs.csv.gz")
#     available = exists(data_path)
#
#     print("samples_path:", data_path)
#     print("available:", available)
#
#     # dt = [('duration', int),
#     #       ('protocol_type', 'S4'),
#     #       ('service', 'S11'),
#     #       ('flag', 'S6'),
#     #       ('src_bytes', int),
#     #       ('dst_bytes', int),
#     #       ('land', int),
#     #       ('wrong_fragment', int),
#     #       ('urgent', int),
#     #       ('hot', int),
#     #       ('num_failed_logins', int),
#     #       ('logged_in', int),
#     #       ('num_compromised', int),
#     #       ('root_shell', int),
#     #       ('su_attempted', int),
#     #       ('num_root', int),
#     #       ('num_file_creations', int),
#     #       ('num_shells', int),
#     #       ('num_access_files', int),
#     #       ('num_outbound_cmds', int),
#     #       ('is_host_login', int),
#     #       ('is_guest_login', int),
#     #       ('count', int),
#     #       ('srv_count', int),
#     #       ('serror_rate', float),
#     #       ('srv_serror_rate', float),
#     #       ('rerror_rate', float),
#     #       ('srv_rerror_rate', float),
#     #       ('same_srv_rate', float),
#     #       ('diff_srv_rate', float),
#     #       ('srv_diff_host_rate', float),
#     #       ('dst_host_count', int),
#     #       ('dst_host_srv_count', int),
#     #       ('dst_host_same_srv_rate', float),
#     #       ('dst_host_diff_srv_rate', float),
#     #       ('dst_host_same_src_port_rate', float),
#     #       ('dst_host_srv_diff_host_rate', float),
#     #       ('dst_host_serror_rate', float),
#     #       ('dst_host_srv_serror_rate', float),
#     #       ('dst_host_rerror_rate', float),
#     #       ('dst_host_srv_rerror_rate', float),
#     #       ('labels', 'S16')]
#
#     # column_names = [c[0] for c in dt]
#     # target_names = column_names[-1]
#     # feature_names = column_names[:-1]
#
#     # exit(0)
#
#     if download_if_missing and not available:
#         _mkdirp(higgs_dir)
#         logger.info("Downloading %s" % archive.url)
#         _fetch_remote(archive, dirname=higgs_dir)
#         logger.debug("extracting archive")
#         archive_path = join(higgs_dir, archive.filename)
#         file_ = GzipFile(filename=archive_path, mode="r")
#         Xy = []
#         for line in file_.readlines():
#             line = line.decode()
#             Xy.append(line.replace("\n", "").split(","))
#         file_.close()
#         logger.debug("extraction done")
#         os.remove(archive_path)
#
#         Xy = np.asarray(Xy, dtype=object)
#
#         # for j in range(42):
#         #     Xy[:, j] = Xy[:, j].astype(DT[j])
#         # X = Xy[:, :-1]
#         # y = Xy[:, -1]
#         # XXX bug when compress!=0:
#         # (error: 'Incorrect data length while decompressing[...] the file
#         #  could be corrupted.')
#
#         joblib.dump(Xy, data_path, compress=0)
#
#     else:
#         print("else reading data from %s" % data_path)
#         Xy = joblib.load(data_path)
#         print("done reading data")
#
#     return Xy
#
#     # joblib.dump(y, targets_path, compress=0)
#     # elif not available:
#     #     if not download_if_missing:
#     #         raise IOError("Data not found and `download_if_missing` is False")
#     #
#     # try:
#     #     X, y
#     # except NameError:
#     #     X = joblib.load(samples_path)
#     #     y = joblib.load(targets_path)
#     #
#     # return Bunch(
#     #     data=X,
#     #     target=y,
#     #     feature_names=feature_names,
#     #     target_names=[target_names],
#     # )
#
#
# def _mkdirp(d):
#     """Ensure directory d exists (like mkdir -p on Unix)
#     No guarantee that the directory is writable.
#     """
#     try:
#         os.makedirs(d)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise
#
#
# if __name__ == "__main__":
#     load_higgs(download_if_missing=True)
