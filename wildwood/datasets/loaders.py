# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np

from .dataset import Dataset
from ._adult import load_adult
from ._bank import load_bank
from ._car import load_car
from ._default_cb import load_default_cb

# TODO: kdd98 https://www.openml.org/d/23513, Il y a plein de features numeriques avec un grand nombre de "missing" values


def load_breastcancer():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data["frame"]

    continuous_columns = [col for col in df.columns if col != "target"]
    categorical_columns = None
    dataset = Dataset(
        name="breastcancer",
        task="binary-classification",
        label_column="target",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_boston():
    from sklearn.datasets import load_boston

    data = load_boston()
    # Load as a dataframe. We set some features as categorical so that we have a
    # regression example with categorical features for tests...
    df = pd.DataFrame(data["data"], columns=data["feature_names"]).astype(
        {"CHAS": "category"}
    )
    df["target"] = data["target"]
    continuous_columns = [
        "ZN",
        "RAD",
        "CRIM",
        "INDUS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]
    categorical_columns = ["CHAS"]
    dataset = Dataset(
        name="boston",
        task="regression",
        label_column="target",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_californiahousing():
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data["frame"]
    continuous_columns = [col for col in df.columns if col != "MedHouseVal"]
    categorical_columns = None
    dataset = Dataset(
        name="californiahousing",
        task="regression",
        label_column="MedHouseVal",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_letor():
    dtype = {str(i): np.float64 for i in range(1, 47)}
    dataset = Dataset.from_dtype(
        name="letor", task="multiclass-classification", label_column="0", dtype=dtype,
    )
    return dataset.load_from_csv("letor.csv.gz", dtype=dtype)


def load_cardio():
    dtype = {
        "b": np.int32,
        "e": np.int32,
        "LBE": np.int32,
        "LB": np.int32,
        "AC": np.int32,
        "FM": np.int32,
        "UC": np.int32,
        "ASTV": np.int32,
        "MSTV": np.float32,
        "ALTV": np.int32,
        "MLTV": np.float32,
        "DL": np.int32,
        "DS": np.int32,
        "DP": np.int32,
        "DR": np.int32,
        "Width": np.int32,
        "Min": np.int32,
        "Max": np.int32,
        "Nmax": np.int32,
        "Nzeros": np.int32,
        "Mode": np.int32,
        "Mean": np.int32,
        "Median": np.int32,
        "Variance": np.int32,
        "Tendency": np.int32,
        "A": np.int32,
        "B": np.int32,
        "C": np.int32,
        "D": np.int32,
        "E": np.int32,
        "AD": np.int32,
        "DE": np.int32,
        "LD": np.int32,
        "FS": np.int32,
        "SUSP": np.int32,
    }
    dataset = Dataset.from_dtype(
        name="cardio",
        task="multiclass-classification",
        label_column="CLASS",
        dtype=dtype,
        # We drop the NSP column which is a 3-class version of the label
        drop_columns=["FileName", "Date", "SegFile", "NSP"],
    )
    return dataset.load_from_csv(
        "cardiotocography.csv.gz", sep=";", decimal=",", dtype=dtype
    )


#
# def load_amazon():
#     from catboost.datasets import amazon
#
#     df_train, df_test = amazon()
#     df = pd.concat([df_train, df_test], axis="column")
#
#     df.info()


def load_churn():
    dtype = {
        "State": "category",
        "Account Length": np.int32,
        "Area Code": "category",
        "Int'l Plan": "category",
        "VMail Plan": "category",
        "VMail Message": np.int32,
        "Day Mins": np.float32,
        "Day Calls": np.int32,
        "Day Charge": np.float32,
        "Eve Mins": np.float32,
        "Eve Calls": np.int32,
        "Eve Charge": np.float32,
        "Night Mins": np.float32,
        "Night Calls": np.int32,
        "Night Charge": np.float32,
        "Intl Mins": np.float32,
        "Intl Calls": np.int32,
        "Intl Charge": np.float32,
        "CustServ Calls": np.int32,
    }
    dataset = Dataset.from_dtype(
        name="churn",
        task="binary-classification",
        label_column="Churn?",
        dtype=dtype,
        # We drop the "Phone" column
        drop_columns=["Phone"],
    )
    return dataset.load_from_csv("churn.csv.gz", dtype=dtype)


def load_epsilon_catboost():
    from catboost.datasets import epsilon

    df_train, df_test = epsilon()
    columns = list(df_train.columns)
    dataset = Dataset(
        name="epsilon",
        task="binary-classification",
        label_column=columns[0],
        continuous_columns=columns[1:],
        categorical_columns=None,
    )

    df = pd.concat([df_train, df_test], axis="index")
    dataset.df_raw = df
    return dataset


def load_covtype():
    from sklearn.datasets import fetch_covtype

    data = fetch_covtype(as_frame=True)
    df = data["frame"]
    continuous_columns = [col for col in df.columns if col != "Cover_Type"]
    categorical_columns = None
    dataset = Dataset(
        name="covtype",
        task="multiclass-classification",
        label_column="Cover_Type",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_diabetes():
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    df = data["frame"]
    continuous_columns = [col for col in df.columns if col != "target"]
    categorical_columns = None
    dataset = Dataset(
        name="diabetes",
        task="regression",
        label_column="target",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_kddcup99():
    from sklearn.datasets import fetch_kddcup99

    # We load the full datasets with 4.8 million rows
    data = fetch_kddcup99(as_frame=True, percent10=False)
    df = data["frame"]
    # We change the dtypes (for some weird reason everything is "object"...)
    dtype = {
        "duration": np.float32,
        "protocol_type": "category",
        "service": "category",
        "flag": "category",
        "src_bytes": np.float32,
        "dst_bytes": np.float32,
        "land": "category",
        "wrong_fragment": np.float32,
        "urgent": np.float32,
        "hot": np.float32,
        "num_failed_logins": np.float32,
        "logged_in": "category",
        "num_compromised": np.float32,
        "root_shell": "category",
        "su_attempted": "category",
        "num_root": np.float32,
        "num_file_creations": np.float32,
        "num_shells": np.float32,
        "num_access_files": np.float32,
        "num_outbound_cmds": np.float32,
        "is_host_login": "category",
        "is_guest_login": "category",
        "count": np.float32,
        "srv_count": np.float32,
        "serror_rate": np.float32,
        "srv_serror_rate": np.float32,
        "rerror_rate": np.float32,
        "srv_rerror_rate": np.float32,
        "same_srv_rate": np.float32,
        "diff_srv_rate": np.float32,
        "srv_diff_host_rate": np.float32,
        "dst_host_count": np.float32,
        "dst_host_srv_count": np.float32,
        "dst_host_same_srv_rate": np.float32,
        "dst_host_diff_srv_rate": np.float32,
        "dst_host_same_src_port_rate": np.float32,
        "dst_host_srv_diff_host_rate": np.float32,
        "dst_host_serror_rate": np.float32,
        "dst_host_srv_serror_rate": np.float32,
        "dst_host_rerror_rate": np.float32,
        "dst_host_srv_rerror_rate": np.float32,
    }
    df = df.astype(dtype)
    dataset = Dataset.from_dtype(
        name="kddcup",
        task="multiclass-classification",
        label_column="labels",
        dtype=dtype,
    )
    dataset.df_raw = df
    return dataset


def load_letter():
    dtype = {
        "X0": np.float32,
        "X1": np.float32,
        "X2": np.float32,
        "X3": np.float32,
        "X4": np.float32,
        "X5": np.float32,
        "X6": np.float32,
        "X7": np.float32,
        "X8": np.float32,
        "X9": np.float32,
        "X10": np.float32,
        "X11": np.float32,
        "X12": np.float32,
        "X13": np.float32,
        "X14": np.float32,
        "X15": np.float32,
    }
    dataset = Dataset.from_dtype(
        name="letter",
        task="multiclass-classification",
        label_column="y",
        dtype=dtype,
        drop_columns=["Unnamed: 0"],
    )
    return dataset.load_from_csv("letter.csv.gz", dtype=dtype)


def load_satimage():
    dtype = {
        "X0": np.float32,
        "X1": np.float32,
        "X2": np.float32,
        "X3": np.float32,
        "X4": np.float32,
        "X5": np.float32,
        "X6": np.float32,
        "X7": np.float32,
        "X8": np.float32,
        "X9": np.float32,
        "X10": np.float32,
        "X11": np.float32,
        "X12": np.float32,
        "X13": np.float32,
        "X14": np.float32,
        "X15": np.float32,
        "X16": np.float32,
        "X17": np.float32,
        "X18": np.float32,
        "X19": np.float32,
        "X20": np.float32,
        "X21": np.float32,
        "X22": np.float32,
        "X23": np.float32,
        "X24": np.float32,
        "X25": np.float32,
        "X26": np.float32,
        "X27": np.float32,
        "X28": np.float32,
        "X29": np.float32,
        "X30": np.float32,
        "X31": np.float32,
        "X32": np.float32,
        "X33": np.float32,
        "X34": np.float32,
        "X35": np.float32,
    }
    dataset = Dataset.from_dtype(
        name="satimage",
        task="multiclass-classification",
        label_column="y",
        drop_columns=["Unnamed: 0"],
        dtype=dtype,
    )
    return dataset.load_from_csv("satimage.csv.gz", dtype=dtype)


def load_sensorless():
    dtype = {
        0: np.float32,
        1: np.float32,
        2: np.float32,
        3: np.float32,
        4: np.float32,
        5: np.float32,
        6: np.float32,
        7: np.float32,
        8: np.float32,
        9: np.float32,
        10: np.float32,
        11: np.float32,
        12: np.float32,
        13: np.float32,
        14: np.float32,
        15: np.float32,
        16: np.float32,
        17: np.float32,
        18: np.float32,
        19: np.float32,
        20: np.float32,
        21: np.float32,
        22: np.float32,
        23: np.float32,
        24: np.float32,
        25: np.float32,
        26: np.float32,
        27: np.float32,
        28: np.float32,
        29: np.float32,
        30: np.float32,
        31: np.float32,
        32: np.float32,
        33: np.float32,
        34: np.float32,
        35: np.float32,
        36: np.float32,
        37: np.float32,
        38: np.float32,
        39: np.float32,
        40: np.float32,
        41: np.float32,
        42: np.float32,
        43: np.float32,
        44: np.float32,
        45: np.float32,
        46: np.float32,
        47: np.float32,
    }
    dataset = Dataset.from_dtype(
        name="sensorless",
        task="multiclass-classification",
        label_column=48,
        dtype=dtype,
    )
    return dataset.load_from_csv("sensorless.csv.gz", sep=" ", header=None, dtype=dtype)


def load_spambase():
    dtype = {
        0: np.float32,
        1: np.float32,
        2: np.float32,
        3: np.float32,
        4: np.float32,
        5: np.float32,
        6: np.float32,
        7: np.float32,
        8: np.float32,
        9: np.float32,
        10: np.float32,
        11: np.float32,
        12: np.float32,
        13: np.float32,
        14: np.float32,
        15: np.float32,
        16: np.float32,
        17: np.float32,
        18: np.float32,
        19: np.float32,
        20: np.float32,
        21: np.float32,
        22: np.float32,
        23: np.float32,
        24: np.float32,
        25: np.float32,
        26: np.float32,
        27: np.float32,
        28: np.float32,
        29: np.float32,
        30: np.float32,
        31: np.float32,
        32: np.float32,
        33: np.float32,
        34: np.float32,
        35: np.float32,
        36: np.float32,
        37: np.float32,
        38: np.float32,
        39: np.float32,
        40: np.float32,
        41: np.float32,
        42: np.float32,
        43: np.float32,
        44: np.float32,
        45: np.float32,
        46: np.float32,
        47: np.float32,
        48: np.float32,
        49: np.float32,
        50: np.float32,
        51: np.float32,
        52: np.float32,
        53: np.float32,
        54: np.float32,
        55: np.int32,
        56: np.int32,
    }
    dataset = Dataset.from_dtype(
        name="spambase", task="binary-classification", label_column=57, dtype=dtype,
    )
    return dataset.load_from_csv("spambase.csv.gz", header=None, dtype=dtype)


#


# class KDDCup(Datasets):  # multiclass
#     def __init__(
#         self,
#         path=None,
#         test_split=0.3,
#         random_state=0,
#         normalize_intervals=False,
#         one_hot_categoricals=False,
#         as_pandas=False,
#         subsample=None,
#     ):
#         from sklearn.datasets import fetch_kddcup99
#
#         print("Loading full KDDCdup datasets (percent10=False)")
#         print("")
#         data, target = fetch_kddcup99(
#             percent10=False, return_X_y=True, random_state=random_state, as_frame=True
#         )  # as_pandas)
#
#         if subsample is not None:
#             print("Subsampling datasets with subsample={}".format(subsample))
#
#         data = data[:subsample]
#         target = target[:subsample]
#
#         discrete = [
#             "protocol_type",
#             "service",
#             "flag",
#             "land",
#             "logged_in",
#             "is_host_login",
#             "is_guest_login",
#         ]
#         continuous = list(set(data.columns) - set(discrete))
#
#         dummies = pd.get_dummies(target)
#         dummies.columns = list(range(len(dummies.columns)))
#         self.target = dummies.idxmax(axis=1)  # .values
#         self.binary = False
#         self.task = "classification"
#
#         self.n_classes = self.target.max() + 1  # 23
#
#         X_continuous = data[continuous].astype("float32")
#         if normalize_intervals:
#             mins = X_continuous.min()
#             X_continuous = (X_continuous - mins) / (X_continuous.max() - mins)
#
#         if one_hot_categoricals:
#             X_discrete = pd.get_dummies(data[discrete], prefix_sep="#")  # .values
#         else:
#             # X_discrete = (data[discrete]).apply(lambda x: pd.factorize(x)[0])
#             X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).astype(int)
#
#         self.one_hot_categoricals = one_hot_categoricals
#         self.data = X_continuous.join(X_discrete)
#
#         if not as_pandas:
#             self.data = self.data.values
#             self.target = self.target.values
#         else:
#             self.data.columns = list(range(self.data.shape[1]))
#
#         self.size, self.n_features = self.data.shape
#         self.nb_continuous_features = len(continuous)  # 34#32
#
#         self.split_train_test(test_split, random_state)
#


# TODO: newsgroup is sparse, so we'll work on it later
def load_newsgroup():
    pass


# class NewsGroups(Datasets):  # multiclass
#     def __init__(
#         self,
#         path=None,
#         test_split=0.3,
#         random_state=0,
#         normalize_intervals=False,
#         one_hot_categoricals=False,
#         as_pandas=False,
#         subsample=None,
#     ):
#         from sklearn.datasets import fetch_20newsgroups_vectorized
#
#         data, target = fetch_20newsgroups_vectorized(
#             return_X_y=True, as_frame=True
#         )  # as_pandas)
#
#         if subsample is not None:
#             print("Subsampling datasets with subsample={}".format(subsample))
#
#         data = data[:subsample]
#         target = target[:subsample]
#
#         self.target = target
#         self.binary = False
#         self.task = "classification"
#
#         self.n_classes = self.target.max() + 1
#
#         if normalize_intervals:
#             mins = data.min()
#             data = (data - mins) / (data.max() - mins)
#
#         self.data = data
#
#         if not as_pandas:
#             self.data = self.data.values
#             self.target = self.target.values
#
#         self.size, self.n_features = self.data.shape
#         self.nb_continuous_features = self.n_features
#
#         self.split_train_test(test_split, random_state)


loaders_small_classification = [
    load_adult,
    load_bank,
    load_breastcancer,
    load_car,
    load_cardio,
    load_churn,
    load_default_cb,
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
]

loaders_small_regression = [load_boston, load_californiahousing, load_diabetes]

loaders_medium = [load_covtype]

loaders_large = []


def describe_datasets(include="small-classification", random_state=42):
    if include == "small-classification":
        loaders = loaders_small_classification
    elif include == "small-regression":
        loaders = loaders_small_regression
    else:
        raise ValueError("include=%r is not supported for now." % include)

    col_name = []
    col_n_samples = []
    col_n_features = []
    col_task = []
    col_n_classes = []
    col_n_features_categorical = []
    col_n_features_continuous = []
    col_scaled_gini = []
    col_n_samples_train = []
    col_n_samples_test = []
    col_n_columns = []
    for loader in loaders:
        dataset = loader()
        dataset.one_hot_encode = True
        dataset.standardize = True
        X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
        n_samples_train, n_columns = X_train.shape
        n_samples_test, _ = X_test.shape
        col_name.append(dataset.name)
        col_task.append(dataset.task)
        col_n_samples.append(dataset.n_samples_in_)
        col_n_features.append(dataset.n_features_in_)
        col_n_classes.append(dataset.n_classes_)
        col_n_features_categorical.append(dataset.n_features_categorical_)
        col_n_features_continuous.append(dataset.n_features_continuous_)
        col_scaled_gini.append(dataset.scaled_gini_)
        col_n_samples_train.append(n_samples_train)
        col_n_samples_test.append(n_samples_test)
        col_n_columns.append(n_columns)

    if "regression" in include:
        df_description = pd.DataFrame(
            {
                "dataset": col_name,
                "task": col_task,
                "n_samples": col_n_samples,
                "n_samples_train": col_n_samples_train,
                "n_samples_test": col_n_samples_test,
                "n_features_cat": col_n_features_categorical,
                "n_features_cont": col_n_features_continuous,
                "n_features": col_n_features,
                "n_columns": col_n_columns,
            }
        )
    else:
        df_description = pd.DataFrame(
            {
                "dataset": col_name,
                "task": col_task,
                "n_samples": col_n_samples,
                "n_samples_train": col_n_samples_train,
                "n_samples_test": col_n_samples_test,
                "n_features_cat": col_n_features_categorical,
                "n_features_cont": col_n_features_continuous,
                "n_features": col_n_features,
                "n_classes": col_n_classes,
                "n_columns": col_n_columns,
                "scaled_gini": col_scaled_gini,
            }
        )

    return df_description


if __name__ == "__main__":
    # df_descriptions = describe_datasets()
    # print(df_descriptions)
    #
    # datasets = load_covtype()

    load_kddcup99()

    # print(datasets)
