import os
from time import time
import pandas as pd
import numpy as np

import logging


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
    FunctionTransformer,
)
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def gini(probs):
    return 1.0 - (probs ** 2).sum()


def scaled_gini(probs):
    n_classes = probs.size
    max_gini = gini(np.ones(n_classes) / n_classes)
    return gini(probs) / max_gini


# TODO: same thing with entropy


class Dataset:
    """

    test_split : None or float

    """

    def __init__(
        self,
        name,
        task,
        *,
        label_column=None,
        continuous_columns=None,
        categorical_columns=None,
        drop_columns=None,
        test_size=0.3,
        standardize=True,
        one_hot_encode=True,
        sparse=False,
        drop="first",
        pd_df_categories=False,
        verbose=False,
    ):
        self.name = name
        self.task = task
        self.label_column = label_column
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.drop_columns = drop_columns
        self.standardize = standardize
        self.one_hot_encode = one_hot_encode
        self.sparse = sparse
        self.drop = drop
        self.filename = None
        self.url = None
        self.test_size = test_size
        self.pd_df_categories = pd_df_categories
        self.verbose = verbose

        self.transformer = None
        self.label_encoder = None
        self.df_raw = None

        self.n_samples_ = None
        self.n_samples_train_ = None
        self.n_samples_test_ = None
        self.n_features_ = None
        self.n_columns_ = None
        self.columns_ = None
        self.categorical_columns_ = None
        self.continuous_columns_ = None
        self.n_features_categorical_ = None
        self.n_features_continuous_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.scaled_gini_ = None
        self.categorical_features_ = None

    def __repr__(self):
        repr = "Dataset("
        repr += "name=%r" % self.name
        repr += ", task=%r" % self.task
        repr += ", label_column=%r" % self.label_column
        repr += ", standardize=%r" % self.standardize
        repr += ", one_hot_encode=%r" % self.one_hot_encode
        repr += ", drop=%r" % self.drop
        repr += ", sparse=%r" % self.sparse
        repr += ")"
        return repr

    @staticmethod
    def from_dtype(
        name, task, label_column, dtype, drop_columns=None, one_hot_encode=True
    ):
        continuous_columns = [
            col for col, col_type in dtype.items() if col_type != "category"
        ]
        categorical_columns = [
            col for col, col_type in dtype.items() if col_type == "category"
        ]
        dataset = Dataset(
            name=name,
            task=task,
            label_column=label_column,
            one_hot_encode=one_hot_encode,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            drop_columns=drop_columns,
        )
        return dataset

    def load_from_csv(self, filename, **kwargs):
        module_path = os.path.dirname(__file__)
        filename = os.path.join(module_path, "data", filename)
        if self.verbose:
            logging.info("Reading from file %s..." % filename)
        tic = time()
        df = pd.read_csv(filename, **kwargs)
        if self.drop_columns:
            df.drop(self.drop_columns, axis="columns", inplace=True)
        toc = time()
        if self.verbose:
            logging.info("Read from file %s in %.2f seconds" % (filename, toc - tic))
        self.df_raw = df
        return self

    def load_from_url(self, url):
        self.url = url
        return self

    # TODO: en l'etat j'ai l'impression que je normalise la colonne de label

    def _build_transform(self):
        """A helper function that builds the transformation corresponding to the
        one_hot_encode and standardize attributes.

        Returns
        -------

        """
        features_transformations = []

        if self.continuous_columns:
            # If continuous_columns is not empty or None
            if self.standardize:
                # If required use StandardScaler
                continuous_transformer = ColumnTransformer(
                    [
                        (
                            "continuous_transformer",
                            StandardScaler(),
                            self.continuous_columns,
                        )
                    ]
                )
            else:
                # Otherwise keep the continuous columns unchanged
                continuous_transformer = ColumnTransformer(
                    [
                        (
                            "continuous_transformer",
                            FunctionTransformer(),
                            self.continuous_columns,
                        )
                    ]
                )
            features_transformations.append(
                ("continuous_transformer", continuous_transformer)
            )

        if self.categorical_columns:
            # If categorical_columns is not empty or None
            if self.one_hot_encode:
                # If required apply one-hot encoding
                categorical_transformer = ColumnTransformer(
                    [
                        (
                            "categorical_transformer",
                            OneHotEncoder(drop=self.drop, sparse=self.sparse),  # TODO drop vs handle_unknwon
                            self.categorical_columns,
                        )
                    ]
                )
            else:
                # Otherwise just use an ordinal encoder (this just replaces the
                # modalities by integers)
                categorical_transformer = ColumnTransformer(
                    [
                        (
                            "categorical_transformer",
                            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                            self.categorical_columns,
                        )
                    ]
                )

            features_transformations.append(
                ("categorical_transformer", categorical_transformer)
            )

        transformer = FeatureUnion(features_transformations)
        self.transformer = transformer

        if self.task == "regression":
            self.label_encoder = FunctionTransformer()
        else:
            self.label_encoder = LabelEncoder()

        return self

    def extract(self, random_state=None):
        self._build_transform()
        df = self.df_raw
        # Don't put self.n_features_ = df.shape[1] since for now df contains the
        # column label
        self.n_samples_, _ = df.shape

        # A list containing the names of the categorical colunms
        self.categorical_columns_ = [
            col
            for col, dtype in df.dtypes.items()
            if col != self.label_column and dtype.name == "category"
        ]
        # A list containing the names of the continuous colunms
        self.continuous_columns_ = [
            col
            for col, dtype in df.dtypes.items()
            if col != self.label_column and dtype.name != "category"
        ]

        self.n_features_categorical_ = len(self.categorical_columns_)
        self.n_features_continuous_ = len(self.continuous_columns_)
        self.n_features_ = self.n_features_categorical_ + self.n_features_continuous_

        if not self.one_hot_encode and self.n_features_categorical_ > 0:
            # If we do not use one-hot encoding, we compute a boolean mask indicating
            # which features are categorical. We use the fact that by construction of
            # the Dataset categorical features come last.
            categorical_features = np.zeros(self.n_features_, dtype=np.bool)
            #
            categorical_features[-self.n_features_categorical_:] = True
            self.categorical_features_ = categorical_features.copy()

        stratify = None if self.task == "regression" else df[self.label_column]

        df_train, df_test = train_test_split(
            df,
            test_size=self.test_size,
            shuffle=True,
            random_state=random_state,
            stratify=stratify,
        )

        self.transformer = self.transformer.fit(df_train)
        X_train = self.transformer.transform(df_train)
        X_test = self.transformer.transform(df_test)

        # An array holding the names of all the columns
        columns = []
        if self.n_features_continuous_ > 0:
            columns.extend(self.continuous_columns_)

        if self.n_features_categorical_ > 0:
            if self.one_hot_encode:
                # Get the list of modalities from the OneHotEncoder
                all_modalities = (
                    self.transformer.transformer_list[-1][1]
                    .transformers_[0][1]
                    .categories_
                )
                for categorical_column, modalities in zip(
                    self.categorical_columns_, all_modalities
                ):
                    # Add the columns for this features
                    columns.extend(
                        [
                            categorical_column
                            # + "#"
                            # + modality
                            + "#"
                            + str(idx_modality)
                            for idx_modality, modality in enumerate(modalities)
                        ]
                    )
            else:
                columns.extend(self.categorical_columns_)
        self.columns_ = columns

        if self.pd_df_categories:
            #columns = (self.continuous_columns or [])+(self.categorical_columns or [])
            X_train = pd.DataFrame(X_train, columns=columns)
            X_test = pd.DataFrame(X_test, columns=columns)
            if self.categorical_columns is not None:
                X_train[self.categorical_columns] = X_train[self.categorical_columns].astype(int).astype('category')
                X_test[self.categorical_columns] = X_test[self.categorical_columns].astype(int).astype('category')

        n_samples_train, n_columns = X_train.shape
        n_samples_test, _ = X_test.shape
        self.n_columns_ = n_columns
        self.n_samples_train_ = n_samples_train
        self.n_samples_test_ = n_samples_test

        self.label_encoder = self.label_encoder.fit(df_train[self.label_column])
        y_train = self.label_encoder.transform(df_train[self.label_column])
        y_test = self.label_encoder.transform(df_test[self.label_column])

        if self.task != "regression":
            self.classes_ = self.label_encoder.classes_
            self.n_classes_ = len(self.classes_)
            # Encode the full column containing the labels to compute its gini index
            y_encoded = LabelEncoder().fit_transform(df[self.label_column])
            label_counts = np.bincount(y_encoded)
            label_probs = label_counts / label_counts.sum()
            self.scaled_gini_ = scaled_gini(label_probs)

        return X_train, X_test, y_train, y_test


def load_adult():
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
    return dataset.load_from_csv("adult.csv.gz", dtype=dtype)


def load_bank():
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
    return dataset.load_from_csv("bank.csv.gz", dtype=dtype)


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
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    df["target"] = data["target"]
    continuous_columns = [col for col in df.columns if col != "target"]
    categorical_columns = None
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


def load_car():
    """Load the car dataset

    Parameters
    ----------
    path
    filename

    Returns
    -------

    """
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
    return dataset.load_from_csv("car.csv.gz", dtype=dtype)


def load_cardio():
    dtype = {
        "b": np.int,
        "e": np.int,
        "LBE": np.int,
        "LB": np.int,
        "AC": np.int,
        "FM": np.int,
        "UC": np.int,
        "ASTV": np.int,
        "MSTV": np.float,
        "ALTV": np.int,
        "MLTV": np.float,
        "DL": np.int,
        "DS": np.int,
        "DP": np.int,
        "DR": np.int,
        "Width": np.int,
        "Min": np.int,
        "Max": np.int,
        "Nmax": np.int,
        "Nzeros": np.int,
        "Mode": np.int,
        "Mean": np.int,
        "Median": np.int,
        "Variance": np.int,
        "Tendency": np.int,
        "A": np.int,
        "B": np.int,
        "C": np.int,
        "D": np.int,
        "E": np.int,
        "AD": np.int,
        "DE": np.int,
        "LD": np.int,
        "FS": np.int,
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


def load_churn():
    dtype = {
        "State": "category",
        "Account Length": np.int,
        "Area Code": "category",
        "Int'l Plan": "category",
        "VMail Plan": "category",
        "VMail Message": np.int,
        "Day Mins": np.float,
        "Day Calls": np.int,
        "Day Charge": np.float,
        "Eve Mins": np.float,
        "Eve Calls": np.int,
        "Eve Charge": np.float,
        "Night Mins": np.float,
        "Night Calls": np.int,
        "Night Charge": np.float,
        "Intl Mins": np.float,
        "Intl Calls": np.int,
        "Intl Charge": np.float,
        "CustServ Calls": np.int,
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


def load_covtype(path="data"):
    from sklearn.datasets import fetch_covtype

    logging.info("Fetching covtype data...")
    tic = time()
    df, y = fetch_covtype(
        data_home=path, download_if_missing=True, return_X_y=True, as_frame=True
    )
    toc = time()
    logging.info("Fetched covtype data in %.2f seconds" % (toc - tic))
    df["y"] = y
    continuous_columns = [col for col in df.columns if col != "y"]
    categorical_columns = None
    dataset = Dataset(
        name="covtype",
        task="multiclass-classification",
        label_column="y",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_default_cb():
    dtype = {
        "LIMIT_BAL": np.int,
        "SEX": "category",
        "EDUCATION": "category",
        "MARRIAGE": "category",
        "AGE": np.int,
        # We consider the PAY_* features as continuous, otherwise some modalities are
        # very rare and lead to problems in train/test splitting
        "PAY_0": np.int,
        "PAY_2": np.int,
        "PAY_3": np.int,
        "PAY_4": np.int,
        "PAY_5": np.int,
        "PAY_6": np.int,
        "BILL_AMT1": np.int,
        "BILL_AMT2": np.int,
        "BILL_AMT3": np.int,
        "BILL_AMT4": np.int,
        "BILL_AMT5": np.int,
        "BILL_AMT6": np.int,
        "PAY_AMT1": np.int,
        "PAY_AMT2": np.int,
        "PAY_AMT3": np.int,
        "PAY_AMT4": np.int,
        "PAY_AMT5": np.int,
        "PAY_AMT6": np.int,
    }
    dataset = Dataset.from_dtype(
        name="default-cb",
        task="binary-classification",
        label_column="default payment next month",
        dtype=dtype,
        drop_columns=["ID"],
    )
    return dataset.load_from_csv("default_cb.csv.gz", dtype=dtype)


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


def load_letter():
    dtype = {
        "X0": np.float,
        "X1": np.float,
        "X2": np.float,
        "X3": np.float,
        "X4": np.float,
        "X5": np.float,
        "X6": np.float,
        "X7": np.float,
        "X8": np.float,
        "X9": np.float,
        "X10": np.float,
        "X11": np.float,
        "X12": np.float,
        "X13": np.float,
        "X14": np.float,
        "X15": np.float,
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
        "X0": np.float,
        "X1": np.float,
        "X2": np.float,
        "X3": np.float,
        "X4": np.float,
        "X5": np.float,
        "X6": np.float,
        "X7": np.float,
        "X8": np.float,
        "X9": np.float,
        "X10": np.float,
        "X11": np.float,
        "X12": np.float,
        "X13": np.float,
        "X14": np.float,
        "X15": np.float,
        "X16": np.float,
        "X17": np.float,
        "X18": np.float,
        "X19": np.float,
        "X20": np.float,
        "X21": np.float,
        "X22": np.float,
        "X23": np.float,
        "X24": np.float,
        "X25": np.float,
        "X26": np.float,
        "X27": np.float,
        "X28": np.float,
        "X29": np.float,
        "X30": np.float,
        "X31": np.float,
        "X32": np.float,
        "X33": np.float,
        "X34": np.float,
        "X35": np.float,
    }
    dataset = Dataset.from_dtype(
        name="satimage", task="multiclass-classification", label_column="y", dtype=dtype
    )
    return dataset.load_from_csv("satimage.csv.gz", dtype=dtype)


def load_sensorless():
    dtype = {
        0: np.float,
        1: np.float,
        2: np.float,
        3: np.float,
        4: np.float,
        5: np.float,
        6: np.float,
        7: np.float,
        8: np.float,
        9: np.float,
        10: np.float,
        11: np.float,
        12: np.float,
        13: np.float,
        14: np.float,
        15: np.float,
        16: np.float,
        17: np.float,
        18: np.float,
        19: np.float,
        20: np.float,
        21: np.float,
        22: np.float,
        23: np.float,
        24: np.float,
        25: np.float,
        26: np.float,
        27: np.float,
        28: np.float,
        29: np.float,
        30: np.float,
        31: np.float,
        32: np.float,
        33: np.float,
        34: np.float,
        35: np.float,
        36: np.float,
        37: np.float,
        38: np.float,
        39: np.float,
        40: np.float,
        41: np.float,
        42: np.float,
        43: np.float,
        44: np.float,
        45: np.float,
        46: np.float,
        47: np.float,
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
        0: np.float,
        1: np.float,
        2: np.float,
        3: np.float,
        4: np.float,
        5: np.float,
        6: np.float,
        7: np.float,
        8: np.float,
        9: np.float,
        10: np.float,
        11: np.float,
        12: np.float,
        13: np.float,
        14: np.float,
        15: np.float,
        16: np.float,
        17: np.float,
        18: np.float,
        19: np.float,
        20: np.float,
        21: np.float,
        22: np.float,
        23: np.float,
        24: np.float,
        25: np.float,
        26: np.float,
        27: np.float,
        28: np.float,
        29: np.float,
        30: np.float,
        31: np.float,
        32: np.float,
        33: np.float,
        34: np.float,
        35: np.float,
        36: np.float,
        37: np.float,
        38: np.float,
        39: np.float,
        40: np.float,
        41: np.float,
        42: np.float,
        43: np.float,
        44: np.float,
        45: np.float,
        46: np.float,
        47: np.float,
        48: np.float,
        49: np.float,
        50: np.float,
        51: np.float,
        52: np.float,
        53: np.float,
        54: np.float,
        55: np.int,
        56: np.int,
    }
    dataset = Dataset.from_dtype(
        name="spambase", task="binary-classification", label_column=57, dtype=dtype,
    )
    return dataset.load_from_csv("spambase.csv.gz", header=None, dtype=dtype)


def load_kddcup(path="data", filename="covtype.csv.gz", random_state=None):
    pass


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
#         print("Loading full KDDCdup dataset (percent10=False)")
#         print("")
#         data, target = fetch_kddcup99(
#             percent10=False, return_X_y=True, random_state=random_state, as_frame=True
#         )  # as_pandas)
#
#         if subsample is not None:
#             print("Subsampling dataset with subsample={}".format(subsample))
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
#             print("Subsampling dataset with subsample={}".format(subsample))
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
        col_n_samples.append(dataset.n_samples_)
        col_n_features.append(dataset.n_features_)
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
    df_descriptions = describe_datasets()
    print(df_descriptions)
