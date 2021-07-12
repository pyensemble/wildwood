# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

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
        drop=None,  # "first",
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
                            OneHotEncoder(
                                drop=self.drop,
                                sparse=self.sparse,
                                handle_unknown="ignore",
                            ),
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
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
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
            categorical_features[-self.n_features_categorical_ :] = True
            self.categorical_features_ = categorical_features

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
                            + "#" + str(idx_modality)
                            for idx_modality, modality in enumerate(modalities)
                        ]
                    )
            else:
                columns.extend(self.categorical_columns_)
        self.columns_ = columns

        if self.pd_df_categories:
            # columns = (self.continuous_columns or [])+(self.categorical_columns or [])
            X_train = pd.DataFrame(X_train, columns=columns)
            X_test = pd.DataFrame(X_test, columns=columns)
            if self.categorical_columns is not None:
                X_train[self.categorical_columns] = (
                    X_train[self.categorical_columns].astype(int).astype("category")
                )
                X_test[self.categorical_columns] = (
                    X_test[self.categorical_columns].astype(int).astype("category")
                )

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

    def compute_gini(self):
        return 1 - ((np.bincount(LabelEncoder().fit_transform(self.df_raw[self.label_column]))/len(self.df_raw))**2).sum()
