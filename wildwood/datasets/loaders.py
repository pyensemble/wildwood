# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np

from .dataset import Dataset, load_raw_dataset, get_X_y_from_dataframe
from ._adult import load_adult
from ._bank import load_bank
from ._car import load_car
from ._default_cb import load_default_cb

# Helper function to load datasets with common logic
def _load_dataset(name, task, label_column, continuous_columns=None, categorical_columns=None, raw=False, df=None, **kwargs):
    """Helper function to load datasets with common logic."""
    if raw:
        return get_X_y_from_dataframe(df, label_column=label_column)
    else:
        dataset = Dataset(
            name=name,
            task=task,
            label_column=label_column,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )
        dataset.df_raw = df
        return dataset

# Dataset loaders
def load_breastcancer(raw=False):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    return _load_dataset(
        name="breastcancer",
        task="binary-classification",
        label_column="target",
        continuous_columns=[col for col in data["frame"].columns if col != "target"],
        raw=raw,
        df=data["frame"],
    )

def load_boston(raw=False):
    dtype = {str(i): np.float64 for i in range(1, 47)}
    label_column = "medv"
    if raw:
        return load_raw_dataset("boston.csv.gz", label_column=label_column, dtype=dtype)
    else:
        continuous_columns = ["zn", "rad", "crim", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "b", "lstat"]
        categorical_columns = ["chas"]
        return Dataset(
            name="boston",
            task="regression",
            label_column=label_column,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        ).load_from_csv("boston.csv.gz", dtype=dtype)

def load_californiahousing(raw=False):
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    return _load_dataset(
        name="californiahousing",
        task="regression",
        label_column="MedHouseVal",
        continuous_columns=[col for col in data["frame"].columns if col != "MedHouseVal"],
        raw=raw,
        df=data["frame"],
    )

def load_letor(raw=False):
    dtype = {str(i): np.float64 for i in range(1, 47)}
    label_column = "0"
    if raw:
        return load_raw_dataset("letor.csv.gz", label_column=label_column, dtype=dtype)
    else:
        return Dataset.from_dtype(
            name="letor",
            task="multiclass-classification",
            label_column=label_column,
            dtype=dtype,
        ).load_from_csv("letor.csv.gz", dtype=dtype)

def load_cardio(raw=False):
    dtype = {
        "b": np.int32, "e": np.int32, "LBE": np.int32, "LB": np.int32, "AC": np.int32,
        "FM": np.int32, "UC": np.int32, "ASTV": np.int32, "MSTV": np.float32,
        "ALTV": np.int32, "MLTV": np.float32, "DL": np.int32, "DS": np.int32,
        "DP": np.int32, "DR": np.int32, "Width": np.int32, "Min": np.int32,
        "Max": np.int32, "Nmax": np.int32, "Nzeros": np.int32, "Mode": np.int32,
        "Mean": np.int32, "Median": np.int32, "Variance": np.int32, "Tendency": np.int32,
        "A": np.int32, "B": np.int32, "C": np.int32, "D": np.int32, "E": np.int32,
        "AD": np.int32, "DE": np.int32, "LD": np.int32, "FS": np.int32, "SUSP": np.int32,
    }
    label_column = "CLASS"
    drop_columns = ["FileName", "Date", "SegFile", "NSP"]
    if raw:
        return load_raw_dataset(
            "cardiotocography.csv.gz",
            label_column=label_column,
            drop_columns=drop_columns,
            dtype=dtype,
            sep=";",
            decimal=",",
        )
    else:
        return Dataset.from_dtype(
            name="cardio",
            task="multiclass-classification",
            label_column=label_column,
            dtype=dtype,
            drop_columns=drop_columns,
        ).load_from_csv("cardiotocography.csv.gz", sep=";", decimal=",", dtype=dtype)

def load_diabetes_cl():
    dtype = {
        'Pregnancies': np.int32, 'Glucose': np.float32, 'BloodPressure': np.float32,
        'SkinThickness': np.float32, 'Insulin': np.float32, 'BMI': np.float32,
        'DiabetesPedigreeFunction': np.float32, 'Age': np.int32,
    }
    return Dataset.from_dtype(
        name="diabetes_cl",
        task="binary-classification",
        label_column='Outcome',
        dtype=dtype,
    ).load_from_csv("diabetes.csv", dtype=dtype)

def load_ionosphere():
    from string import ascii_lowercase
    dtype = {"column_" + x: np.float32 for x in list(ascii_lowercase) + ["a" + y for y in ascii_lowercase[:8]]}
    del dtype["column_b"]  # Constant column
    return Dataset.from_dtype(
        name="ionosphere",
        task="binary-classification",
        label_column='column_ai',
        drop_columns=["column_b"],
        dtype=dtype,
    ).load_from_csv("ionosphere_data.xls", dtype=dtype)

def load_phoneme():
    dtype = {"V" + str(x): np.float32 for x in range(1, 6)}
    return Dataset.from_dtype(
        name="phoneme",
        task="binary-classification",
        label_column='Class',
        dtype=dtype,
    ).load_from_csv("phoneme.csv.gz", dtype=dtype)

def load_wilt2():
    dtype = {x: np.float32 for x in ['GLCM_Pan', 'Mean_G', 'Mean_R', 'Mean_NIR', 'SD_Plan']}
    return Dataset.from_dtype(
        name="wilt2",
        task="binary-classification",
        label_column='class',
        dtype=dtype,
    ).load_from_csv("wilt2.csv.gz", dtype=dtype)

def load_banknote():
    dtype = {"V" + str(x): np.float32 for x in range(1, 5)}
    return Dataset.from_dtype(
        name="banknote",
        task="binary-classification",
        label_column='Class',
        dtype=dtype,
    ).load_from_csv("banknote.csv.gz", dtype=dtype)

def load_heart():
    dtype = {
        'age': np.int32, 'sex': "category", 'cp': "category", 'trestbps': np.float32,
        'chol': np.float32, 'fbs': "category", 'restecg': "category", 'thalach': np.float32,
        'exang': "category", 'oldpeak': np.float32, 'slope': "category", 'ca': np.int32,
        'thal': "category",
    }
    return Dataset.from_dtype(
        name="heart",
        task="binary-classification",
        label_column='target',
        dtype=dtype,
    ).load_from_csv("heart.xls", dtype=dtype)

def load_heart2():
    dtype = {x: np.float32 for x in ['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']}
    dtype["age"] = "category"
    return Dataset.from_dtype(
        name="heart2",
        task="binary-classification",
        label_column='class',
        dtype=dtype,
    ).load_from_csv("heart2.csv.gz", dtype=dtype)

def load_gamma_particle():
    dtype = {x: np.float32 for x in ['fLength', 'fWidth', 'fSize', 'fConc', 'Conc1', 'Asym', 'M3Long', 'M3Trans', 'Alpha', 'Dist']}
    return Dataset.from_dtype(
        name="gamma_particle",
        task="binary-classification",
        label_column='class',
        drop_columns=["ID"],
        dtype=dtype,
    ).load_from_csv("gamma_particle.csv", dtype=dtype)

def load_cc_default():
    dtype = {'student': "category", 'balance': np.float32, 'income': np.float32}
    return Dataset.from_dtype(
        name="cc_default",
        task="binary-classification",
        label_column='default',
        drop_columns=['Unnamed: 0'],
        dtype=dtype,
    ).load_from_csv("credit_card_defaulter.xls", dtype=dtype)

def load_hcv():
    dtype = {x: np.float32 for x in ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']}
    dtype['Age'] = np.int32
    dtype['Sex'] = "category"
    return Dataset.from_dtype(
        name="hcv",
        task="binary-classification",
        label_column='Category',
        drop_columns=['Unnamed: 0'],
        dtype=dtype,
    ).load_from_csv("hepatitis.csv.gz", dtype=dtype)

def load_smoke():
    dtype = {x: np.float32 for x in ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT']}
    return Dataset.from_dtype(
        name="smoke",
        task="binary-classification",
        label_column='Fire Alarm',
        drop_columns=['Unnamed: 0', 'UTC'],
        dtype=dtype,
    ).load_from_csv("smoke_detection_iot.csv", dtype=dtype)

def load_churn(raw=False):
    dtype = {
        "State": "category", "Account Length": np.int32, "Area Code": "category",
        "Int'l Plan": "category", "VMail Plan": "category", "VMail Message": np.int32,
        "Day Mins": np.float32, "Day Calls": np.int32, "Day Charge": np.float32,
        "Eve Mins": np.float32, "Eve Calls": np.int32, "Eve Charge": np.float32,
        "Night Mins": np.float32, "Night Calls": np.int32, "Night Charge": np.float32,
        "Intl Mins": np.float32, "Intl Calls": np.int32, "Intl Charge": np.float32,
        "CustServ Calls": np.int32,
    }
    label_column = "Churn?"
    drop_columns = ["Phone"]
    if raw:
        return load_raw_dataset("churn.csv.gz", label_column=label_column, drop_columns=drop_columns, dtype=dtype)
    else:
        return Dataset.from_dtype(
            name="churn",
            task="binary-classification",
            label_column=label_column,
            dtype=dtype,
            drop_columns=drop_columns,
        ).load_from_csv("churn.csv.gz", dtype=dtype)

# Additional dataset loaders (e.g., load_covtype, load_diabetes, etc.) can be refactored similarly.

# List of dataset loaders
loaders_small_classification = [
    load_adult, load_bank, load_breastcancer, load_car, load_cardio, load_churn,
    load_default_cb, load_letter, load_satimage, load_sensorless, load_spambase,
]

loaders_small_regression = [load_boston, load_californiahousing, load_diabetes]
loaders_medium = [load_covtype]
loaders_large = []

# Function to describe datasets
def describe_datasets(include="small-classification", random_state=42):
    if include == "small-classification":
        loaders = loaders_small_classification
    elif include == "small-regression":
        loaders = loaders_small_regression
    else:
        raise ValueError("include=%r is not supported for now." % include)

    col_name, col_n_samples, col_n_features, col_task, col_n_classes = [], [], [], [], []
    col_n_features_categorical, col_n_features_continuous = [], []
    col_scaled_gini, col_n_samples_train, col_n_samples_test, col_n_columns = [], [], [], []

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
        return pd.DataFrame({
            "dataset": col_name, "task": col_task, "n_samples": col_n_samples,
            "n_samples_train": col_n_samples_train, "n_samples_test": col_n_samples_test,
            "n_features_cat": col_n_features_categorical, "n_features_cont": col_n_features_continuous,
            "n_features": col_n_features, "n_columns": col_n_columns,
        })
    else:
        return pd.DataFrame({
            "dataset": col_name, "task": col_task, "n_samples": col_n_samples,
            "n_samples_train": col_n_samples_train, "n_samples_test": col_n_samples_test,
            "n_features_cat": col_n_features_categorical, "n_features_cont": col_n_features_continuous,
            "n_features": col_n_features, "n_classes": col_n_classes,
            "n_columns": col_n_columns, "scaled_gini": col_scaled_gini,
        })

if __name__ == "__main__":
    # Example usage
    load_kddcup99()