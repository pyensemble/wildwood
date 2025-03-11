# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np

from .dataset import Dataset, load_raw_dataset, get_X_y_from_dataframe
from ._adult import load_adult
from ._bank import load_bank
from ._car import load_car
from ._default_cb import load_default_cb

# TODO: kdd98 https://www.openml.org/d/23513, Il y a plein de features numeriques avec un grand nombre de "missing" values

def _load_dataset(name, task, filename, label_column, dtype=None, drop_columns=None, sep=",", decimal=".", raw=False, header='infer'):
    """Helper function to load datasets with consistent logic."""
    if raw:
        return load_raw_dataset(filename, label_column=label_column, dtype=dtype, drop_columns=drop_columns, sep=sep, decimal=decimal, header=header)
    dataset = Dataset.from_dtype(name=name, task=task, label_column=label_column, dtype=dtype, drop_columns=drop_columns)
    return dataset.load_from_csv(filename, sep=sep, decimal=decimal, dtype=dtype, header=header)

def _load_sklearn_dataset(name, task, load_function, label_column):
    """Helper function for loading datasets from sklearn."""
    data = load_function(as_frame=True)
    df = data["frame"]
    continuous_columns = [col for col in df.columns if col != label_column]
    dataset = Dataset(
        name=name,
        task=task,
        label_column=label_column,
        continuous_columns=continuous_columns,
        categorical_columns=None,
    )
    dataset.df_raw = df
    return dataset

def load_breastcancer(raw=False):
    from sklearn.datasets import load_breast_cancer
    return _load_sklearn_dataset("breastcancer", "binary-classification", load_breast_cancer, "target")

def load_boston(raw=False):
    dtype = {"zn": np.float64, "rad": np.float64, "crim": np.float64, "indus": np.float64,
             "nox": np.float64, "rm": np.float64, "age": np.float64, "dis": np.float64,
             "tax": np.float64, "ptratio": np.float64, "b": np.float64, "lstat": np.float64,
             "chas": "category"}
    return _load_dataset(
        name="boston",
        task="regression",
        filename="boston.csv.gz",
        label_column="medv",
        dtype=dtype,
        raw=raw,
    )

def load_californiahousing(raw=False):
    from sklearn.datasets import fetch_california_housing
    return _load_sklearn_dataset("californiahousing", "regression", fetch_california_housing, "MedHouseVal")

def load_letor(raw=False):
    dtype = {str(i): np.float64 for i in range(1, 47)}
    return _load_dataset(
        name="letor",
        task="multiclass-classification",
        filename="letor.csv.gz",
        label_column="0",
        dtype=dtype,
        raw=raw,
    )

def load_cardio(raw=False):
    dtype = {"b": np.int32, "e": np.int32, "LBE": np.int32, "LB": np.int32, "AC": np.int32, "FM": np.int32,
             "UC": np.int32, "ASTV": np.int32, "MSTV": np.float32, "ALTV": np.int32, "MLTV": np.float32,
             "DL": np.int32, "DS": np.int32, "DP": np.int32, "DR": np.int32, "Width": np.int32, "Min": np.int32,
             "Max": np.int32, "Nmax": np.int32, "Nzeros": np.int32, "Mode": np.int32, "Mean": np.int32,
             "Median": np.int32, "Variance": np.int32, "Tendency": np.int32, "A": np.int32, "B": np.int32,
             "C": np.int32, "D": np.int32, "E": np.int32, "AD": np.int32, "DE": np.int32, "LD": np.int32,
             "FS": np.int32, "SUSP": np.int32}
    return _load_dataset(
        name="cardio",
        task="multiclass-classification",
        filename="cardiotocography.csv.gz",
        label_column="CLASS",
        dtype=dtype,
        drop_columns=["FileName", "Date", "SegFile", "NSP"],
        sep=";",
        decimal=",",
        raw=raw,
    )

def load_diabetes_cl():
    dtype = {'Pregnancies': np.int32, 'Glucose': np.float32, 'BloodPressure': np.float32,
             'SkinThickness': np.float32, 'Insulin': np.float32, 'BMI': np.float32,
             'DiabetesPedigreeFunction': np.float32, 'Age': np.int32}
    return _load_dataset(
        name="diabetes_cl",
        task="binary-classification",
        filename="diabetes.csv",
        label_column="Outcome",
        dtype=dtype,
    )

def load_ionosphere():
    from string import ascii_lowercase
    dtype = {"column_"+x: np.float32 for x in list(ascii_lowercase) + ["a" + y for y in ascii_lowercase[:8]]}
    del dtype["column_b"]
    return _load_dataset(
        name="ionosphere",
        task="binary-classification",
        filename="ionosphere_data.xls",
        label_column="column_ai",
        dtype=dtype,
        drop_columns=["column_b"],
    )

def load_phoneme():
    dtype = {"V"+str(x): np.float32 for x in range(1, 6)}
    return _load_dataset(
        name="phoneme",
        task="binary-classification",
        filename="phoneme.csv.gz",
        label_column="Class",
        dtype=dtype,
    )

def load_wilt2():
    dtype = {x: np.float32 for x in ['GLCM_Pan', 'Mean_G', 'Mean_R', 'Mean_NIR', 'SD_Plan']}
    return _load_dataset(
        name="wilt2",
        task="binary-classification",
        filename="wilt2.csv.gz",
        label_column="class",
        dtype=dtype,
    )

def load_banknote():
    dtype = {"V"+str(x): np.float32 for x in range(1, 5)}
    return _load_dataset(
        name="banknote",
        task="binary-classification",
        filename="banknote.csv.gz",
        label_column="Class",
        dtype=dtype,
    )

def load_heart():
    dtype = {'age': np.int32, 'sex': "category", 'cp': "category", 'trestbps': np.float32, 'chol': np.float32,
             'fbs': "category", 'restecg': "category", 'thalach': np.float32, 'exang': "category",
             'oldpeak': np.float32, 'slope': "category", 'ca': np.int32, 'thal': "category"}
    return _load_dataset(
        name="heart",
        task="binary-classification",
        filename="heart.xls",
        label_column="target",
        dtype=dtype,
    )

def load_heart2():
    dtype = {'age': "category", 'gender': np.float32, 'impluse': np.float32, 'pressurehight': np.float32,
             'pressurelow': np.float32, 'glucose': np.float32, 'kcm': np.float32, 'troponin': np.float32}
    return _load_dataset(
        name="heart2",
        task="binary-classification",
        filename="heart2.csv.gz",
        label_column="class",
        dtype=dtype,
    )

def load_gamma_particle():
    dtype = {x: np.float32 for x in ['fLength', 'fWidth', 'fSize', 'fConc', 'Conc1', 'Asym', 'M3Long',
             'M3Trans', 'Alpha', 'Dist']}
    return _load_dataset(
        name="gamma_particle",
        task="binary-classification",
        filename="gamma_particle.csv",
        label_column="class",
        dtype=dtype,
        drop_columns=["ID"],
    )

def load_cc_default():
    dtype = {'student': "category", 'balance': np.float32, 'income': np.float32}
    return _load_dataset(
        name="cc_default",
        task="binary-classification",
        filename="credit_card_defaulter.xls",
        label_column="default",
        dtype=dtype,
        drop_columns=['Unnamed: 0'],
    )

def load_hcv():
    dtype = {x: np.float32 for x in ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']}
    dtype['Age'] = np.int32
    dtype['Sex'] = "category"
    return _load_dataset(
        name="hcv",
        task="binary-classification",
        filename="hepatitis.csv.gz",
        label_column="Category",
        dtype=dtype,
        drop_columns=['Unnamed: 0'],
    )

def load_smoke():
    dtype = {x: np.float32 for x in ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2',
             'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT']}
    return _load_dataset(
        name="smoke",
        task="binary-classification",
        filename="smoke_detection_iot.csv",
        label_column="Fire Alarm",
        dtype=dtype,
        drop_columns=['Unnamed: 0', 'UTC'],
    )

def load_churn(raw=False):
    dtype = {"State": "category", "Account Length": np.int32, "Area Code": "category", "Int'l Plan": "category",
             "VMail Plan": "category", "VMail Message": np.int32, "Day Mins": np.float32, "Day Calls": np.int32,
             "Day Charge": np.float32, "Eve Mins": np.float32, "Eve Calls": np.int32, "Eve Charge": np.float32,
             "Night Mins": np.float32, "Night Calls": np.int32, "Night Charge": np.float32, "Intl Mins": np.float32,
             "Intl Calls": np.int32, "Intl Charge": np.float32, "CustServ Calls": np.int32}
    return _load_dataset(
        name="churn",
        task="binary-classification",
        filename="churn.csv.gz",
        label_column="Churn?",
        dtype=dtype,
        drop_columns=["Phone"],
        raw=raw,
    )

def load_epsilon_catboost(raw=False):
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

def load_covtype(raw=False):
    from sklearn.datasets import fetch_covtype
    return _load_sklearn_dataset("covtype", "multiclass-classification", fetch_covtype, "Cover_Type")

def load_diabetes(raw=False):
    from sklearn.datasets import load_diabetes
    return _load_sklearn_dataset("diabetes", "regression", load_diabetes, "target")

def load_kddcup99(raw=False):
    from sklearn.datasets import fetch_kddcup99
    data = fetch_kddcup99(as_frame=True, percent10=False)
    df = data["frame"]
    dtype = {"duration": np.float32, "protocol_type": "category", "service": "category", "flag": "category",
             "src_bytes": np.float32, "dst_bytes": np.float32, "land": "category", "wrong_fragment": np.float32,
             "urgent": np.float32, "hot": np.float32, "num_failed_logins": np.float32, "logged_in": "category",
             "num_compromised": np.float32, "root_shell": "category", "su_attempted": "category",
             "num_root": np.float32, "num_file_creations": np.float32, "num_shells": np.float32,
             "num_access_files": np.float32, "num_outbound_cmds": np.float32, "is_host_login": "category",
             "is_guest_login": "category", "count": np.float32, "srv_count": np.float32, "serror_rate": np.float32,
             "srv_serror_rate": np.float32, "rerror_rate": np.float32, "srv_rerror_rate": np.float32,
             "same_srv_rate": np.float32, "diff_srv_rate": np.float32, "srv_diff_host_rate": np.float32,
             "dst_host_count": np.float32, "dst_host_srv_count": np.float32, "dst_host_same_srv_rate": np.float32,
             "dst_host_diff_srv_rate": np.float32, "dst_host_same_src_port_rate": np.float32,
             "dst_host_srv_diff_host_rate": np.float32, "dst_host_serror_rate": np.float32,
             "dst_host_srv_serror_rate": np.float32, "dst_host_rerror_rate": np.float32,
             "dst_host_srv_rerror_rate": np.float32}
    df = df.astype(dtype)
    return _load_dataset(
        name="kddcup",
        task="multiclass-classification",
        filename="kddcup99.csv.gz",
        label_column="labels",
        dtype=dtype,
        raw=raw,
    )

def load_letter(raw=False):
    dtype = {"X0": np.float32, "X1": np.float32, "X2": np.float32, "X3": np.float32, "X4": np.float32,
             "X5": np.float32, "X6": np.float32, "X7": np.float32, "X8": np.float32, "X9": np.float32,
             "X10": np.float32, "X11": np.float32, "X12": np.float32, "X13": np.float32, "X14": np.float32,
             "X15": np.float32}
    return _load_dataset(
        name="letter",
        task="multiclass-classification",
        filename="letter.csv.gz",
        label_column="y",
        dtype=dtype,
        drop_columns=["Unnamed: 0"],
        raw=raw,
    )

def load_satimage(raw=False):
    dtype = {"X0": np.float32, "X1": np.float32, "X2": np.float32, "X3": np.float32, "X4": np.float32,
             "X5": np.float32, "X6": np.float32, "X7": np.float32, "X8": np.float32, "X9": np.float32,
             "X10": np.float32, "X11": np.float32, "X12": np.float32, "X13": np.float32, "X14": np.float32,
             "X15": np.float32, "X16": np.float32, "X17": np.float32, "X18": np.float32, "X19": np.float32,
             "X20": np.float32, "X21": np.float32, "X22": np.float32, "X23": np.float32, "X24": np.float32,
             "X25": np.float32, "X26": np.float32, "X27": np.float32, "X28": np.float32, "X29": np.float32,
             "X30": np.float32, "X31": np.float32, "X32": np.float32, "X33": np.float32, "X34": np.float32,
             "X35": np.float32}
    return _load_dataset(
        name="satimage",
        task="multiclass-classification",
        filename="satimage.csv.gz",
        label_column="y",
        dtype=dtype,
        drop_columns=["Unnamed: 0"],
        raw=raw,
    )

def load_sensorless(raw=False):
    dtype = {i: np.float32 for i in range(48)}
    return _load_dataset(
        name="sensorless",
        task="multiclass-classification",
        filename="sensorless.csv.gz",
        label_column=48,
        dtype=dtype,
        sep=" ",
        header=None,
        raw=raw,
    )

def load_spambase(raw=False):
    dtype = {i: np.float32 for i in range(57)}
    dtype[56] = np.int32
    dtype[55] = np.int32
    return _load_dataset(
        name="spambase",
        task="binary-classification",
        filename="spambase.csv.gz",
        label_column=57,
        dtype=dtype,
        header=None,
        raw=raw,
    )

def load_newsgroup():
    pass

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
