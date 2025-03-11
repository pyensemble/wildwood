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

def _create_dataset(name, task, label_column, continuous_columns=None, categorical_columns=None, df=None):
    """Helper function to create dataset with consistent parameters"""
    dataset = Dataset(
        name=name,
        task=task,
        label_column=label_column,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    if df is not None:
        dataset.df_raw = df
    return dataset

def _load_dataset_from_sklearn(loader_fn, name, task, label_column):
    """Helper function to load datasets from sklearn"""
    data = loader_fn(as_frame=True)
    df = data["frame"]
    continuous_columns = [col for col in df.columns if col != label_column]
    return _create_dataset(name, task, label_column, continuous_columns=continuous_columns, df=df)

def _load_csv_dataset(name, task, filename, dtype, label_column, drop_columns=None, sep=",", decimal=".", raw=False, header='infer'):
    """Helper function to load CSV datasets"""
    if raw:
        return load_raw_dataset(filename, label_column=label_column, dtype=dtype, drop_columns=drop_columns, sep=sep, decimal=decimal, header=header)
    
    dataset = Dataset.from_dtype(name=name, task=task, label_column=label_column, dtype=dtype, drop_columns=drop_columns)
    return dataset.load_from_csv(filename, sep=sep, decimal=decimal, dtype=dtype, header=header)

def load_breastcancer(raw=False):
    from sklearn.datasets import load_breast_cancer
    return _load_dataset_from_sklearn(load_breast_cancer, "breastcancer", "binary-classification", "target")

def load_boston(raw=False):
    dtype = {str(i): np.float64 for i in range(1, 47)}
    return _load_csv_dataset("boston", "regression", "boston.csv.gz", dtype, "medv", raw=raw)

def load_californiahousing(raw=False):
    from sklearn.datasets import fetch_california_housing
    return _load_dataset_from_sklearn(fetch_california_housing, "californiahousing", "regression", "MedHouseVal")

def load_letor(raw=False):
    dtype = {str(i): np.float64 for i in range(1, 47)}
    return _load_csv_dataset("letor", "multiclass-classification", "letor.csv.gz", dtype, "0", raw=raw)

def load_cardio(raw=False):
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
    drop_columns = ["FileName", "Date", "SegFile", "NSP"]
    return _load_csv_dataset("cardio", "multiclass-classification", "cardiotocography.csv.gz", dtype, "CLASS", drop_columns=drop_columns, sep=";", decimal=",", raw=raw)

def load_diabetes_cl():
    dtype = {
        'Pregnancies': np.int32,
        'Glucose': np.float32,
        'BloodPressure': np.float32,
        'SkinThickness': np.float32,
        'Insulin': np.float32,
        'BMI': np.float32,
        'DiabetesPedigreeFunction': np.float32,
        'Age': np.int32,
    }
    return _load_csv_dataset("diabetes_cl", "binary-classification", "diabetes.csv", dtype, "Outcome")

def load_ionosphere():
    from string import ascii_lowercase
    dtype = {"column_"+x: np.float32 for x in list(ascii_lowercase) + ["a" + y for y in ascii_lowercase[:8]]}
    del dtype["column_b"] # this column is completely constant
    drop_columns = ["column_b"]
    return _load_csv_dataset("ionosphere", "binary-classification", "ionosphere_data.xls", dtype, "column_ai", drop_columns=drop_columns)

def load_phoneme():
    dtype = {"V"+str(x): np.float32 for x in range(1, 6)}
    return _load_csv_dataset("phoneme", "binary-classification", "phoneme.csv.gz", dtype, "Class")

def load_wilt2():
    dtype = {x: np.float32 for x in ['GLCM_Pan', 'Mean_G', 'Mean_R', 'Mean_NIR', 'SD_Plan']}
    return _load_csv_dataset("wilt2", "binary-classification", "wilt2.csv.gz", dtype, "class")

def load_banknote():
    dtype = {"V"+str(x): np.float32 for x in range(1, 5)}
    return _load_csv_dataset("banknote", "binary-classification", "banknote.csv.gz", dtype, "Class")

def load_heart():
    dtype = {
        'age': np.int32,
        'sex': "category",
        'cp': "category",
        'trestbps': np.float32,
        'chol': np.float32,
        'fbs': "category",
        'restecg': "category",
        'thalach': np.float32,
        'exang': "category",
        'oldpeak': np.float32,
        'slope': "category",
        'ca': np.int32,
        'thal': "category",
    }
    return _load_csv_dataset("heart", "binary-classification", "heart.xls", dtype, "target")

def load_heart2():
    dtype = {x: np.float32 for x in ['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose',
       'kcm', 'troponin']}
    dtype["age"] = "category"
    return _load_csv_dataset("heart2", "binary-classification", "heart2.csv.gz", dtype, "class")

def load_gamma_particle():
    dtype = {x: np.float32 for x in ['fLength', 'fWidth', 'fSize', 'fConc', 'Conc1', 'Asym', 'M3Long',
       'M3Trans', 'Alpha', 'Dist']}
    drop_columns = ["ID"]
    return _load_csv_dataset("gamma_particle", "binary-classification", "gamma_particle.csv", dtype, "class", drop_columns=drop_columns)

def load_cc_default():
    dtype = {'student': "category", 'balance':np.float32, 'income': np.float32}
    drop_columns = ['Unnamed: 0']
    return _load_csv_dataset("cc_default", "binary-classification", "credit_card_defaulter.xls", dtype, "default", drop_columns=drop_columns)

def load_hcv():
    dtype = {x:np.float32 for x in ['ALB', 'ALP', 'ALT', 'AST',
       'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']}
    dtype['Age'] = np.int32
    dtype['Sex'] = "category"
    drop_columns = ['Unnamed: 0']
    return _load_csv_dataset("hcv", "binary-classification", "hepatitis.csv.gz", dtype, "Category", drop_columns=drop_columns)

def load_smoke():
    dtype = {x: np.float32 for x in ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]',
       'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5',
       'NC0.5', 'NC1.0', 'NC2.5', 'CNT']}
    drop_columns = ['Unnamed: 0', 'UTC']
    return _load_csv_dataset("smoke", "binary-classification", "smoke_detection_iot.csv", dtype, "Fire Alarm", drop_columns=drop_columns)

def load_churn(raw=False):
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
    drop_columns = ["Phone"]
    return _load_csv_dataset("churn", "binary-classification", "churn.csv.gz", dtype, "Churn?", drop_columns=drop_columns, raw=raw)

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
    return _load_dataset_from_sklearn(fetch_covtype, "covtype", "multiclass-classification", "Cover_Type")

def load_diabetes(raw=False):
    from sklearn.datasets import load_diabetes
    return _load_dataset_from_sklearn(load_diabetes, "diabetes", "regression", "target")

def load_kddcup99(raw=False):
    from sklearn.datasets import fetch_kddcup99

    data = fetch_kddcup99(as_frame=True, percent10=False)
    df = data["frame"]
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
    label_column = "labels"

    if raw:
        return get_X_y_from_dataframe(df, label_column=label_column)
    else:
        dataset = Dataset.from_dtype(
            name="kddcup",
            task="multiclass-classification",
            label_column=label_column,
            dtype=dtype,
        )
        dataset.df_raw = df
        return dataset

def load_letter(raw=False):
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
    drop_columns = ["Unnamed: 0"]
    return _load_csv_dataset("letter", "multiclass-classification", "letter.csv.gz", dtype, "y", drop_columns=drop_columns, raw=raw)

def load_satimage(raw=False):
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
    drop_columns = ["Unnamed: 0"]
    return _load_csv_dataset("satimage", "multiclass-classification", "satimage.csv.gz", dtype, "y", drop_columns=drop_columns, raw=raw)

def load_sensorless(raw=False):
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
    return _load_csv_dataset("sensorless", "multiclass-classification", "sensorless.csv.gz", dtype, 48, sep=" ", header=None, raw=raw)

def load_spambase(raw=False):
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
    return _load_csv_dataset("spambase", "binary-classification", "spambase.csv.gz", dtype, 57, header=None, raw=raw)

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

def _collect_dataset_stats(dataset, random_state):
    """Helper function to collect dataset statistics"""
    dataset.one_hot_encode = True
    dataset.standardize = True
    X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
    n_samples_train, n_columns = X_train.shape
    n_samples_test, _ = X_test.shape
    
    return {
        "name": dataset.name,
        "task": dataset.task,
        "n_samples": dataset.n_samples_in_,
        "n_features": dataset.n_features_in_,
        "n_classes": dataset.n_classes_,
        "n_features_categorical": dataset.n_features_categorical_,
        "n_features_continuous": dataset.n_features_continuous_,
        "scaled_gini": dataset.scaled_gini_,
        "n_samples_train": n_samples_train, 
        "n_samples_test": n_samples_test,
        "n_columns": n_columns
    }

def describe_datasets(include="small-classification", random_state=42):
    """Generate dataset descriptions"""
    if include == "small-classification":
        loaders = loaders_small_classification
    elif include == "small-regression":
        loaders = loaders_small_regression
    else:
        raise ValueError("include=%r is not supported for now." % include)

    # Collect stats for all datasets
    stats = [_collect_dataset_stats(loader(), random_state) for loader in loaders]
    
    # Create DataFrame with appropriate columns
    columns = ["dataset", "task", "n_samples", "n_samples_train", "n_samples_test",
              "n_features_cat", "n_features_cont", "n_features", "n_columns"]
    
    if "regression" not in include:
        columns.extend(["n_classes", "scaled_gini"])
        
    df_stats = pd.DataFrame(stats)
    return df_stats[columns]

if __name__ == "__main__":
    # df_descriptions = describe_datasets()
    # print(df_descriptions)
    #
    # datasets = load_covtype()

    load_kddcup99()

    # print(datasets)
