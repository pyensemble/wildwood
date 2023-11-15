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


def load_breastcancer(raw=False):
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data["frame"]

    label_column = "target"
    if raw:
        return get_X_y_from_dataframe(df, label_column=label_column)
    else:
        continuous_columns = [col for col in df.columns if col != label_column]
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


def load_boston(raw=False):
    dtype = {str(i): np.float64 for i in range(1, 47)}
    label_column = "medv"
    if raw:
        X, y = load_raw_dataset("boston.csv.gz", label_column=label_column, dtype=dtype)
        return X.astype({"chas": "category"}), y
    else:
        continuous_columns = [
            "zn",
            "rad",
            "crim",
            "indus",
            "nox",
            "rm",
            "age",
            "dis",
            "tax",
            "ptratio",
            "b",
            "lstat",
        ]
        categorical_columns = ["chas"]
        dataset = Dataset(
            name="boston",
            task="regression",
            label_column=label_column,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )
        # dataset.df_raw = df
        # return dataset
        # dataset = Dataset.from_dtype(
        #     name="boston",
        #     task="regression",
        #     label_column=label_column,
        #     dtype=dtype,
        # )
        return dataset.load_from_csv("boston.csv.gz", dtype=dtype)


# def load_boston(raw=False):
#     from sklearn.datasets import load_boston
#
#     data = load_boston()
#     # Load as a dataframe. We set some features as categorical so that we have a
#     # regression example with categorical features for tests...
#     df = pd.DataFrame(data["data"], columns=data["feature_names"]).astype(
#         {"CHAS": "category"}
#     )
#     label_column = "target"
#     df[label_column] = data[label_column]
#
#     if raw:
#         return get_X_y_from_dataframe(df, label_column=label_column)
#     else:
#         continuous_columns = [
#             "ZN",
#             "RAD",
#             "CRIM",
#             "INDUS",
#             "NOX",
#             "RM",
#             "AGE",
#             "DIS",
#             "TAX",
#             "PTRATIO",
#             "B",
#             "LSTAT",
#         ]
#         categorical_columns = ["CHAS"]
#         dataset = Dataset(
#             name="boston",
#             task="regression",
#             label_column=label_column,
#             continuous_columns=continuous_columns,
#             categorical_columns=categorical_columns,
#         )
#         dataset.df_raw = df
#         return dataset


def load_californiahousing(raw=False):
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data["frame"]
    label_column = "MedHouseVal"

    if raw:
        return get_X_y_from_dataframe(df, label_column=label_column)
    else:
        continuous_columns = [col for col in df.columns if col != label_column]
        categorical_columns = None
        dataset = Dataset(
            name="californiahousing",
            task="regression",
            label_column=label_column,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )
        dataset.df_raw = df
        return dataset


def load_letor(raw=False):
    dtype = {str(i): np.float64 for i in range(1, 47)}
    label_column = "0"
    if raw:
        return load_raw_dataset("letor.csv.gz", label_column=label_column, dtype=dtype,)
    else:
        dataset = Dataset.from_dtype(
            name="letor",
            task="multiclass-classification",
            label_column=label_column,
            dtype=dtype,
        )
        return dataset.load_from_csv("letor.csv.gz", dtype=dtype)


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
    label_column = "CLASS"
    drop_columns = ["FileName", "Date", "SegFile", "NSP"]
    filename = "cardiotocography.csv.gz"
    if raw:
        return load_raw_dataset(
            filename,
            label_column=label_column,
            drop_columns=drop_columns,
            dtype=dtype,
            sep=";",
            decimal=",",
        )
    else:
        dataset = Dataset.from_dtype(
            name="cardio",
            task="multiclass-classification",
            label_column=label_column,
            dtype=dtype,
            # We drop the NSP column which is a 3-class version of the label
            drop_columns=drop_columns,
        )
        return dataset.load_from_csv(filename, sep=";", decimal=",", dtype=dtype)


#
# def load_amazon():
#     from catboost.datasets import amazon
#
#     df_train, df_test = amazon()
#     df = pd.concat([df_train, df_test], axis="column")
#
#     df.info()

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
    # downloaded from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    label_column = 'Outcome'
    filename = "diabetes.csv"

    dataset = Dataset.from_dtype(
        name="diabetes_cl",
        task="binary-classification",
        label_column=label_column,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)

def load_ionosphere():
    from string import ascii_lowercase
    dtype = {"column_"+x: np.float32 for x in list(ascii_lowercase) + ["a" + y for y in ascii_lowercase[:8]]}
    del dtype["column_b"] # this column is completely constant

    # downloaded from https://www.kaggle.com/datasets/prashant111/ionosphere?select=ionosphere_data.csv
    label_column = 'column_ai'
    drop_columns = ["column_b"]
    filename = "ionosphere_data.xls"

    dataset = Dataset.from_dtype(
        name="ionosphere",
        task="binary-classification",
        drop_columns=drop_columns,
        label_column=label_column,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)

def load_phoneme():

    dtype = {"V"+str(x): np.float32 for x in range(1, 6)}

    # downloaded from https://www.openml.org/search?type=data&sort=runs&status=active&format=ARFF&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=lte_10&id=1489
    label_column = 'Class'
    filename = "phoneme.csv.gz"

    dataset = Dataset.from_dtype(
        name="phoneme",
        task="binary-classification",
        label_column=label_column,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)


def load_wilt2():

    dtype = {x: np.float32 for x in ['GLCM_Pan', 'Mean_G', 'Mean_R', 'Mean_NIR', 'SD_Plan']}
    # downloaded from https://www.openml.org/search?type=data&sort=runs&status=active&format=ARFF&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=lte_10&id=40983
    label_column = 'class'
    filename = "wilt2.csv.gz"

    dataset = Dataset.from_dtype(
        name="wilt2",
        task="binary-classification",
        label_column=label_column,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)

def load_banknote():

    dtype = {"V"+str(x): np.float32 for x in range(1, 5)}

    # downloaded from https://www.openml.org/search?type=data&sort=runs&status=active&format=ARFF&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=lte_10&id=1462
    label_column = 'Class'
    filename = "banknote.csv.gz"

    dataset = Dataset.from_dtype(
        name="banknote",
        task="binary-classification",
        label_column=label_column,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)

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
    # downloaded from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?select=heart.csv
    label_column = 'target'
    filename = "heart.xls"

    dataset = Dataset.from_dtype(
        name="heart",
        task="binary-classification",
        label_column=label_column,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)

def load_heart2():

    dtype = {x: np.float32 for x in ['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose',
       'kcm', 'troponin']}
    dtype["age"] = "category"
    # downloaded from https://data.mendeley.com/datasets/65gxgy2nmg
    label_column = 'class'
    filename = "heart2.csv.gz"

    dataset = Dataset.from_dtype(
        name="heart2",
        task="binary-classification",
        label_column=label_column,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)

def load_gamma_particle():

    dtype = {x: np.float32 for x in ['fLength', 'fWidth', 'fSize', 'fConc', 'Conc1', 'Asym', 'M3Long',
       'M3Trans', 'Alpha', 'Dist']}

    # downloaded from https://www.kaggle.com/datasets/ppb00x/find-gamma-particles-in-magic-telescope
    label_column = 'class'
    drop_columns = ["ID"]
    filename = "gamma_particle.csv"

    dataset = Dataset.from_dtype(
        name="gamma_particle",
        task="binary-classification",
        label_column=label_column,
        drop_columns=drop_columns,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)


def load_cc_default():

    dtype = {'student': "category", 'balance':np.float32, 'income': np.float32}

    # downloaded from https://www.kaggle.com/datasets/d4rklucif3r/defaulter
    label_column = 'default'
    drop_columns = ['Unnamed: 0']
    filename = "credit_card_defaulter.xls"

    dataset = Dataset.from_dtype(
        name="cc_default",
        task="binary-classification",
        label_column=label_column,
        drop_columns=drop_columns,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)


def load_hcv():

    dtype = {x:np.float32 for x in ['ALB', 'ALP', 'ALT', 'AST',
       'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']}
    dtype['Age'] = np.int32
    dtype['Sex'] = "category"

    # downloaded from https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset
    # Values in column 'Category' which are different from '0=Blood Donor' were replaced by '1=Pathology'

    label_column = 'Category'
    drop_columns = ['Unnamed: 0']
    filename = "hepatitis.csv.gz"

    dataset = Dataset.from_dtype(
        name="hcv",
        task="binary-classification",
        label_column=label_column,
        drop_columns=drop_columns,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)

def load_smoke():

    dtype = {x: np.float32 for x in ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]',
       'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5',
       'NC0.5', 'NC1.0', 'NC2.5', 'CNT']}

    # downloaded from https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset
    label_column = 'Fire Alarm'
    drop_columns = ['Unnamed: 0', 'UTC']
    filename = "smoke_detection_iot.csv"

    dataset = Dataset.from_dtype(
        name="smoke",
        task="binary-classification",
        label_column=label_column,
        drop_columns=drop_columns,
        dtype=dtype,
    )
    return dataset.load_from_csv(filename, dtype=dtype)

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

    label_column = "Churn?"
    drop_columns = ["Phone"]
    filename = "churn.csv.gz"

    if raw:
        return load_raw_dataset(
            filename, label_column=label_column, drop_columns=drop_columns, dtype=dtype,
        )
    else:
        dataset = Dataset.from_dtype(
            name="churn",
            task="binary-classification",
            label_column=label_column,
            dtype=dtype,
            drop_columns=drop_columns,
        )
        return dataset.load_from_csv(filename, dtype=dtype)


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

    data = fetch_covtype(as_frame=True)
    df = data["frame"]
    label_column = "Cover_Type"
    if raw:
        return get_X_y_from_dataframe(df, label_column=label_column,)
    else:
        continuous_columns = [col for col in df.columns if col != label_column]
        categorical_columns = None
        dataset = Dataset(
            name="covtype",
            task="multiclass-classification",
            label_column=label_column,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )
        dataset.df_raw = df
        return dataset


def load_diabetes(raw=False):
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    df = data["frame"]
    label_column = "target"
    if raw:
        return get_X_y_from_dataframe(df, label_column=label_column)
    else:
        continuous_columns = [col for col in df.columns if col != label_column]
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


def load_kddcup99(raw=False):
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
    filename = "letter.csv.gz"
    drop_columns = ["Unnamed: 0"]
    label_column = "y"

    if raw:
        return load_raw_dataset(
            filename, label_column=label_column, drop_columns=drop_columns, dtype=dtype,
        )
    else:
        dataset = Dataset.from_dtype(
            name="letter",
            task="multiclass-classification",
            label_column=label_column,
            dtype=dtype,
            drop_columns=drop_columns,
        )
        return dataset.load_from_csv(filename, dtype=dtype)


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
    filename = "satimage.csv.gz"
    label_column = "y"
    drop_columns = ["Unnamed: 0"]
    if raw:
        return load_raw_dataset(
            filename, label_column=label_column, drop_columns=drop_columns, dtype=dtype,
        )
    else:
        dataset = Dataset.from_dtype(
            name="satimage",
            task="multiclass-classification",
            label_column=label_column,
            drop_columns=drop_columns,
            dtype=dtype,
        )
        return dataset.load_from_csv(filename, dtype=dtype)


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
    filename = "sensorless.csv.gz"
    label_column = 48
    if raw:
        return load_raw_dataset(
            filename, label_column=label_column, dtype=dtype, sep=" ", header=None
        )
    else:
        dataset = Dataset.from_dtype(
            name="sensorless",
            task="multiclass-classification",
            label_column=label_column,
            dtype=dtype,
        )
        return dataset.load_from_csv(filename, sep=" ", header=None, dtype=dtype)


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
    filename = "spambase.csv.gz"
    label_column = 57
    if raw:
        return load_raw_dataset(
            filename, label_column=label_column, dtype=dtype, header=None
        )
    else:
        dataset = Dataset.from_dtype(
            name="spambase", task="binary-classification", label_column=57, dtype=dtype
        )
        return dataset.load_from_csv(filename, header=None, dtype=dtype)


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
