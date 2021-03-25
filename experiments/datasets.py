import os
from gzip import GzipFile
from time import time


import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, average_precision_score, mean_squared_error

from sklearn.preprocessing import MinMaxScaler

def evaluate_classifier(clf, test_data, test_target, binary=True):
    tic = time()

    predicted_proba_test = clf.predict_proba(test_data)
    toc = time()

    print(f"predicted in {toc - tic:.3f}s")


    predicted_test = np.argmax(predicted_proba_test, axis=1)

    if binary:
        roc_auc = roc_auc_score(test_target, predicted_proba_test[:, 1], multi_class="ovo")
        avg_precision_score = average_precision_score(test_target, predicted_proba_test[:, 1])
    else:
        onehot_target_test = onehotencode(test_target)

        try:
            roc_auc = roc_auc_score(onehot_target_test, predicted_proba_test, multi_class="ovo")
        except ValueError as verror:
            print("FAILED to compute roc auc with following error : ")
            print(verror)
            roc_auc = 0.0
        avg_precision_score = average_precision_score(onehot_target_test, predicted_proba_test)

    acc = accuracy_score(test_target, predicted_test)
    log_loss_value = log_loss(test_target, predicted_proba_test, labels=list(range(predicted_proba_test.shape[1])) if not binary else None)
    print(f"ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")
    print("ROC AUC computed with multi_class='ovo' (see sklearn docs)")

    print(f"Log loss: {log_loss_value :.4f}")

    print(f"Average precision score: {avg_precision_score :.4f}")

def evaluate_regressor(reg, test_data, test_target):
    tic = time()

    predicted_test = reg.predict(test_data)
    toc = time()

    print(f"predicted in {toc - tic:.3f}s")

    mse = mean_squared_error(test_target, predicted_test)

    print(f"Mean squared error: {mse :.4f}")


def onehotencode(y):
    encoded = np.zeros((len(y), np.max(y).astype(int)+1))
    encoded[np.arange(len(y)), y] = 1
    return encoded

class Datasets:

    def __init__(self):
        pass

    def split_train_test(self, test_split, random_state):
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.data, self.target, test_size=test_split, random_state=random_state)

    
    #def get_test_size(self, test_split):
    #        if test_split <= 0:
    #            raise Error("test_split should be positive")
    #    if type(test_split) == int:
    #        return test_split
    #    elif type(test_split) == float and 0<test_split<1:
    #       return int(self.size*test_split)
    #   else:
    #       raise ValueError("test split should be positive integer number or float proportion, got " + str(test_split))

    def get_class_proportions(self, data=None):
        if data is None:
            data = self.target
        return np.bincount(data.astype(int))/len(data)

    def get_train_sample_weights(self):
        return (1/(self.n_classes * self.get_class_proportions(self.target_train)))[self.target_train]

    def info(self):
        print("Dataset info : ")
        print("task : {}".format(self.task))
        print("sample count (train + test) : {} with {} for training".format(self.size, len(self.data_train)))
        print("number of features : {} ".format(self.n_features))
        if self.task =="classification":
            print(" of which continuous : {}".format(self.nb_continuous_features))
            print("One hot encoded features : {}".format(self.one_hot_categoricals))
            print("number of classes : {}".format(self.n_classes))
            print("class proportions : ")
            print(self.get_class_proportions())
        print("")

class Higgs(Datasets):#binary
    
    def __init__(self, filename=None, subsample=None, test_split=0.005, random_state=0, normalize_intervals=False, as_pandas=False):
        filename = filename or os.path.dirname(__file__) + "/HIGGS.csv.gz"
        URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        if not os.path.exists(filename):
            reply = str(input("Higgs dataset file not provided, would you like to download it ? (2.6 Go) (y/n): ")).lower().strip()
            if reply == 'y':
                print(f"Downloading {URL} to {filename} (2.6 GB)...")
                urlretrieve(URL, filename)
                print("done.")
            else:
                print("Higgs boson dataset unavailable, exiting")
                exit()

        print(f"Loading Higgs boson dataset from {filename}...")
        tic = time()
        with GzipFile(filename) as f:
            self.df = pd.read_csv(f, header=None, dtype=np.float32)
        toc = time()
        print(f"Loaded {self.df.values.nbytes / 1e9:0.3f} GB in {toc - tic:0.3f}s")

        if subsample is not None:
            print("Subsampling dataset with subsample={}".format(subsample))

        self.data = self.df[:subsample]
        self.target = self.data.pop(0).astype(int).values

        if normalize_intervals:
            mins = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)

        if not as_pandas:
            self.data = np.ascontiguousarray(self.data.values)

        self.binary = True
        self.task = "classification"
        self.n_classes = 2
        self.one_hot_categoricals = False
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features


        print("Making train/test split ...")
        
        self.split_train_test(test_split, random_state)
        #self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
        #    self.data, self.target, test_size=self.get_test_size(test_split), random_state=random_state)

        print("Done.")

    
class Moons(Datasets):#binary
    
    def __init__(self, n_samples=10000, test_split=0.3, random_state=0, normalize_intervals=False):
        
        from sklearn.datasets import make_moons
        
        self.data, self.target = make_moons(n_samples=n_samples, random_state=random_state)
        if normalize_intervals:
            self.data = MinMaxScaler.fit_transform(self.data)

        self.binary = True
        self.task = "classification"
        self.n_classes = 2
        self.one_hot_categoricals = False
        self.size = n_samples
        self.n_features = 2
        self.nb_continuous_features = self.n_features


        self.split_train_test(test_split, random_state)
        #self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
        #    self.data, self.target, test_size=self.get_test_size(test_split), random_state=random_state)


class Adult(Datasets):#binary

    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "adult.csv.zip"), "r")
        with archive.open("adult.csv") as f:
            data = pd.read_csv(f, header=None)
        data = data[:subsample]
        y = data.pop(13)
        discrete = [1, 3, 4, 5, 6, 7, 8, 12]
        continuous = list(set(range(13)) - set(discrete))
        X_continuous = data[continuous].astype("float32")

        if normalize_intervals:
            #X_continuous = MinMaxScaler().fit_transform(X_continuous)
            min_cols = X_continuous.min()
            X_continuous = (X_continuous - min_cols)/(X_continuous.max() - min_cols)

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#")#.values
        else:
            #X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).values.astype(int)
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).astype(int)


        self.binary = True
        self.task = "classification"

        self.n_classes = 2
        self.nb_continuous_features = len(continuous)#X_continuous.shape[1]
        self.target = pd.get_dummies(y).values[:, 1]
        if as_pandas:
            self.data = X_continuous.join(X_discrete)
            self.data.columns = list(range(13))
        else:
            self.data = np.hstack((X_continuous, X_discrete))

        self.one_hot_categoricals = one_hot_categoricals
        self.size, self.n_features = data.shape
        self.split_train_test(test_split, random_state)


class Bank(Datasets):#binary
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "bank.csv.zip"), "r")
        with archive.open("bank.csv") as f:
            data = pd.read_csv(f)
        data = data[:subsample]
        y = data.pop("y")
        discrete = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "campaign",
            "poutcome",
        ]
        continuous = ["age", "balance", "duration", "pdays", "previous"]
        X_continuous = data[continuous].astype("float32")


        if normalize_intervals:
            min_cols = X_continuous.min()
            X_continuous = (X_continuous - min_cols)/(X_continuous.max() - min_cols)

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#")#.values
        else:
            #X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).values.astype(int)
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).astype(int)

        self.binary = True
        self.task = "classification"

        self.n_classes = 2
        self.target = pd.get_dummies(y).values[:, 1]
        if as_pandas:
            self.data = X_continuous.join(X_discrete)
            self.data.columns = list(range(len(discrete)+len(continuous)))
        else:
            self.data = np.hstack((X_continuous, X_discrete))

        self.one_hot_categoricals = one_hot_categoricals
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = len(continuous)#X_continuous.shape[1]
        #X = MinMaxScaler().fit_transform(X) # Why normalize again after adding discrete features ??
        self.split_train_test(test_split, random_state)

class Car(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "car.csv.zip"), "r")
        with archive.open("car.csv") as f:
            data = pd.read_csv(f, header=None)
        data = data[:subsample]
        y = data.pop(6)
        self.binary = False
        self.task = "classification"

        self.target = np.argmax(pd.get_dummies(y).values, axis=1)
        self.n_classes = np.max(self.target).astype(int)+1

        if one_hot_categoricals:
            self.data = pd.get_dummies(data, prefix_sep="#").astype("float32")
        else:
            self.data = data.apply(lambda x: pd.factorize(x)[0]).astype(int)#.values#.astype("float32")


        if normalize_intervals:
            mins = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)

        if not as_pandas:
            self.data = self.data.values

        self.one_hot_categoricals = one_hot_categoricals
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = 0

        self.split_train_test(test_split, random_state)

class Cardio(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "cardiotocography.csv.zip"), "r")
        with archive.open("cardiotocography.csv", ) as f:
            data = pd.read_csv(f, sep=";", decimal=",")

        data = data[:subsample]
        data.drop(
            [
                "FileName",
                "Date",
                "SegFile",
                "A",
                "B",
                "C",
                "D",
                "E",
                "AD",
                "DE",
                "LD",
                "FS",
                "SUSP",
            ],
            axis=1,
            inplace=True,
        )
        # A 10-class label
        y_class = data.pop("CLASS").values
        y_class -= 1
        # A 3-class label
        y_nsp = data.pop("NSP").values
        y_nsp -= 1
        continuous = [
            "b",
            "e",
            "LBE",
            "LB",
            "AC",
            "FM",
            "UC",
            "ASTV",
            "MSTV",
            "ALTV",
            "MLTV",
            "DL",
            "DS",
            "DP",
            "Width",
            "Min",
            "Max",
            "Nmax",
            "Nzeros",
            "Mode",
            "Mean",
            "Median",
            "Variance",
        ]
        discrete = ["Tendency"]
        X_continuous = data[continuous].astype("float32")

        if normalize_intervals:
            min_cols = X_continuous.min()
            X_continuous = (X_continuous - min_cols)/(X_continuous.max() - min_cols)

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#")#.values
        else:
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).astype(int)

        if as_pandas:
            self.data = X_continuous.join(X_discrete)
            self.data.columns = list(range(len(discrete)+len(continuous)))
        else:
            self.data = np.hstack((X_continuous, X_discrete)).astype("float32")

        self.target = y_nsp
        self.binary = False
        self.task = "classification"

        self.n_classes = np.max(self.target).astype(int)+1

        self.one_hot_categoricals = one_hot_categoricals
        self.size, self.n_features = data.shape
        self.nb_continuous_features = len(continuous)

        self.split_train_test(test_split, random_state)

class Churn(Datasets):#binary
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "churn.csv.zip"), "r")
        with archive.open("churn.csv") as f:
            data = pd.read_csv(f)
        data = data[:subsample]
        y = data.pop("Churn?")
        discrete = [
            "State",
            "Area Code",
            "Int'l Plan",
            "VMail Plan",
        ]

        continuous = [
            "Account Length",
            "Day Mins",
            "Day Calls",
            "Eve Calls",
            "Day Charge",
            "Eve Mins",
            "Eve Charge",
            "Night Mins",
            "Night Calls",
            "Night Charge",
            "Intl Mins",
            "Intl Calls",
            "Intl Charge",
            "CustServ Calls",
            "VMail Message",
        ]
        self.target = pd.get_dummies(y).values[:, 1]
        self.binary = True
        self.task = "classification"

        self.n_classes = 2

        X_continuous = data[continuous].astype("float32")

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#")#.values
        else:
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).astype(int)
        if normalize_intervals:
            min_cols = X_continuous.min()
            X_continuous = (X_continuous - min_cols)/(X_continuous.max() - min_cols)

        if as_pandas:
            self.data = X_continuous.join(X_discrete)
            self.data.columns = list(range(self.data.shape[1]))#len(discrete)+len(continuous)))
        else:
            self.data = np.hstack((X_continuous, X_discrete)).astype("float32")


        self.one_hot_categoricals = one_hot_categoricals
        self.size = data.shape[0]
        self.n_features = len(continuous) + len(discrete)
        self.nb_continuous_features = len(continuous)

        self.split_train_test(test_split, random_state)

class Default_cb(Datasets):#binary
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "default_cb.csv.zip"), "r")
        with archive.open("default_cb.csv") as f:
            data = pd.read_csv(f)
        data = data[:subsample]
        continuous = [
            "AGE",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "LIMIT_BAL",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ]
        discrete = [
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
        ]
        _ = data.pop("ID")
        y = data.pop("default payment next month")
        self.target = pd.get_dummies(y).values[:, 1]
        self.task = "classification"

        self.binary = True
        self.n_classes = 2

        X_continuous = data[continuous].astype("float32")

        if normalize_intervals:
            min_cols = X_continuous.min()
            X_continuous = (X_continuous - min_cols)/(X_continuous.max() - min_cols)

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#")#.values
        else:
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).astype(int)

        if as_pandas:
            self.data = X_continuous.join(X_discrete)
            self.data.columns = list(range(len(discrete)+len(continuous)))
        else:
            self.data = np.hstack((X_continuous, X_discrete)).astype("float32")


        self.one_hot_categoricals = one_hot_categoricals
        self.size = data.shape[0]
        self.n_features = len(continuous) + len(discrete)
        self.nb_continuous_features = len(continuous)

        self.split_train_test(test_split, random_state)

class Letter(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "letter.csv.zip"), "r")
        with archive.open("letter.csv") as f:
            data = pd.read_csv(f)
        data = data[:subsample]
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
        self.target = data.pop("y").values
        self.data = data.astype("float32")
        if normalize_intervals:
            mins = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)
        if not as_pandas:
            self.data = self.data.values
        self.binary = False
        self.task = "classification"

        self.n_classes = np.max(self.target).astype(int)+1

        self.one_hot_categoricals = one_hot_categoricals
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class Satimage(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "satimage.csv.zip"), "r")
        with archive.open("satimage.csv") as f:
            data = pd.read_csv(f)
        data = data[:subsample]
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
        self.target = data.pop("y").values
        self.data = data.astype("float32")
        if normalize_intervals:
            mins = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)
        if not as_pandas:
            self.data = self.data.values
        self.binary = False
        self.task = "classification"

        self.n_classes = np.max(self.target).astype(int)+1

        self.one_hot_categoricals = one_hot_categoricals
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class Sensorless(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "sensorless.csv.zip"), "r")
        with archive.open("sensorless.csv") as f:
            data = pd.read_csv(f, sep=" ", header=None)
        data = data[:subsample]
        y = data.pop(48).values
        y -= 1

        self.target = y
        self.binary = False
        self.task = "classification"

        self.n_classes = np.max(self.target).astype(int)+1
        self.data = data.astype("float32")
        if normalize_intervals:
            mins = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)
        if not as_pandas:
            self.data = self.data.values

        self.one_hot_categoricals = one_hot_categoricals
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class Spambase(Datasets):#binary
    def __init__(self, path="data", test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        archive = zipfile.ZipFile(os.path.join(path, "spambase.csv.zip"), "r")
        with archive.open("spambase.csv") as f:
            data = pd.read_csv(f, header=None)

        data = data[:subsample]

        y = data.pop(57).values

        self.target = y
        self.binary = True
        self.task = "classification"

        self.n_classes = 2
        self.data = data.astype("float32")
        if normalize_intervals:
            mins = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)
        if not as_pandas:
            self.data = self.data.values

        self.one_hot_categoricals = one_hot_categoricals
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class Covtype(Datasets):#multiclass
    def __init__(self, path=None, test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        from sklearn.datasets import fetch_covtype
        data, target = fetch_covtype(return_X_y=True, random_state=random_state, as_frame = True)#as_pandas)

        if subsample is not None:
            print("Subsampling dataset with subsample={}".format(subsample))

        data = data[:subsample]
        target = target[:subsample]

        self.target = target.astype(int) - 1#.values
        self.binary = False
        self.task = "classification"

        self.n_classes = 7

        self.data = data#.astype("float32")


        X_continuous = data[data.columns[:10]]
        if normalize_intervals:
            mins = X_continuous.min()
            X_continuous = (X_continuous - mins)/(X_continuous.max() - mins)

        if not one_hot_categoricals:
            soil_type = data.apply(lambda x : x[14:].argmax(), axis=1)
            wilderness_area = data.apply(lambda x : x[10:14].argmax(), axis=1)
            X_discrete = pd.concat([soil_type, wilderness_area], axis=1)
        else:
            X_discrete = data[data.columns[10:]]

        self.one_hot_categoricals = one_hot_categoricals
        self.data = X_continuous.join(X_discrete)
        if not as_pandas:
            self.data = self.data.values
            self.target = self.target.values
        else:
            self.data.columns = list(range(self.data.shape[1]))

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = 10

        self.split_train_test(test_split, random_state)

class KDDCup(Datasets):#multiclass
    def __init__(self, path=None, test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        from sklearn.datasets import fetch_kddcup99
        print("WARNING : loading only 10 percent of KDDCdup dataset (default) give percent10=False parameter to change this")
        print("")
        data, target = fetch_kddcup99(return_X_y=True, random_state=random_state, as_frame = True)#as_pandas)

        if subsample is not None:
            print("Subsampling dataset with subsample={}".format(subsample))

        data = data[:subsample]
        target = target[:subsample]

        discrete = ["protocol_type", "service", "flag", "land", "logged_in", "is_host_login", "is_guest_login"]
        continuous  = list(set(data.columns) - set(discrete))

        dummies = pd.get_dummies(target)
        dummies.columns = list(range(len(dummies.columns)))
        self.target = dummies.idxmax(axis=1)#.values
        self.binary = False
        self.task = "classification"

        self.n_classes = self.target.max()+1#23


        X_continuous = data[continuous].astype("float32")
        if normalize_intervals:
            mins = X_continuous.min()
            X_continuous = (X_continuous - mins)/(X_continuous.max() - mins)

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#")  # .values
        else:
            #X_discrete = (data[discrete]).apply(lambda x: pd.factorize(x)[0])
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).astype(int)

        self.one_hot_categoricals = one_hot_categoricals
        self.data = X_continuous.join(X_discrete)


        if not as_pandas:
            self.data = self.data.values
            self.target = self.target.values
        else:
            self.data.columns = list(range(self.data.shape[1]))

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = 34#32

        self.split_train_test(test_split, random_state)

class NewsGroups(Datasets):#multiclass
    def __init__(self, path=None, test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        from sklearn.datasets import fetch_20newsgroups_vectorized

        data, target = fetch_20newsgroups_vectorized(return_X_y=True, as_frame = True)#as_pandas)

        if subsample is not None:
            print("Subsampling dataset with subsample={}".format(subsample))

        data = data[:subsample]
        target = target[:subsample]

        self.target = target
        self.binary = False
        self.task = "classification"

        self.n_classes = self.target.max()+1

        if normalize_intervals:
            mins = data.min()
            data = (data - mins)/(data.max() - mins)

        self.data = data

        if not as_pandas:
            self.data = self.data.values
            self.target = self.target.values

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class BreastCancer(Datasets):#binary
    def __init__(self, path=None, test_split=0.3, random_state=0, normalize_intervals=False, one_hot_categoricals=False, as_pandas=False, subsample=None):
        from sklearn.datasets import load_breast_cancer

        data, target = load_breast_cancer(return_X_y=True, as_frame = True)#as_pandas)

        data = data[:subsample]
        target = target[:subsample]

        self.data = data
        self.target = target
        self.binary = True
        self.task = "classification"

        self.n_classes = 2


        if normalize_intervals:
            self.data = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)

        self.one_hot_categoricals = False


        if not as_pandas:
            self.data = self.data.values
            self.target = self.target.values
        else:
            self.data.columns = list(range(self.data.shape[1]))

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)


class CaliforniaHousing(Datasets):#regression
    def __init__(self, path=None, test_split=0.3, random_state=0, normalize_intervals=False, as_pandas=False, subsample=None):
        from sklearn.datasets import fetch_california_housing

        data, target = fetch_california_housing(return_X_y=True, as_frame = True)#as_pandas)

        data = data[:subsample]
        target = target[:subsample]

        self.data = data
        self.target = target
        self.task = "regression"

        if normalize_intervals:
            mins = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)

        if not as_pandas:
            self.data = self.data.values
            self.target = self.target.values

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class Boston(Datasets):#regression
    def __init__(self, path=None, test_split=0.3, random_state=0, normalize_intervals=False, as_pandas=False, subsample=None):
        from sklearn.datasets import load_boston

        data, target = load_boston(return_X_y=True)

        data = data[:subsample]
        target = target[:subsample]

        self.data = data
        self.target = target
        self.task = "regression"

        if normalize_intervals:
            self.data = MinMaxScaler.fit_transform(self.data)

        if as_pandas:
            self.data = pd.DataFrame(self.data)
            self.target = pd.Series(self.target)

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)



class Diabetes(Datasets):#regression
    def __init__(self, path=None, test_split=0.3, random_state=0, normalize_intervals=False, as_pandas=False, subsample=None):
        from sklearn.datasets import load_diabetes

        data, target = load_diabetes(return_X_y=True, as_frame=True)

        data = data[:subsample]
        target = target[:subsample]

        self.data = data
        self.target = target
        self.task = "regression"

        if normalize_intervals:
            mins = self.data.min()
            self.data = (self.data - mins)/(self.data.max() - mins)

        if not as_pandas:
            self.data = self.data.values
            self.target = self.target.values

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)




def load_dataset(args, as_pandas=False):
    print("Loading dataset {}".format(args.dataset))
    if args.dataset == "Moons":
        return Moons(random_state=args.random_state, normalize_intervals=args.normalize_intervals)
    #elif args.dataset == "Adult":
    #    return Adult(path=args.dataset_path, random_state=args.random_state,
    #                 normalize_intervals=args.normalize_intervals, subsample=args.dataset_subsample)
    elif args.dataset == "Higgs":
        return Higgs(filename=args.dataset_path + "/HIGGS.csv.gz", subsample=args.dataset_subsample, random_state=args.random_state, normalize_intervals=args.normalize_intervals)
    elif args.dataset == "Adult":
        return Adult(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals, one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Bank":
        return Bank(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals, one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Car":
        return Car(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                     one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Cardio":
        return Cardio(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                   one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Churn":
        return Churn(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                          one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Default_cb":
        return Default_cb(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                      one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Letter":
        return Letter(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                        one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Satimage":
        return Satimage(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                          one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Sensorless":
        return Sensorless(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                        one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Spambase":
        return Spambase(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                    one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "Covtype":
        return Covtype(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                    one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "KDDCup":
        return KDDCup(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                   one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas, subsample=args.dataset_subsample)
    elif args.dataset == "NewsGroups":
        return NewsGroups(path=args.dataset_path, random_state=args.random_state,
                      normalize_intervals=args.normalize_intervals,
                      one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas,
                      subsample=args.dataset_subsample)
    elif args.dataset == "BreastCancer":
        return BreastCancer(path=args.dataset_path, random_state=args.random_state,
                      normalize_intervals=args.normalize_intervals,
                      one_hot_categoricals=args.one_hot_categoricals, as_pandas=as_pandas,
                      subsample=args.dataset_subsample)
    elif args.dataset == "CaliforniaHousing":
        return CaliforniaHousing(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals, as_pandas=as_pandas,
                     subsample=args.dataset_subsample)
    elif args.dataset == "Boston":
        return Boston(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals, as_pandas=as_pandas,
                     subsample=args.dataset_subsample)
    elif args.dataset == "Diabetes":
        return Diabetes(path=args.dataset_path, random_state=args.random_state, normalize_intervals=args.normalize_intervals, as_pandas=as_pandas,
                     subsample=args.dataset_subsample)

    else:
        raise ValueError("unknown dataset " + args.dataset)

