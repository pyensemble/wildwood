import os
from gzip import GzipFile
from time import time


import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Datasets:

    def __init__(self):
        pass

    def split_train_test(self, test_split, random_state):
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.data, self.target, test_size=self.get_test_size(test_split), random_state=random_state)

    
    def get_test_size(self, test_split):
        if test_split <= 0:
            raise Error("test_split should be positive")
        if type(test_split) == int:
            return test_split
        elif type(test_split) == float and 0<test_split<1:
            return int(self.size*test_split)
        else:
            raise ValueError("test split should be positive integer number or float proportion, got " + str(test_split))

    def get_class_proportions(self):
        return np.bincount(self.target.astype(int))/self.size


class Higgs(Datasets):#binary
    
    def __init__(self, filename=None, subsample=None, test_split=0.005, random_state=0, normalize_intervals=False):
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

        self.data = np.ascontiguousarray(self.df.values[:subsample, 1:])

        if normalize_intervals:
            self.data = MinMaxScaler.fit_transform(self.data)

        self.binary = True
        self.n_classes = 2
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.target = self.df.values[:subsample, 0]

        print("Making train/test split ...")
        
        self.split_train_test(test_split, random_state)
        #self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
        #    self.data, self.target, test_size=self.get_test_size(test_split), random_state=random_state)

        print("Done.")

    
class Moons(Datasets):#binary
    
    def __init__(self, n_samples=10000, test_split=0.05, random_state=0, normalize_intervals=False):
        
        from sklearn.datasets import make_moons
        
        self.data, self.target = make_moons(n_samples=n_samples, random_state=random_state)
        if normalize_intervals:
            self.data = MinMaxScaler.fit_transform(self.data)

        self.binary = True
        self.n_classes = 2
        self.size = n_samples
        self.n_features = 2
        self.nb_continuous_features = self.n_features


        self.split_train_test(test_split, random_state)
        #self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
        #    self.data, self.target, test_size=self.get_test_size(test_split), random_state=random_state)


class Adult(Datasets):#binary

    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "adult.csv.zip"), "r")
        with archive.open("adult.csv") as f:
            data = pd.read_csv(f, header=None)
        y = data.pop(13)
        discrete = [1, 3, 4, 5, 6, 7, 8, 12]
        continuous = list(set(range(13)) - set(discrete))
        X_continuous = data[continuous].astype("float32")

        if normalize_intervals:
            X_continuous = MinMaxScaler().fit_transform(X_continuous)

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#").values
        else:
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).values

        self.binary = True
        self.target = pd.get_dummies(y).values[:, 1]
        self.data = np.hstack((X_continuous, X_discrete)).astype("float32")
        if normalize_intervals:
            self.data = MinMaxScaler().fit_transform(self.data)

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = X_continuous.shape[1]
        self.split_train_test(test_split, random_state)


class Bank(Datasets):#binary
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "bank.csv.zip"), "r")
        with archive.open("bank.csv") as f:
            data = pd.read_csv(f)
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
            X_continuous = MinMaxScaler().fit_transform(X_continuous)

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#").values
        else:
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).values

        self.binary = True
        self.target = pd.get_dummies(y).values[:, 1]
        self.data = np.hstack((X_continuous, X_discrete)).astype("float32")
        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = X_continuous.shape[0]
        #X = MinMaxScaler().fit_transform(X) # Why normalize again after adding discrete features ??
        self.split_train_test(test_split, random_state)

class Car(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "car.csv.zip"), "r")
        with archive.open("car.csv") as f:
            data = pd.read_csv(f, header=None)
        y = data.pop(6)
        self.binary = False
        self.target = np.argmax(pd.get_dummies(y).values, axis=1)

        if one_hot_categoricals:
            self.data = pd.get_dummies(data, prefix_sep="#").values.astype("float32")
        else:
            self.data = data.apply(lambda x: pd.factorize(x)[0]).values.astype("float32")

        if normalize_intervals:
            self.data = MinMaxScaler().fit_transform(self.data)

        self.size, self.n_features = data.shape
        self.nb_continuous_features = 0

        self.split_train_test(test_split, random_state)

class Cardio(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "cardiotocography.csv.zip"), "r")
        with archive.open("cardiotocography.csv", ) as f:
            data = pd.read_csv(f, sep=";", decimal=",")

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
            X_continuous = MinMaxScaler().fit_transform(X_continuous)

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#").values
        else:
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).values


        self.data = np.hstack((X_continuous, X_discrete)).astype("float32")
        self.target = y_nsp
        self.binary = False

        self.size, self.n_features = data.shape
        self.nb_continuous_features = len(continuous)

        self.split_train_test(test_split, random_state)

class Churn(Datasets):#binary
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "churn.csv.zip"), "r")
        with archive.open("churn.csv") as f:
            data = pd.read_csv(f)
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

        X_continuous = data[continuous].astype("float32")

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#").values
        else:
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).values


        self.data = np.hstack((X_continuous, X_discrete)).astype("float32")
        if normalize_intervals:
            self.data = MinMaxScaler().fit_transform(self.data)


        self.size = data.shape[0]
        self.n_features = len(continuous) + len(discrete)
        self.nb_continuous_features = len(continuous)

        self.split_train_test(test_split, random_state)

class Default_cb(Datasets):#binary
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "default_cb.csv.zip"), "r")
        with archive.open("default_cb.csv") as f:
            data = pd.read_csv(f)
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
        self.binary = True

        X_continuous = data[continuous].astype("float32")

        if one_hot_categoricals:
            X_discrete = pd.get_dummies(data[discrete], prefix_sep="#").values
        else:
            X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).values


        self.data = np.hstack((X_continuous, X_discrete)).astype("float32")
        if normalize_intervals:
            self.data = MinMaxScaler().fit_transform(self.data)


        self.size = data.shape[0]
        self.n_features = len(continuous) + len(discrete)
        self.nb_continuous_features = len(continuous)

        self.split_train_test(test_split, random_state)

class Letter(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "letter.csv.zip"), "r")
        with archive.open("letter.csv") as f:
            data = pd.read_csv(f)
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
        self.target = data.pop("y").values
        self.data = data.values.astype("float32")
        self.binary = False

        if normalize_intervals:
            self.data = MinMaxScaler().fit_transform(self.data)

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class Satimage(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "satimage.csv.zip"), "r")
        with archive.open("satimage.csv") as f:
            data = pd.read_csv(f)
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
        self.target = data.pop("y").values
        self.data = data.values.astype("float32")
        self.binary = False

        if normalize_intervals:
            self.data = MinMaxScaler().fit_transform(self.data)

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class Sensorless(Datasets):#multiclass
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "sensorless.csv.zip"), "r")
        with archive.open("sensorless.csv") as f:
            data = pd.read_csv(f, sep=" ", header=None)
        y = data.pop(48).values
        y -= 1

        self.target = y
        self.binary = False
        self.data = data.values.astype("float32")

        if normalize_intervals:
            self.data = MinMaxScaler().fit_transform(self.data)

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)

class Spambase(Datasets):#binary
    def __init__(self, path="data", test_split=0.05, random_state=0, normalize_intervals=False, one_hot_categoricals=False):
        archive = zipfile.ZipFile(os.path.join(path, "spambase.csv.zip"), "r")
        with archive.open("spambase.csv") as f:
            data = pd.read_csv(f, header=None)
        y = data.pop(57).values

        self.target = y
        self.binary = True
        self.data = data.values.astype("float32")

        if normalize_intervals:
            self.data = MinMaxScaler().fit_transform(self.data)

        self.size, self.n_features = self.data.shape
        self.nb_continuous_features = self.n_features

        self.split_train_test(test_split, random_state)


def load_dataset(args):
    print("Loading dataset {}".format(args.dataset))
    if args.dataset == "Moons":
        return Moons(random_state=args.random_state, normalize_intervals=args.normalize_intervals)
    elif args.dataset == "Adult":
        return Adult(path=args.dataset_path, random_state=args.random_state,
                     normalize_intervals=args.normalize_intervals)
    elif args.dataset == "Higgs":
        return Higgs(filename=args.dataset_filename, subsample=args.dataset_subsample, random_state=args.random_state, normalize_intervals=args.normalize_intervals)
    elif args.dataset == "Adult":
        return Adult(random_state=args.random_state, normalize_intervals=args.normalize_intervals, one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Bank":
        return Bank(random_state=args.random_state, normalize_intervals=args.normalize_intervals, one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Car":
        return Car(random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                     one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Cardio":
        return Cardio(random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                   one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Churn":
        return Churn(random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                          one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Default_cb":
        return Default_cb(random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                      one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Letter":
        return Letter(random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                        one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Satimage":
        return Satimage(random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                          one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Sensorless":
        return Sensorless(random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                        one_hot_categoricals=args.one_hot_categoricals)
    elif args.dataset == "Spambase":
        return Spambase(random_state=args.random_state, normalize_intervals=args.normalize_intervals,
                    one_hot_categoricals=args.one_hot_categoricals)

    else:
        raise ValueError("unknown dataset " + args.dataset)

