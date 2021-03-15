
from gzip import GzipFile
from time import time


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Datasets:

    def __init__(self):
        pass

    
    def get_test_size(self, test_split):
        if test_split <= 0:
            raise Error("test_split should be positive")
        if type(test_split) == int:
            return test_split
        elif type(test_split) == float and 0<test_split<1:
            return int(self.size*test_split)
        else:
            raise Error("test split should be positive integer number or float proportion, got " + str(test_split))
    

class Higgs(Datasets):
    
    def __init__(self, filename, subsample=None, test_split=0.005, random_state=0):

        print(f"Loading Higgs boson dataset from {filename}...")
        tic = time()
        with GzipFile(filename) as f:
            self.df = pd.read_csv(f, header=None, dtype=np.float32)
        toc = time()
        print(f"Loaded {self.df.values.nbytes / 1e9:0.3f} GB in {toc - tic:0.3f}s")
        
        self.data = np.ascontiguousarray(self.df.values[:subsample, 1:])
        self.n_classes = 2
        self.n_samples, self.n_features = self.data.shape
        self.target = self.df.values[:subsample, 0]

        
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.data, self.target, test_size=self.get_test_size(test_split), random_state=random_state)

    
class Moons(Datasets):
    
    def __init__(self, n_samples=10000, test_split=0.05, random_state=0):
        
        from sklearn.datasets import make_moons
        

        self.data, self.target = make_moons(n_samples=n_samples, random_state=random_state)
        self.n_classes = 2
        self.size = n_samples
        self.n_features = 2
                
        
            
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.data, self.target, test_size=self.get_test_size(test_split), random_state=random_state)

