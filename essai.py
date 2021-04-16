import sys
import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder, LabelBinarizer


y = ["one", "one", "three", "two", "one"]

print(LabelBinarizer().fit_transform(y))
print(LabelEncoder().fit_transform(y))


y = ["one", "one", "two", "two", "one"]

print(LabelBinarizer().fit_transform(y).astype(np.float32))
print(LabelEncoder().fit_transform(y))
