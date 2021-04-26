import sys
import numpy as np
import pandas as pd

from numba import float32, jitclass, jit

# sys.path.extend([".."])
#
# from wildwood.forest import ForestClassifier
# from wildwood.dataset import load_bank
#
#
# dataset = load_bank()
# dataset.one_hot_encode = True
# dataset.standardize = False
# dataset.drop = None
# X_train, X_test, y_train, y_test = dataset.extract(random_state=42)
#
#
# clf = ForestClassifier(
#     n_estimators=1,
#     n_jobs=1,
#     class_weight="balanced",
#     random_state=42,
#     aggregation=False,
#     max_features=None,
#     dirichlet=0.0,
# )
#
# clf.fit(X_train, y_train)
#
#
# print(clf.path_leaf(X_train[:1]))

C_type = [
    ("left", float32),
    ("right", float32),
]


@jitclass(C_type)
class C(object):
    def __init__(self):
        self.left = 3.14
        self.right = 2.78


@jit(nopython=True, nogil=True)
def printc(c):
    print("C(left=", c.left, ", right=", c.right, ")")



@jit(nopython=True, nogil=True, locals={"tmp": float32})
def swap(c):
    c.left, c.right = c.right, c.left
    # tmp = c.right
    # c.right = c.left
    # c.left = tmp


@jit(nopython=True, nogil=True)
def main():
    c = C()
    printc(c)
    swap(c)
    printc(c)
    swap(c)
    printc(c)

main()
