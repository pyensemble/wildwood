from time import time
import argparse

import numpy as np
import datasets
import pandas as pd
from catboost import CatBoostClassifier#, Pool

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default="Moons")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_true", default=False)
parser.add_argument('--datasets-path', type=str, default="data")
parser.add_argument('--datasets-subsample', type=int, default=100000)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--n-jobs', type=int, default=-1)
parser.add_argument('--random-state', type=int, default=0)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--no-cat-features', action="store_true", default=False)


args = parser.parse_args()

if args.one_hot_categoricals:
    print("WARNING : received one_hot_categoricals=True, will not use LightGBM's special management of categorical features")
use_cat_features = not(args.no_cat_features or args.one_hot_categoricals)

print("Running CatBoost classifier with training set {}".format(args.dataset))

print("with arguments")
print(args)
dataset = datasets.load_dataset(args, as_pandas=use_cat_features)

if dataset.task != "classification":
    print("The loaded datasets is not for classification ... exiting")
    exit()

print("Training CatBoost classifier ...")
cat_features = np.arange(dataset.nb_continuous_features, dataset.n_features) if use_cat_features else None
print("cat features are ")
print(cat_features)
clf = CatBoostClassifier(n_estimators=args.n_estimators, random_seed=args.random_state, cat_features=cat_features, thread_count=args.n_jobs)

#def preprocess_for_cat_features(data, nb_continuous):
#    """Simply use pandas dataframe for now ..."""
#    return pd.DataFrame(data[:,:nb_continuous]).join(pd.DataFrame(data[:,nb_continuous:], columns=range(nb_continuous, data.shape[1])).astype(int))

sample_weights = dataset.get_train_sample_weights()
tic = time()
clf.fit(dataset.data_train, dataset.target_train, verbose=bool(args.verbose), sample_weight=sample_weights, cat_features=cat_features)
toc = time()

print(f"fitted in {toc - tic:.3f}s")

datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)

