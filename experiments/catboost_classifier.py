from time import time
import argparse

import numpy as np
import datasets
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, average_precision_score
from catboost import CatBoostClassifier, Pool

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Moons")
parser.add_argument('--normalize-intervals', type=bool, default=False)
parser.add_argument('--one-hot-categoricals', type=bool, default=False)
parser.add_argument('--dataset-path', type=str, default="data")
parser.add_argument('--dataset-subsample', type=int, default=100000)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--random-state', type=int, default=None)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--specify-cat-features', type=bool, default=False)


args = parser.parse_args()


print("Running CatBoost classifier with training set {}".format(args.dataset))

dataset = datasets.load_dataset(args)

print("Training CatBoost classifier ...")
cat_features = np.arange(dataset.nb_continuous_features, dataset.n_features) if args.specify_cat_features else []

clf = CatBoostClassifier(n_estimators=args.n_estimators, random_seed=args.random_state, cat_features=cat_features)

def preprocess_for_cat_features(data, nb_continuous):
    """Simply use pandas dataframe for now ..."""
    return pd.DataFrame(data[:,:nb_continuous]).join(pd.DataFrame(data[:,nb_continuous:], columns=range(nb_continuous, data.shape[1])).astype(int))

if len(cat_features) > 0:
    train_pool = preprocess_for_cat_features(dataset.data_train, dataset.nb_continuous_features)
    test_pool = preprocess_for_cat_features(dataset.data_test, dataset.nb_continuous_features)
else:
    train_pool = dataset.data_train
    test_pool = dataset.data_test

tic = time()
clf.fit(train_pool, dataset.target_train, verbose=bool(args.verbose), sample_weight=dataset.get_train_sample_weights(), cat_features=cat_features)
toc = time()

print(f"done in {toc - tic:.3f}s")


predicted_proba_test = clf.predict_proba(test_pool)
predicted_test = np.argmax(predicted_proba_test, axis=1)


if dataset.binary:
    roc_auc = roc_auc_score(dataset.target_test ,predicted_proba_test[:,1] , multi_class="ovo")
    avg_precision_score = average_precision_score(dataset.target_test, predicted_proba_test[:, 1])
else:
    onehot_target_test = datasets.onehotencode(dataset.target_test)

    roc_auc = roc_auc_score(onehot_target_test , predicted_proba_test, multi_class="ovo")
    avg_precision_score = average_precision_score(onehot_target_test, predicted_proba_test)

acc = accuracy_score(dataset.target_test, predicted_test)
log_loss_value = log_loss(dataset.target_test, predicted_proba_test)
print(f"ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")
print("ROC AUC computed with multi_class='ovo' (see sklearn docs)")

print(f"Log loss: {log_loss_value :.4f}")

print(f"Average precision score: {avg_precision_score :.4f}")

