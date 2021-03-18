from time import time
import argparse

import numpy as np
import datasets
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, average_precision_score
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Moons")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_true", default=False)
parser.add_argument('--dataset-path', type=str, default="data")
parser.add_argument('--dataset-subsample', type=int, default=100000)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--random-state', type=int, default=0)
parser.add_argument('--no-cat-features', action="store_true", default=False)


args = parser.parse_args()

if args.one_hot_categoricals:
    print("WARNING : received one_hot_categoricals=True, will not use LightGBM's special management of categorical features")
use_cat_features = not(args.no_cat_features or args.one_hot_categoricals)


print("Running Lightgbm classifier with training set {}".format(args.dataset))
print ("")
print("with arguments")
print(args)
print ("")

dataset = datasets.load_dataset(args)

print("Training Lightgbm classifier ...")
cat_features = np.arange(dataset.nb_continuous_features, dataset.n_features)

def preprocess_for_cat_features(data, nb_continuous):
    """Simply use pandas dataframe for now ..."""
    print("Converting data to pandas.DataFrame to manage cat features ...")
    return pd.DataFrame(data[:,:nb_continuous]).join(pd.DataFrame(data[:,nb_continuous:], columns=range(nb_continuous, data.shape[1])).astype('category'))

if len(cat_features) > 0 and use_cat_features:
    train_pool = preprocess_for_cat_features(dataset.data_train, dataset.nb_continuous_features)
    test_pool = preprocess_for_cat_features(dataset.data_test, dataset.nb_continuous_features)
else:
    train_pool = dataset.data_train
    test_pool = dataset.data_test

sample_weight = dataset.get_train_sample_weights()
tic = time()

clf = lgb.LGBMClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
clf.fit(train_pool, dataset.target_train, sample_weight=sample_weight)#, categorical_feature=list(cat_features))# if args.specify_cat_features else None)
toc = time()

print(f"done in {toc - tic:.3f}s")

datasets.evaluate_classifier(clf, test_pool, dataset.target_test, binary=dataset.binary)

"""
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
"""