from time import time
import argparse

import numpy as np
import datasets

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, average_precision_score
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Moons")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_true", default=False)
parser.add_argument('--dataset-path', type=str, default="data")
parser.add_argument('--dataset-subsample', type=int, default=100000)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--random-state', type=int, default=0)
parser.add_argument('--n-jobs', type=int, default=-1)
parser.add_argument('--max-depth', type=int, default=None)
parser.add_argument('--verbose', type=int, default=0)


args = parser.parse_args()


print("Running XGBoost classifier with training set {}".format(args.dataset))

dataset = datasets.load_dataset(args)
sample_weights = dataset.get_train_sample_weights()

print("Training XGBoost classifier ...")
tic = time()

clf = xgb.XGBClassifier(random_state=args.random_state, use_label_encoder=False, n_estimators=args.n_estimators, n_jobs=args.n_jobs, max_depth=args.max_depth)
print("Running XGBClassifier with use_label_encoder=False")
clf.fit(dataset.data_train, dataset.target_train, verbose=bool(args.verbose), sample_weight=sample_weights)
toc = time()

print(f"fitted in {toc - tic:.3f}s")

datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)
"""
predicted_proba_test = clf.predict_proba(dataset.data_test)
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