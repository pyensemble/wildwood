from time import time
import argparse

import numpy as np
import datasets

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, average_precision_score
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Moons")
parser.add_argument('--normalize-intervals', type=bool, default=False)
parser.add_argument('--one-hot-categoricals', type=bool, default=False)
parser.add_argument('--dataset-path', type=str, default="data")
parser.add_argument('--dataset-subsample', type=int, default=100000)
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--random-state', type=int, default=None)


args = parser.parse_args()


print("Running Lightgbm classifier with training set {}".format(args.dataset))

dataset = datasets.load_dataset(args)

print("Training Lightgbm classifier ...")
tic = time()

clf = lgb.LGBMClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
clf.fit(dataset.data_train, dataset.target_train, categorical_feature=dataset.categorical_features, sample_weight=dataset.get_train_sample_weights())
toc = time()

print(f"done in {toc - tic:.3f}s")


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
