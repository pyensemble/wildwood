import sys
import subprocess
from time import time
from datetime import datetime
import logging
import pickle as pkl
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

sys.path.extend([".", ".."])

from wildwood.datasets import loaders_small_classification, load_churn
from wildwood.forest import ForestClassifier


from wildwood.datasets import (
    load_adult,
    load_bank,
    load_breastcancer,
    load_car,
    load_cardio,
    load_churn,
    load_default_cb,
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
)

import matplotlib.pyplot as plt

# # import xgboost as xgb
# from catboost import CatBoostClassifier
# import lightgbm as lgb


# TODO: pre-compile wildwood


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

loaders = [
    load_adult,
    load_bank,
    # load_breastcancer,
    # load_car,
    # load_cardio,
    load_churn,
    load_default_cb,
    # load_letter,
    # load_satimage,
    # load_sensorless,
    # load_spambase,
]

# loaders = [load_churn]

# TODO: wildwood avec et sans aggregation ?

random_state = 42

classifiers = [
    # lambda: (
    #     "LRW",
    #     LogisticRegression(
    #         class_weight="balanced",
    #         solver="saga",
    #         random_state=random_state,
    #         max_iter=100,
    #         verbose=False,
    #     ),
    # ),
    # lambda: (
    #     "LR",
    #     LogisticRegression(
    #         solver="saga", random_state=random_state, max_iter=100, verbose=False,
    #     ),
    # ),
    # lambda: ("RF", RandomForestClassifier(n_jobs=-1, random_state=random_state)),
    # lambda: (
    #     "RFW",
    #     RandomForestClassifier(
    #         class_weight="balanced", n_jobs=-1, random_state=random_state
    #     ),
    # ),
    # lambda: ("WW", ForestClassifier(n_jobs=-1, random_state=random_state)),
    # lambda: (
    #     "WildWood (Agg)",
    #     ForestClassifier(class_weight="balanced", n_jobs=-1, random_state=random_state),
    # ),
    lambda: (
        "WildWood (One Hot)",
        ForestClassifier(
            n_estimators=1,
            n_jobs=1,
            class_weight="balanced",
            random_state=random_state,
            aggregation=True,
            max_features=None,
            # dirichlet=0.0,
        ),
    ),
    lambda: (
        "WildWood (Ordinal)",
        ForestClassifier(
            n_estimators=1,
            n_jobs=1,
            class_weight="balanced",
            random_state=random_state,
            aggregation=True,
            max_features=None,
            # dirichlet=0.0,
        ),
    ),
    lambda: (
        "WildWood (Categorical)",
        ForestClassifier(
            n_estimators=1,
            n_jobs=1,
            class_weight="balanced",
            random_state=random_state,
            aggregation=True,
            max_features=None,
            # dirichlet=0.0,
        ),
    ),
]


data_extractions = [
    # # For LRW
    # {"one_hot_encode": True, "standardize": True, "drop": "first"},
    # # For RFW
    # {"one_hot_encode": True, "standardize": False, "drop": None},
    {"one_hot_encode": True, "standardize": False, "drop": None, "categorical": False},
    {"one_hot_encode": False, "standardize": False, "drop": None, "categorical": False},
    {"one_hot_encode": False, "standardize": False, "drop": None, "categorical": True},
]

# data_extraction = {
#     # "LogisticRegression": {
#     #     "one_hot_encode": True,
#     #     "standardize": True,
#     #     "drop": "first",
#     # },
#     # "RandomForestClassifier": {
#     #     "one_hot_encode": True,
#     #     "standardize": False,
#     #     "drop": None,
#     # },
#     # "ForestClassifier": {"one_hot_encode": True, "standardize": False, "drop": None},
#     "ForestClassifier": {"one_hot_encode": False, "standardize": False, "drop": None}
# }


# Number of time each experiment is repeated, one for each seed (leading
# data_random_states = [42, 43, 44, 46, 47, 49, 50, 52, 53, 55]
data_random_states = [42]
clf_random_state = 42

col_data = []
col_classifier = []
col_classifier_title = []
col_repeat = []
col_fit_time = []
col_predict_time = []
col_roc_auc = []
col_roc_auc_weighted = []
col_avg_precision_score = []
col_avg_precision_score_weighted = []
col_log_loss = []
col_accuracy = []

col_roc_auc_train = []
col_roc_auc_weighted_train = []
col_avg_precision_score_train = []
col_avg_precision_score_weighted_train = []
col_log_loss_train = []
col_accuracy_train = []


for Clf, data_extraction in zip(classifiers, data_extractions):
    clf_title, clf = Clf()
    clf_name = clf.__class__.__name__
    logging.info("=" * 128)
    logging.info("Launching experiments for %s" % clf_name)
    for loader in loaders:
        dataset = loader()
        data_name = dataset.name
        task = dataset.task
        for key, val in data_extraction.items():
            if key != "categorical":
                setattr(dataset, key, val)
        categorical = data_extraction["categorical"]
        logging.info("-" * 64)
        logging.info("Launching task for %r" % dataset)
        for repeat, data_random_state in enumerate(data_random_states):
            clf_title, clf = Clf()
            col_data.append(data_name)
            col_classifier.append(clf_name)
            col_classifier_title.append(clf_title)
            col_repeat.append(repeat)
            X_train, X_test, y_train, y_test = dataset.extract(
                random_state=data_random_state
            )
            logging.info(
                "datasets: %s random_state: %d repeat: %d n_samples_train: %d "
                "n_samples_test: %d n_features: %d n_features_continuous: %d "
                "n_features_categorical: %d n_columns: %d"
                % (
                    dataset.name,
                    data_random_state,
                    repeat,
                    dataset.n_samples_train_,
                    dataset.n_samples_test_,
                    dataset.n_features_,
                    dataset.n_features_continuous_,
                    dataset.n_features_categorical_,
                    dataset.n_columns_,
                )
            )

            if hasattr(clf, "categorical_features") and categorical:
                # If it is WildWood's classifier, we set the categorical features
                logging.info(
                    "Setting categorical_features: %r" % dataset.categorical_features_
                )
                clf.categorical_features = dataset.categorical_features_

            y_train_binary = LabelBinarizer().fit_transform(y_train)
            y_test_binary = LabelBinarizer().fit_transform(y_test)
            tic = time()
            clf.fit(X_train, y_train)
            toc = time()
            fit_time = toc - tic
            logging.info("Fitted %s in %.2f seconds on " % (clf_name, fit_time))
            col_fit_time.append(fit_time)
            tic = time()
            y_scores = clf.predict_proba(X_test)
            y_scores_train = clf.predict_proba(X_train)
            toc = time()
            predict_time = toc - tic
            col_predict_time.append(predict_time)
            logging.info("Predict %s in %.2f seconds" % (clf_name, fit_time))
            y_pred = np.argmax(y_scores, axis=1)
            y_pred_train = np.argmax(y_scores_train, axis=1)

            if task == "binary-classification":
                roc_auc = roc_auc_score(y_test, y_scores[:, 1])
                roc_auc_train = roc_auc_score(y_train, y_scores_train[:, 1])
                roc_auc_weighted = roc_auc
                roc_auc_weighted_train = roc_auc_train
                avg_precision_score = average_precision_score(y_test, y_scores[:, 1])
                avg_precision_score_train = average_precision_score(
                    y_train, y_scores_train[:, 1]
                )
                avg_precision_score_weighted = avg_precision_score
                avg_precision_score_weighted_train = avg_precision_score_train
                log_loss_ = log_loss(y_test, y_scores)
                log_loss_train_ = log_loss(y_train, y_scores_train)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_train = accuracy_score(y_train, y_pred_train)
            elif task == "multiclass-classification":
                roc_auc = roc_auc_score(
                    y_test, y_scores, multi_class="ovr", average="macro"
                )
                roc_auc_train = roc_auc_score(
                    y_train, y_scores_train, multi_class="ovr", average="macro"
                )
                roc_auc_weighted = roc_auc_score(
                    y_test, y_scores, multi_class="ovr", average="weighted"
                )
                roc_auc_weighted_train = roc_auc_score(
                    y_train, y_scores_train, multi_class="ovr", average="weighted"
                )

                avg_precision_score = average_precision_score(y_test_binary, y_scores)
                avg_precision_score_train = average_precision_score(
                    y_train_binary, y_scores_train
                )
                avg_precision_score_weighted = average_precision_score(
                    y_test_binary, y_scores, average="weighted"
                )
                avg_precision_score_weighted_train = average_precision_score(
                    y_train_binary, y_scores_train, average="weighted"
                )
                log_loss_ = log_loss(y_test, y_scores)
                log_loss_train_ = log_loss(y_train, y_scores_train)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_train = accuracy_score(y_train, y_pred_train)
            # TODO: regression
            else:
                raise ValueError("Task %s not understood" % task)

            col_roc_auc.append(roc_auc)
            col_roc_auc_train.append(roc_auc_train)
            col_roc_auc_weighted.append(roc_auc_weighted)
            col_roc_auc_weighted_train.append(roc_auc_weighted_train)
            col_avg_precision_score.append(avg_precision_score)
            col_avg_precision_score_train.append(avg_precision_score_train)
            col_avg_precision_score_weighted.append(avg_precision_score_weighted)
            col_avg_precision_score_weighted_train.append(
                avg_precision_score_weighted_train
            )
            col_log_loss.append(log_loss_)
            col_log_loss_train.append(log_loss_train_)
            col_accuracy.append(accuracy)
            col_accuracy_train.append(accuracy_train)

            logging.info(
                "AUC= %.2f, AUCW: %.2f, AVGP: %.2f, AVGPW: %.2f, LOGL: %.2f, ACC: %.2f"
                % (
                    roc_auc,
                    roc_auc_weighted,
                    avg_precision_score,
                    avg_precision_score_weighted,
                    log_loss_,
                    accuracy,
                )
            )


results = pd.DataFrame(
    {
        "datasets": col_data,
        "classifier": col_classifier,
        "classifier_title": col_classifier_title,
        "repeat": col_repeat,
        "fit_time": col_fit_time,
        "predict_time": col_predict_time,
        "roc_auc": col_roc_auc,
        "roc_auc_w": col_roc_auc_weighted,
        "avg_prec": col_avg_precision_score,
        "avg_prec_w": col_avg_precision_score_weighted,
        "log_loss": col_log_loss,
        "accuracy": col_accuracy,
        "roc_auc_train": col_roc_auc_train,
        "roc_auc_w_train": col_roc_auc_weighted_train,
        "avg_prec_train": col_avg_precision_score_train,
        "avg_prec_w_train": col_avg_precision_score_weighted_train,
        "log_loss_train": col_log_loss_train,
        "accuracy_train": col_accuracy_train,
    }
)

print(results)

now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

# Get the commit number as a string
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
commit = commit.decode("utf-8").strip()

filename = "benchmarks_" + now + ".pickle"
with open(filename, "wb") as f:
    pkl.dump({"datetime": now, "commit": commit, "results": results}, f)


logging.info("Saved results in file %s" % filename)
