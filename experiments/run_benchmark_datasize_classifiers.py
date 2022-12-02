# License: BSD 3 clause

"""
This script produces experiments with defaults hyper-parameters for increasing data subset.

"""


import sys
import os
import subprocess
import time
from datetime import datetime
import logging
import pickle as pkl
import numpy as np
import pandas as pd
import argparse


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from run_hyperopt_classfiers import set_dataloader

sys.path.extend([".", ".."])
from wildwood.forest import ForestClassifier  # noqa: E402

DATA_EXTRACTION = {
    "RandomForestClassifier": {
        "one_hot_encode": True,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
    "HistGradientBoostingClassifier": {
        "one_hot_encode": False,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
    "XGBClassifier": {
        "one_hot_encode": True,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
    "LGBMClassifier": {
        "one_hot_encode": False,
        "standardize": False,
        "drop": None,
        "pd_df_categories": True,
    },
    "CatBoostClassifier": {
        "one_hot_encode": True,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
    "WildWood": {
        "one_hot_encode": False,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
}

def fit_kwargs_generator(clf_name, dataset):
    if clf_name == "RandomForestClassifier":
        return {}
    elif clf_name == "WildWood":
        return {"categorical_features": dataset.categorical_features_}
    elif clf_name == "XGBClassifier":
        return {}
    elif clf_name == "CatBoostClassifier":
        return {  # "cat_features": dataset.categorical_columns,
                "verbose": False}
    elif clf_name == "LGBMClassifier":
        return {"categorical_feature": "auto"}
    elif clf_name == "HistGradientBoostingClassifier":
        return {}
    else:
        raise NotImplementedError("Unknown classifier name:" + clf_name)


def set_classifier(clf_name, fit_seed, n_jobs=-1):
    classifier_setting = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, n_jobs=n_jobs, random_state=fit_seed
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=fit_seed
        ),
        "XGBClassifier": xgb.XGBClassifier(
            use_label_encoder=False,
            n_jobs=n_jobs,
            tree_method="hist",
            random_state=fit_seed,
        ),
        "LGBMClassifier": lgb.LGBMClassifier(n_jobs=n_jobs, random_state=fit_seed),
        "CatBoostClassifier": CatBoostClassifier(
            thread_count=n_jobs,
            random_state=fit_seed,
            logging_level="Silent",
            allow_writing_files=False,
        ),
        "WildWood": ForestClassifier(
            n_estimators=10, n_jobs=n_jobs, random_state=fit_seed, handle_unknown="consider_missing"
        ),
    }
    return classifier_setting[clf_name]


def run_default_params_exp(
    dataset,
    clf_name,
    learning_task,
    proportions
):

    col_data = []
    col_train_data_proportion = []
    col_classifier = []
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
    col_repeat = []

    for key, val in DATA_EXTRACTION[clf_name].items():
        setattr(dataset, key, val)

    X_train, X_test, y_train, y_test = dataset.extract(
        random_state=random_states["data_extract_random_state"]
    )

    # drop to few occurrence classes in kddcup
    if dataset.name == "kddcup":
        from collections import Counter

        c = Counter(y_train)
        classes_to_drop = [key for (key, val) in dict(c).items() if val < 20]

        drop_mask_train = np.isin(y_train, classes_to_drop)
        print("Drop %4d samples in train set" % drop_mask_train.sum())
        X_train = X_train[(~drop_mask_train)]
        y_train = y_train[(~drop_mask_train)]

        drop_mask_test = np.isin(y_test, classes_to_drop)
        print("Drop %4d samples in test set" % drop_mask_test.sum())
        X_test = X_test[(~drop_mask_test)]
        y_test = y_test[(~drop_mask_test)]

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # deal with nan in kick
    elif dataset.name == "kick" and clf_name in [
        "RandomForestClassifier10",
        "RandomForestClassifier100",
        "HistGradientBoostingClassifier",
        "WildWood10",
        "WildWood100",
    ]:
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
    # END special cases

    print("Run fitting with " + clf_name + " ...")
    fit_seeds = [0, 1, 2, 3, 4]
    for rep, fit_seed in enumerate(fit_seeds):
        logging.info("Repeat %d" % rep)

        for prop in proportions:
            logging.info("proportion %f" % prop)

            n_train = max(1, int(prop*len(X_train)))
            clf = set_classifier(clf_name, fit_seed)

            if clf_name == "WildWood":  # pre-compile wildwood
                clf.fit(X_train[:100], y_train[:100])
            if hasattr(X_test, "flags"):  # if it is a numpy array
                X_test = np.nan_to_num(X_test)

            if prop != 1.0:
                if hasattr(X_train, "flags"):
                    X_train_frac = np.nan_to_num(X_train[:n_train].copy())
                    y_train_frac = np.nan_to_num(y_train[:n_train].copy())
                    X_test = np.nan_to_num(X_test)
                    # print(X_train.flags)
                    # print(y_train.flags)
                    # print(X_train[:n_train].copy().flags)
                    # print(y_train[:n_train].copy().flags)
                else:
                    X_train_frac = X_train[:n_train]
                    y_train_frac = y_train[:n_train]
            else:
                X_train_frac = np.nan_to_num(X_train)
                y_train_frac = np.nan_to_num(y_train)

            tic = timer()
            clf.fit(X_train_frac, y_train_frac, **fit_kwargs_generator(clf_name, dataset))
            toc = timer()
            fit_time = toc - tic
            logging.info("Fitted %s in %.2f seconds" % (clf_name, fit_time))
            col_fit_time.append(fit_time)
            y_scores_train = clf.predict_proba(X_train_frac)
            tic = timer()
            y_scores = clf.predict_proba(X_test)
            toc = timer()
            predict_time = toc - tic
            col_predict_time.append(predict_time)
            logging.info("Predict %s in %.2f seconds" % (clf_name, predict_time))
            y_pred = np.argmax(y_scores, axis=1)
            y_pred_train = np.argmax(y_scores_train, axis=1)
            y_train_binary = LabelBinarizer().fit_transform(y_train_frac)
            y_test_binary = LabelBinarizer().fit_transform(y_test)

            if learning_task == "binary-classification":
                roc_auc = roc_auc_score(y_test, y_scores[:, 1])
                roc_auc_train = roc_auc_score(y_train_frac, y_scores_train[:, 1])
                roc_auc_weighted = roc_auc
                roc_auc_weighted_train = roc_auc_train
                avg_precision_score = average_precision_score(y_test, y_scores[:, 1])
                avg_precision_score_train = average_precision_score(
                    y_train_frac, y_scores_train[:, 1]
                )
                avg_precision_score_weighted = avg_precision_score
                avg_precision_score_weighted_train = avg_precision_score_train
                log_loss_ = log_loss(y_test, y_scores)
                log_loss_train_ = log_loss(y_train_frac, y_scores_train)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_train = accuracy_score(y_train_frac, y_pred_train)
            elif learning_task == "multiclass-classification":
                roc_auc = roc_auc_score(
                    y_test, y_scores, multi_class="ovr", average="macro"
                )
                roc_auc_train = roc_auc_score(
                    y_train_frac, y_scores_train, multi_class="ovr", average="macro"
                )
                roc_auc_weighted = roc_auc_score(
                    y_test, y_scores, multi_class="ovr", average="weighted"
                )
                roc_auc_weighted_train = roc_auc_score(
                    y_train_frac, y_scores_train, multi_class="ovr", average="weighted"
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
                log_loss_train_ = log_loss(y_train_frac, y_scores_train)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_train = accuracy_score(y_train_frac, y_pred_train)
            else:
                raise ValueError(
                    "Task %s not understood" % learning_task
                )

            col_data.append(dataset.name)
            col_train_data_proportion.append(prop)
            col_classifier.append(clf_name)

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
            col_repeat.append(rep)

            logging.info(
                "AUC= %.2f, AUCW: %.2f, AVGP: %.2f, AVGPW: %.2f, LOGL: %.2f, ACC: %.2f, REP: %d"
                % (
                    float(roc_auc),
                    float(roc_auc_weighted),
                    float(avg_precision_score),
                    float(avg_precision_score_weighted),
                    float(log_loss_),
                    float(accuracy),
                    rep,
                )
            )

    results = pd.DataFrame(
        {
            "dataset": col_data,
            "train_proportion": col_train_data_proportion,
            "classifier": col_classifier,
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
            "repeat": col_repeat,
        }
    )
    return results


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clf_name",
        choices=[
            "LGBMClassifier",
            "XGBClassifier",
            "CatBoostClassifier",
            "RandomForestClassifier",
            "HistGradientBoostingClassifier",
            "WildWood",
        ],
    )
    parser.add_argument(
        "--dataset_name",
        choices=[
            "adult",
            "bank",
            "breastcancer",
            "car",
            "cardio",
            "churn",
            "default-cb",
            "letter",
            "satimage",
            "sensorless",
            "spambase",
            "amazon",
            "covtype",
            "internet",
            "kick",
            "kddcup",
            "higgs",
        ],
    )
    parser.add_argument("--random_state_seed", type=int, default=42)

    args = parser.parse_args()

    clf_name = args.clf_name
    loader = set_dataloader(args.dataset_name)
    random_state_seed = args.random_state_seed

    if not os.path.exists("results"):
        os.mkdir("results")
    results_home_path = "results/"

    random_states = {
        "data_extract_random_state": random_state_seed,
        "train_val_split_random_state": 1 + random_state_seed,
        "expe_random_state": 2 + random_state_seed,
    }

    timer = time.time
    logging.info("=" * 128)
    dataset = loader()
    learning_task = dataset.task
    logging.info("Launching experiments for %s" % dataset.name)

    if not os.path.exists(results_home_path + dataset.name):
        os.mkdir(results_home_path + dataset.name)
    results_dataset_path = results_home_path + dataset.name + "/"

    results = run_default_params_exp(dataset, clf_name, dataset.task, [0.2, 0.4, 0.6, 0.8, 1.0])
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = "exp_datasize_" + now + ".pickle"

    with open(results_dataset_path + filename, "wb") as f:
        pkl.dump(
            {
                "datetime": now,
                "commit": commit,
                "results": results,
            },
            f,
        )

    logging.info("Saved results in file %s" % results_dataset_path + filename)
