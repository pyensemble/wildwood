import sys
import os
import subprocess
import time
from datetime import datetime
import logging
import pickle as pkl
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

sys.path.extend([".", ".."])

from wildwood.datasets import (  # noqa: E402
    load_adult,
    # load_bank,
    # load_breastcancer,
    # load_car,
    # load_cardio,
    # load_churn,
    # load_default_cb,
    # load_letter,
    # load_satimage,
    # load_sensorless,
    # load_spambase,
    # load_amazon,
    # load_covtype,
    # load_higgs,
    # load_kddcup
)
from wildwood.forest import ForestClassifier

import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier


# TODO: pre-compile wildwood

timer = time.time #time.process_time #

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

loaders = [
    load_adult,
    # load_bank,
    # load_breastcancer,
    # load_car,
    # load_cardio,
    # load_churn,
    # load_default_cb,
    # load_letter,
    # load_satimage,
    # load_sensorless,
    # load_spambase,
    # load_covtype,
    # load_higgs,
    # load_kddcup
]

# TODO: wildwood avec et sans aggregation ?

random_state = 42


def fit_kwargs_generator(clf_name, dataset):
    if clf_name == "RandomForestClassifier":
        return {}
    elif clf_name == "WildWood":
        return {"categorical_features": dataset.categorical_features_}
    elif clf_name == "XGBClassifier":
        return {}
    elif clf_name == "CatBoostClassifier":
        return {"cat_features": dataset.categorical_columns, "verbose":False}
    elif clf_name == "LGBMClassifier":
        return {"categorical_feature": "auto"}
    elif clf_name == "HistGradientBoostingClassifier":
        return {}
    else:
        print("ERROR : NOT Found : ", clf_name)



def set_classifier(clf_name, categorical_features, fit_seed, n_jobs=-1):
    classifier_setting = {
        "RandomForestClassifier": RandomForestClassifier(n_jobs=n_jobs, random_state=fit_seed),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(categorical_features=categorical_features, random_state=fit_seed),
        "XGBClassifier": xgb.XGBClassifier(use_label_encoder=False, n_jobs=n_jobs, random_state=fit_seed),
        "LGBMClassifier": lgb.LGBMClassifier(n_jobs=n_jobs, random_state=fit_seed),
        "CatBoostClassifier": CatBoostClassifier(thread_count=n_jobs, random_state=fit_seed),
        "WildWood": ForestClassifier(n_jobs=n_jobs, random_state=fit_seed),
    }
    return classifier_setting[clf_name]


data_extraction = {
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
        "one_hot_encode": False,
        "standardize": False,
        "drop": None,
        "pd_df_categories": True,
    },
    "WildWood": {
        "one_hot_encode": False,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
}

clf_names = [
    "LGBMClassifier",
    "XGBClassifier",
    "CatBoostClassifier",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "WildWood"
]

n_estimators = 100

random_state_seed = 42
fit_seeds = [0, 1, 2, 3, 4]

# Number of time each experiment is repeated, one for each seed (leading
# data_random_states = [42, 43, 44, 46, 47, 49, 50, 52, 53, 55]
data_random_states = [42]
clf_random_state = 42



if not os.path.exists("results"):
    os.mkdir("results")
results_home_path = "results/"

random_states = {
    "data_extract_random_state": random_state_seed,
    "train_val_split_random_state": 1 + random_state_seed,
    "expe_random_state": 2 + random_state_seed,
}


for loader in loaders:

    logging.info("=" * 128)
    dataset = loader()
    learning_task = dataset.task
    logging.info("Launching experiments for %s" % dataset.name)

    if not os.path.exists(results_home_path + dataset.name):
        os.mkdir(results_home_path + dataset.name)
    results_dataset_path = results_home_path + dataset.name + "/"

    col_data = []
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

    for clf_name in clf_names:
        for key, val in data_extraction[clf_name].items():
            setattr(dataset, key, val)

        X_train, X_test, y_train, y_test = dataset.extract(
            random_state=random_states["data_extract_random_state"]
        )

        print("Run fitting with " + clf_name + " ...")

        for rep, fit_seed in enumerate(fit_seeds):
            clf = set_classifier(clf_name, dataset.categorical_features_, fit_seed)

            if clf_name == "WildWood":
                clf.fit(X_train[:100], y_train[:100])
            tic = timer()
            clf.fit(
                X_train,
                y_train,
                **fit_kwargs_generator(clf_name, dataset)
            )
            toc = timer()
            fit_time = toc - tic
            logging.info("Fitted %s in %.2f seconds" % (clf_name, fit_time))
            col_fit_time.append(fit_time)
            # col_fit_time.append(fit_time)
            y_scores_train = clf.predict_proba(X_train)
            tic = timer()
            y_scores = clf.predict_proba(X_test)
            toc = timer()
            predict_time = toc - tic
            col_predict_time.append(predict_time)
            # col_predict_time.append(predict_time)
            logging.info("Predict %s in %.2f seconds" % (clf_name, predict_time))
            y_pred = np.argmax(y_scores, axis=1)
            y_pred_train = np.argmax(y_scores_train, axis=1)
            y_train_binary = LabelBinarizer().fit_transform(y_train)
            y_test_binary = LabelBinarizer().fit_transform(y_test)

            if learning_task == "binary-classification":
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
            elif learning_task == "multiclass-classification":
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
            else:
                raise ValueError(
                    "Task %s not understood" % learning_task
                )  # TODO: regression

            col_data.append(dataset.name)
            col_classifier.append(clf_name)

            col_roc_auc.append(roc_auc)
            col_roc_auc_train.append(roc_auc_train)
            col_roc_auc_weighted.append(roc_auc_weighted)
            col_roc_auc_weighted_train.append(roc_auc_weighted_train)
            col_avg_precision_score.append(avg_precision_score)
            col_avg_precision_score_train.append(avg_precision_score_train)
            col_avg_precision_score_weighted.append(avg_precision_score_weighted)
            col_avg_precision_score_weighted_train.append(avg_precision_score_weighted_train)
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
                    rep
                )
            )


    results = pd.DataFrame(
        {
            "dataset": col_data,
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
            "repeat": col_repeat
        }
    )

    print(results)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = "exp_default_params_" + now + ".pickle"

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