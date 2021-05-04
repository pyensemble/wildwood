import sys
import subprocess
from time import time
from datetime import datetime
import logging
import pickle as pkl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
from wildwood.dataset import (  # noqa: E402
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
from wildwood.dataset._amazon import load_amazon  # noqa: E402

from experiments.experiment import (  # noqa: E402
    LogRegExperiment,
    RFExperiment,
    HGBExperiment,
    LGBExperiment,
    XGBExperiment,
    CABExperiment,
)

# import xgboost as xgb
# from catboost import CatBoostClassifier
# import lightgbm as lgb
# from wildwood.forest import ForestClassifier


def get_train_sample_weights(labels, n_classes):
    # TODO: maybe put this function in _utils.py?
    def get_class_proportions(data):
        return np.bincount(data.astype(int)) / len(data)

    return (1 / (n_classes * get_class_proportions(labels)))[labels]


data_extraction = {
    "LogisticRegression": {
        "one_hot_encode": True,
        "standardize": True,
        "drop": "first",
        "pd_df_categories": False,
    },
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
    # "WildWood": {
    #     "one_hot_encode": False,
    #     "standardize": False,
    #     "drop": None,
    #     "pd_df_categories": False,
    # },
}


def set_experiment(
    clf_name,
    learning_task,
    n_estimators,
    max_hyperopt_eval,
    categorical_columns,
    expe_random_states,
):
    experiment_setting = {
        "LogisticRegression": LogRegExperiment(
            learning_task,
            max_hyperopt_evals=max_hyperopt_eval,
            random_state=expe_random_states,
        ),
        "RandomForestClassifier": RFExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            random_state=expe_random_states,
        ),
        "HistGradientBoostingClassifier": HGBExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            categorical_features=categorical_columns,
            random_state=expe_random_states,
        ),
        "XGBClassifier": XGBExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            random_state=expe_random_states,
        ),
        "LGBMClassifier": LGBExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            categorical_features=categorical_columns,
            random_state=expe_random_states,
        ),
        "CatBoostClassifier": CABExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            categorical_features=categorical_columns,
            random_state=expe_random_states,
        ),
    }
    return experiment_setting[clf_name]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# TODO: add big datasets here
loaders = [
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
]

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

"""
SET UP HERE
"""
clf_names = [
    "LGBMClassifier",
    "XGBClassifier",
    "CatBoostClassifier",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
]
# TODO: LogisticRegression
n_estimators = 5000
max_hyperopt_eval = 100
do_class_weights = False

# random_states = [42, 43, 44, 46, 47, 49, 50, 52, 53, 55]
random_state_seed = 42


"""
SET UP HERE
"""

random_states = {
    "data_extract_random_state": random_state_seed,
    "train_val_split_random_state": 1 + random_state_seed,
    "expe_random_state": 2 + random_state_seed,
}

for clf_name in clf_names:  # for clf_name in clf_names[:2]:

    logging.info("=" * 128)
    logging.info("Launching experiments for %s" % clf_name)

    for loader in loaders:  # for loader in loaders[:2]:
        dataset = loader()
        learning_task = dataset.task
        col_data.append(dataset.name)
        col_classifier.append(clf_name)

        for key, val in data_extraction[clf_name].items():
            setattr(dataset, key, val)

        X_train, X_test, y_train, y_test = dataset.extract(
            random_state=random_states["data_extract_random_state"]
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=random_states["train_val_split_random_state"],
            stratify=y_train
            if learning_task in ["binary-classification", "multiclass-classification"]
            else None,
        )
        sample_weights_tr = get_train_sample_weights(y_tr, dataset.n_classes_)

        exp = set_experiment(
            clf_name,
            learning_task,
            n_estimators,
            max_hyperopt_eval,
            dataset.categorical_columns,
            random_states["expe_random_state"],
        )

        print("Run train-val hyperopt exp...")
        tuned_cv_result = exp.optimize_params(
            X_tr,
            y_tr,
            X_val,
            y_val,
            max_evals=max_hyperopt_eval,
            sample_weight=sample_weights_tr if do_class_weights else None,
            verbose=True,
        )

        print("Run fitting with tuned params...")
        tic = time()
        model = exp.fit(
            tuned_cv_result["params"],
            X_train,
            y_train,
            sample_weight=get_train_sample_weights(y_train, dataset.n_classes_)
            if do_class_weights
            else None,
        )
        toc = time()
        fit_time = toc - tic
        logging.info("Fitted %s in %.2f seconds on " % (clf_name, fit_time))
        col_fit_time.append(fit_time)
        tic = time()
        y_scores = model.predict_proba(X_test)
        y_scores_train = model.predict_proba(X_train)
        toc = time()
        predict_time = toc - tic
        col_predict_time.append(predict_time)
        logging.info("Predict %s in %.2f seconds" % (clf_name, fit_time))
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
    }
)

print(results)

now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

# Get the commit number as a string
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
commit = commit.decode("utf-8").strip()

filename = "exp_hyperopt_" + now + ".pickle"
with open(filename, "wb") as f:
    pkl.dump(
        {
            "datetime": now,
            "commit": commit,
            "n_estimators": n_estimators,
            "max_hyperopt_eval": max_hyperopt_eval,
            "do_class_weights": do_class_weights,
            "results": results,
        },
        f,
    )

logging.info("Saved results in file %s" % filename)
