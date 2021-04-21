import sys
import subprocess
from datetime import datetime
import pickle as pkl
import numpy as np
import logging
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


sys.path.extend([".", ".."])

from wildwood.dataset import loaders_small_classification
from wildwood.forest import ForestClassifier


def get_train_sample_weights(y_train, n_classes):
    def get_class_proportions(data):
        return np.bincount(data.astype(int)) / len(data)

    return (1 / (n_classes * get_class_proportions(y_train)))[y_train]


classifiers = [
    lambda n: (
        "RFW",
        RandomForestClassifier(
            n_estimators=n,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    ),
    lambda n: (
        "WildWood",
        ForestClassifier(
            n_estimators=n,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    ),
    lambda n: (
        "CatBoost",
        CatBoostClassifier(n_estimators=n, thread_count=-1, random_state=random_state),
    ),
    lambda n: (
        "LightGBM",
        lgb.LGBMClassifier(n_estimators=n, random_state=random_state, n_jobs=-1),
    ),
    lambda n: (
        "XGBoost",
        xgb.XGBClassifier(
            n_estimators=n,
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
        ),
    ),
]

random_state = 42


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

data_extraction = {
    "RandomForestClassifier": {
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
    "ForestClassifier": {
        "one_hot_encode": True,
        "standardize": False,
        "drop": None,
        "pd_df_categories": False,
    },
    "CatBoostClassifier": {
        "one_hot_encode": False,
        "standardize": False,
        "drop": None,
        "pd_df_categories": True,
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
}


def fit_kwargs_generator(clf_name, y_train, dataset):
    if clf_name == "RandomForestClassifier":
        return {}
    elif clf_name == "ForestClassifier":
        return {"categorical_features": dataset.categorical_features_}
    elif clf_name == "CatBoostClassifier":
        return {
            "sample_weight": get_train_sample_weights(y_train, dataset.n_classes_),
            "cat_features": dataset.categorical_columns,
        }
    elif clf_name == "LGBMClassifier":
        return {
            "sample_weight": get_train_sample_weights(y_train, dataset.n_classes_),
            "categorical_feature": dataset.categorical_columns,
        }
    elif clf_name == "XGBClassifier":
        return {"sample_weight": get_train_sample_weights(y_train, dataset.n_classes_)}
    else:
        print("ERROR : NOT Found : ", clf_name)


# Number of time each experiment is repeated, one for each seed (leading
data_random_states = [42, 43, 44]  # , 46, 47, 49, 50, 52, 53, 55]
# data_random_states = [42]
clf_random_state = 42

col_data = []
col_classifier = []
col_classifier_title = []
col_n_trees = []
col_repeat = []
col_roc_auc = []
col_roc_auc_weighted = []
col_avg_precision_score = []
col_avg_precision_score_weighted = []
col_log_loss = []
col_accuracy = []

n_datasets = 3  # set to None to use all
n_treess = [1, 2, 5, 10, 20, 50]

for n in n_treess:
    for Clf in classifiers:
        clf_title, clf = Clf(n)
        clf_name = clf.__class__.__name__

        for loader in loaders_small_classification[:n_datasets]:
            dataset = loader()
            data_name = dataset.name
            task = dataset.task
            for key, val in data_extraction[clf_name].items():
                setattr(dataset, key, val)
            logging.info("-" * 64)
            logging.info("Launching task for dataset %r" % dataset)
            for repeat, data_random_state in enumerate(data_random_states):
                clf_title, clf = Clf(n)
                logging.info(
                    "Repeat: %d random_state: %d" % (repeat, data_random_state)
                )
                col_data.append(data_name)
                col_classifier.append(clf_name)
                col_classifier_title.append(clf_title)
                col_n_trees.append(n)
                col_repeat.append(repeat)

                X_train, X_test, y_train, y_test = dataset.extract(
                    random_state=data_random_state
                )
                y_test_binary = LabelBinarizer().fit_transform(y_test)

                clf.fit(
                    X_train,
                    y_train,
                    **(fit_kwargs_generator(clf_name, y_train, dataset))
                )

                y_scores = clf.predict_proba(X_test)

                y_pred = np.argmax(y_scores, axis=1)

                if task == "binary-classification":
                    roc_auc = roc_auc_score(y_test, y_scores[:, 1])
                    roc_auc_weighted = roc_auc
                    avg_precision_score = average_precision_score(
                        y_test, y_scores[:, 1]
                    )
                    avg_precision_score_weighted = avg_precision_score
                    log_loss_ = log_loss(y_test, y_scores)
                    accuracy = accuracy_score(y_test, y_pred)
                elif task == "multiclass-classification":
                    roc_auc = roc_auc_score(
                        y_test, y_scores, multi_class="ovr", average="macro"
                    )
                    roc_auc_weighted = roc_auc_score(
                        y_test, y_scores, multi_class="ovr", average="weighted"
                    )
                    avg_precision_score = average_precision_score(
                        y_test_binary, y_scores
                    )
                    avg_precision_score_weighted = average_precision_score(
                        y_test_binary, y_scores, average="weighted"
                    )
                    log_loss_ = log_loss(y_test, y_scores)
                    accuracy = accuracy_score(y_test, y_pred)
                # TODO: regression
                else:
                    raise ValueError("Task %s not understood" % task)

                col_roc_auc.append(roc_auc)
                col_roc_auc_weighted.append(roc_auc_weighted)
                col_avg_precision_score.append(avg_precision_score)
                col_avg_precision_score_weighted.append(avg_precision_score_weighted)
                col_log_loss.append(log_loss_)
                col_accuracy.append(accuracy)


results = pd.DataFrame(
    {
        "dataset": col_data,
        "classifier": col_classifier,
        "classifier_title": col_classifier_title,
        "repeat": col_repeat,
        "n_trees": col_n_trees,
        "roc_auc": col_roc_auc,
        "roc_auc_w": col_roc_auc_weighted,
        "avg_prec": col_avg_precision_score,
        "avg_prec_w": col_avg_precision_score_weighted,
        "log_loss": col_log_loss,
        "accuracy": col_accuracy,
    }
)

print(results)

now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

# Get the commit number as a string
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
commit = commit.decode("utf-8").strip()

with open("ntrees_experiment_" + now + ".pickle", "wb") as f:
    pkl.dump({"datetime": now, "commit": commit, "results": results}, f)
