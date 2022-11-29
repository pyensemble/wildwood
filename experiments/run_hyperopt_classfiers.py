# License: BSD 3 clause

"""
This script produces hyper-parameters optimization experiments (Table 1 and 3) from the WildWood's paper.

"""


import sys
import os
import subprocess
from time import time
from datetime import datetime
import logging
import pickle as pkl
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer


sys.path.extend([".", ".."])
from wildwood.datasets import (  # noqa: E402
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
    load_amazon,
    load_covtype,
    load_kick,
    load_internet,
    load_higgs,
    load_kddcup99,
)

from experiment import (  # noqa: E402
    RFExperiment,
    HGBExperiment,
    LGBExperiment,
    XGBExperiment,
    CABExperiment,
    WWExperiment,
)


def get_train_sample_weights(labels, n_classes):
    def get_class_proportions(data):
        return np.bincount(data.astype(int)) / len(data)

    return (1 / (n_classes * get_class_proportions(labels)))[labels]


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


def set_experiment(
    clf_name,
    learning_task,
    n_estimators,
    max_hyperopt_eval,
    early_stopping_round,
    categorical_columns,
    categorical_features,
    expe_random_states,
    use_gpu,
    output_folder_path,
):
    experiment_setting = {
        "RandomForestClassifier": RFExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=10,
            random_state=expe_random_states,
            output_folder_path=output_folder_path,
        ),
        "HistGradientBoostingClassifier": HGBExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            categorical_features=categorical_features,
            random_state=expe_random_states,
            output_folder_path=output_folder_path,
        ),
        "XGBClassifier": XGBExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            early_stopping_round=early_stopping_round,
            random_state=expe_random_states,
            use_gpu=use_gpu,
            output_folder_path=output_folder_path,
        ),
        "LGBMClassifier": LGBExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            early_stopping_round=early_stopping_round,
            categorical_features=categorical_columns,
            random_state=expe_random_states,
            use_gpu=use_gpu,
            output_folder_path=output_folder_path,
        ),
        "CatBoostClassifier": CABExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            early_stopping_round=early_stopping_round,
            categorical_features=categorical_columns,
            random_state=expe_random_states,
            use_gpu=use_gpu,
            output_folder_path=output_folder_path,
        ),
        "WildWood": WWExperiment(
            learning_task,
            n_estimators=n_estimators,
            max_hyperopt_evals=max_hyperopt_eval,
            categorical_features=categorical_features,
            random_state=expe_random_states,
            output_folder_path=output_folder_path,
        ),
    }
    return experiment_setting[clf_name]


def set_dataloader(dataset_name):
    loaders_mapping = {
        "adult": load_adult,
        "bank": load_bank,
        "breastcancer": load_breastcancer,
        "car": load_car,
        "cardio": load_cardio,
        "churn": load_churn,
        "default-cb": load_default_cb,
        "letter": load_letter,
        "satimage": load_satimage,
        "sensorless": load_sensorless,
        "spambase": load_spambase,
        "amazon": load_amazon,
        "covtype": load_covtype,
        "internet": load_internet,
        "kick": load_kick,
        "kddcup": load_kddcup99,
        "higgs": load_higgs,
    }
    return loaders_mapping[dataset_name]


def run_hyperopt(
    dataset,
    clf_name,
    learning_task,
    n_estimators,
    max_hyperopt_eval,
    early_stopping_round,
    do_class_weights,
    use_gpu,
    results_dataset_path,
):
    col_data = []
    col_classifier = []
    col_fit_time = []
    col_fit_time_std = []
    col_predict_time = []
    col_roc_auc = []
    col_roc_auc_weighted = []
    col_avg_precision_score = []
    col_avg_precision_score_weighted = []
    col_log_loss = []
    col_accuracy = []
    col_roc_auc_std = []
    col_log_loss_std = []

    col_roc_auc_train = []
    col_roc_auc_weighted_train = []
    col_avg_precision_score_train = []
    col_avg_precision_score_weighted_train = []
    col_log_loss_train = []
    col_accuracy_train = []

    col_data.append(dataset.name)
    col_classifier.append(clf_name)

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
        early_stopping_round,
        dataset.categorical_columns,
        dataset.categorical_features_,
        random_states["expe_random_state"],
        use_gpu,
        results_dataset_path,
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
    fit_time_list, predict_time_list = [], []
    (
        roc_auc_list,
        roc_auc_train_list,
        roc_auc_weighted_list,
        roc_auc_weighted_train_list,
        avg_precision_score_list,
        avg_precision_score_train_list,
        avg_precision_score_weighted_list,
        avg_precision_score_weighted_train_list,
        log_loss_list,
        log_loss_train_list,
        accuracy_list,
        accuracy_train_list,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])

    for fit_seed in fit_seeds:
        tic = time()
        model, _ = exp.fit(
            tuned_cv_result["params"],
            X_train,
            y_train,
            None,
            n_estimators=tuned_cv_result["best_n_estimators"],
            sample_weight=get_train_sample_weights(y_train, dataset.n_classes_)
            if do_class_weights
            else None,
            seed=fit_seed,
        )
        toc = time()
        fit_time = toc - tic
        logging.info("Fitted %s in %.2f seconds" % (clf_name, fit_time))
        fit_time_list.append(fit_time)
        # col_fit_time.append(fit_time)
        tic = time()
        y_scores = model.predict_proba(X_test)
        y_scores_train = model.predict_proba(X_train)
        toc = time()
        predict_time = toc - tic
        predict_time_list.append(predict_time)
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
            raise ValueError("Task %s not understood" % learning_task)

        roc_auc_list.append(roc_auc)
        roc_auc_train_list.append(roc_auc_train)
        roc_auc_weighted_list.append(roc_auc_weighted)
        roc_auc_weighted_train_list.append(roc_auc_weighted_train)
        avg_precision_score_list.append(avg_precision_score)
        avg_precision_score_train_list.append(avg_precision_score_train)
        avg_precision_score_weighted_list.append(avg_precision_score_weighted)
        avg_precision_score_weighted_train_list.append(
            avg_precision_score_weighted_train
        )
        log_loss_list.append(log_loss_)
        log_loss_train_list.append(log_loss_train_)
        accuracy_list.append(accuracy)
        accuracy_train_list.append(accuracy_train)

    roc_auc, roc_auc_train, roc_auc_weighted, roc_auc_weighted_train = (
        np.mean(roc_auc_list),
        np.mean(roc_auc_train_list),
        np.mean(roc_auc_weighted_list),
        np.mean(roc_auc_weighted_train_list),
    )
    (
        avg_precision_score,
        avg_precision_score_train,
        avg_precision_score_weighted,
        avg_precision_score_weighted_train,
    ) = (
        np.mean(avg_precision_score_list),
        np.mean(avg_precision_score_train_list),
        np.mean(avg_precision_score_weighted_list),
        np.mean(avg_precision_score_weighted_train_list),
    )
    log_loss_, log_loss_train_, accuracy, accuracy_train = (
        np.mean(log_loss_list),
        np.mean(log_loss_train_list),
        np.mean(accuracy_list),
        np.mean(accuracy_train_list),
    )

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

    col_fit_time.append(np.mean(fit_time_list))
    col_fit_time_std.append(np.std(fit_time_list))
    col_predict_time.append(np.mean(predict_time_list))
    col_log_loss_std.append(np.std(log_loss_list))
    col_roc_auc_std.append(np.std(roc_auc_list))

    logging.info(
        "AUC= %.2f, AUCW: %.2f, AVGP: %.2f, AVGPW: %.2f, LOGL: %.2f, ACC: %.2f"
        % (
            float(roc_auc),
            float(roc_auc_weighted),
            float(avg_precision_score),
            float(avg_precision_score_weighted),
            float(log_loss_),
            float(accuracy),
        )
    )

    results = pd.DataFrame(
        {
            "dataset": col_data,
            "classifier": col_classifier,
            "fit_time": col_fit_time,
            "fit_time_std": col_fit_time_std,
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
            "log_loss_std": col_log_loss_std,
            "roc_auc_std": col_roc_auc_std,
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
    parser.add_argument("-t", "--n_estimators", type=int, default=100)
    parser.add_argument("-n", "--hyperopt_evals", type=int, default=50)
    parser.add_argument("-o", "--output_folder_path", default=None)
    parser.add_argument("--do_class_weights", type=bool, default=False)
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--early_stopping_round", type=int, default=5)
    parser.add_argument("--random_state_seed", type=int, default=42)

    args = parser.parse_args()

    clf_name = args.clf_name
    loader = set_dataloader(args.dataset_name)
    n_estimators = args.n_estimators
    max_hyperopt_eval = args.hyperopt_evals
    do_class_weights = args.do_class_weights
    early_stopping_round = args.early_stopping_round
    use_gpu = args.use_gpu
    random_state_seed = args.random_state_seed

    if args.output_folder_path is None:
        if not os.path.exists("results"):
            os.mkdir("results")
        results_home_path = "results/"
    else:
        results_home_path = args.output_folder_path

    random_states = {
        "data_extract_random_state": random_state_seed,
        "train_val_split_random_state": 1 + random_state_seed,
        "expe_random_state": 2 + random_state_seed,
    }
    fit_seeds = [0, 1, 2, 3, 4]

    logging.info("=" * 128)
    dataset = loader()
    learning_task = dataset.task
    logging.info("Launching experiments for %s" % dataset.name)

    if not os.path.exists(results_home_path + dataset.name):
        os.mkdir(results_home_path + dataset.name)
    results_dataset_path = results_home_path + dataset.name + "/"

    results = run_hyperopt(
        dataset,
        clf_name,
        learning_task,
        n_estimators,
        max_hyperopt_eval,
        early_stopping_round,
        do_class_weights,
        use_gpu,
        results_dataset_path,
    )

    print(results)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = (
        "exp_hyperopt_"
        + str(n_estimators)
        + "_"
        + str(max_hyperopt_eval)
        + "_"
        + ("W" if do_class_weights else "N")
        + "_"
        + now
        + ".pickle"
    )

    with open(results_dataset_path + filename, "wb") as f:
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

    logging.info("Saved results in file %s" % results_dataset_path + filename)
