# License: BSD 3 clause

"""
Benchmarks used for internal development of wildwood. This runs benchmarks for
ForestClassifier only, following the configuration from `config.yaml` file and saves
results in mlflow.
"""
import logging
import sys
import os
import yaml
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)

from itertools import product, starmap
from collections import namedtuple

import mlflow


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y/%m/%d %I:%M:%S",
)


sys.path.extend([".", ".."])

from wildwood.forest import ForestClassifier  # noqa: E402
from wildwood.datasets import loader_from_name


CONFIG_FILE = "config.yaml"
RUNS_CONFIG_DIR = "runs_configs/"
EXPERIMENT_NAME = "ww_bench_v1"


def read_yaml(filename):
    with open(filename, "r") as f:
        try:
            contents = yaml.safe_load(f)
            return contents
        except yaml.YAMLError as exc:
            logging.error(exc)


def dict_product(**items):
    """Computes the cartesian product of a list-valued dictionary

    Parameters
    ----------
    items : dict
        A dictionary mapping keys to list of values

    Returns
    -------
    output : list
        A list containing dict corresponding to the cartesian products of all values
        in the lists
    """
    Product = namedtuple("Product", items.keys())
    products = starmap(Product, product(*items.values()))
    return list(map(lambda nt: nt._asdict(), products))


def compute_metrics(
    *,
    learning_task,
    y_train,
    y_test,
    y_scores_train,
    y_scores_test,
    y_pred_train,
    y_pred_test,
):
    """Compute evaluation metrics for several learning tasks.
    """
    if learning_task == "binary-classification":
        roc_auc_train = roc_auc_score(y_train, y_scores_train[:, 1])
        roc_auc_test = roc_auc_score(y_test, y_scores_test[:, 1])

        roc_auc_w_train = roc_auc_train
        roc_auc_w_test = roc_auc_test

        avg_pr_train = average_precision_score(y_train, y_scores_train[:, 1])
        avg_pr_test = average_precision_score(y_test, y_scores_test[:, 1])

        avg_pr_w_train = avg_pr_train
        avg_pr_w_test = avg_pr_test

        log_loss_train = log_loss(y_train, y_scores_train)
        log_loss_test = log_loss(y_test, y_scores_test)

        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)

    elif learning_task == "multiclass-classification":
        y_train_binary = LabelBinarizer().fit_transform(y_train)
        y_test_binary = LabelBinarizer().fit_transform(y_test)

        roc_auc_train = roc_auc_score(
            y_train, y_scores_train, multi_class="ovr", average="macro"
        )

        roc_auc_test = roc_auc_score(
            y_test, y_scores_test, multi_class="ovr", average="macro"
        )

        roc_auc_w_train = roc_auc_score(
            y_train, y_scores_train, multi_class="ovr", average="weighted"
        )
        roc_auc_w_test = roc_auc_score(
            y_test, y_scores_test, multi_class="ovr", average="weighted"
        )

        avg_pr_train = average_precision_score(y_train_binary, y_scores_train)
        avg_pr_test = average_precision_score(y_test_binary, y_scores_test)

        avg_pr_w_train = average_precision_score(
            y_train_binary, y_scores_train, average="weighted"
        )
        avg_pr_w_test = average_precision_score(
            y_test_binary, y_scores_test, average="weighted"
        )

        log_loss_train = log_loss(y_train, y_scores_train)
        log_loss_test = log_loss(y_test, y_scores_test)

        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
    else:
        raise ValueError("Task %s not understood" % learning_task)

    return dict(
        roc_auc_train=roc_auc_train,
        roc_auc_test=roc_auc_test,
        roc_auc_w_train=roc_auc_w_train,
        roc_auc_w_test=roc_auc_w_test,
        avg_pr_train=avg_pr_train,
        avg_pr_test=avg_pr_test,
        avg_pr_w_train=avg_pr_w_train,
        avg_pr_w_test=avg_pr_w_test,
        log_loss_train=log_loss_train,
        log_loss_test=log_loss_test,
        accuracy_train=accuracy_train,
        accuracy_test=accuracy_test,
    )


def launch_run(*, run_config, experiment_id):
    """

    Parameters
    ----------
    run_config : dict
        The configuration of the run

    experiment_id : str
        Id of the experiment that groups runs in mlflow

    Returns
    -------
    output : dict
        Metrics computed during this run
    """

    wildwood_kwargs = {
        key.replace("wildwood_", ""): val
        for key, val in run_config.items()
        if key.startswith("wildwood")
    }

    dataset_name = run_config["dataset"]
    dataset_random_state = run_config["dataset_random_state"]
    loader = loader_from_name[dataset_name]

    # Just get the task from the dataset
    dataset = loader()
    learning_task = dataset.task

    # But we use the raw data in wildwood
    X, y = loader(raw=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=dataset_random_state, shuffle=True, stratify=y
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    kwargs_one_tree = wildwood_kwargs.copy()
    kwargs_one_tree["n_estimators"] = 1

    # Fit a single tree on the full dataset to force pre-compilation (doing so on a
    # subset often fails).
    # TODO: debug such cases
    clf = ForestClassifier(**kwargs_one_tree)
    clf.fit(X_train, y_train)

    # Instantiate again just to be sure
    clf = ForestClassifier(**wildwood_kwargs)

    with mlflow.start_run(experiment_id=experiment_id):
        # Fit and timing
        tic = time()
        # clf.fit(X_train, y_train, **fit_kwargs_generator(clf_name, dataset_name))
        # TODO: include computations with an without categorical features ?
        clf.fit(X_train, y_train)

        toc = time()
        fit_time = toc - tic
        logging.info(f"Fitted for experiment {filename} in {fit_time}s")

        # Predict and timing
        tic = time()
        y_scores_train = clf.predict_proba(X_train)
        toc = time()
        predict_train_time = toc - tic

        tic = time()
        y_scores_test = clf.predict_proba(X_test)
        toc = time()
        predict_test_time = toc - tic

        # col_predict_time.append(predict_time)
        logging.info(
            f"Predicted for experiment {filename} on train in {predict_train_time}s and test in {predict_test_time}s"
        )

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        metrics = compute_metrics(
            learning_task=learning_task,
            y_train=y_train,
            y_test=y_test,
            y_scores_train=y_scores_train,
            y_scores_test=y_scores_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
        )

        mlflow_metrics = dict(
            **metrics,
            fit_time=fit_time,
            predict_train_time=predict_train_time,
            predict_test_time=predict_test_time,
        )

        mlflow_params = dict(
            **wildwood_kwargs,
            dataset=dataset_name,
            dataset_random_state=dataset_random_state,
        )

        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(mlflow_metrics)


def create_runs_configs(*, config_file, runs_config_dir):
    """Create the yaml configuration files from the config file describing all
    experiments.

    Parameters
    ----------
    config_file : str
        The path to the yaml file containing the configuration of the experiment

    runs_config_dir : str
        Path of the directory where the configuration files of the runs are saved
    """
    config = read_yaml(config_file)
    logging.info(f"Read {config_file}")

    parameters = dict(
        **{f"wildwood_{key}": val for key, val in config["hyper_parameters"].items()},
        dataset=config["datasets"]["names"],
        dataset_random_state=config["datasets"]["random_states"],
    )

    runs_configs = dict_product(**parameters)
    n_runs = len(runs_configs)
    max_run_idx = n_runs - 1
    n_digits = len(str(max_run_idx))

    logging.info(f"Found {n_runs} runs")

    if not os.path.exists(runs_config_dir):
        os.makedirs(runs_config_dir)
        logging.info(f"Created {runs_config_dir} directory")

    for run_idx, run_config in enumerate(runs_configs):
        run_config["config"] = CONFIG_FILE
        run_idx_formatted = str(run_idx).zfill(n_digits)
        filename = os.path.join(RUNS_CONFIG_DIR, f"run_{run_idx_formatted}.yaml")
        with open(filename, "w") as f:
            yaml.dump(run_config, f)
            logging.info(f"Wrote file {filename}")


if __name__ == "__main__":
    create_runs_configs(config_file=CONFIG_FILE, runs_config_dir=RUNS_CONFIG_DIR)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    for filename in sorted(os.listdir(RUNS_CONFIG_DIR)):
        run_config_filename = os.path.join(RUNS_CONFIG_DIR, filename)
        run_config = read_yaml(run_config_filename)
        logging.info(f"**** Launching run for config file {run_config_filename} ****")
        launch_run(run_config=run_config, experiment_id=experiment_id)
        logging.info(f"**** Finished run for config file {run_config_filename} ****")
        # Once the run is finished, we can remove the file
        os.remove(run_config_filename)
        logging.info(f"**** Removed config file {run_config_filename} ****")
