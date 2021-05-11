from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import numpy as np
import os
import time
from datetime import datetime
import pickle as pkl

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer

from sklearn.experimental import enable_hist_gradient_boosting

#  This estimator is still experimental in sklearn 0.24
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# TODO: to add regressors for every Experiment


class Experiment(object):
    """
    inspired from
    https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/experiment.py
    """

    def __init__(
        self,
        learning_task="binary-classification",
        bst_name=None,
        n_estimators=100,
        hyperopt_evals=50,
        categorical_features=None,
        early_stopping_round=5,
        random_state=0,
        output_folder_path="./",
        use_gpu=False,
        verbose=True,
    ):
        self.learning_task_, self.bst_name = learning_task, bst_name
        if learning_task in ["binary-classification", "multiclass-classification"]:
            self.learning_task = "classification"
        else:
            self.learning_task = learning_task

        self.n_estimators, self.best_loss = n_estimators, np.inf
        self.categorical_features = categorical_features
        self.early_stopping_round = early_stopping_round
        self.hyperopt_evals, self.hyperopt_eval_num = hyperopt_evals, 0
        self.output_folder_path = os.path.join(
            output_folder_path, ""  # TODO: put dataset name
        )  # TODO: write output
        self.default_params, self.best_params = None, None
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.best_n_estimators = None

        # to specify definitions in particular experiments
        self.title = None
        self.space = None
        self.trials = None

        if self.learning_task == "classification":
            self.metric = "logloss"
        elif self.learning_task == "regression":
            self.metric = "rmse"
        else:
            raise ValueError('Task must be "classification" or "regression"')

    def optimize_params(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        sample_weight,
        max_evals=None,
        verbose=True,
    ):
        max_evals = max_evals or self.hyperopt_evals
        self.trials = Trials()
        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        _ = fmin(
            fn=lambda params: self.run(
                X_train,
                y_train,
                X_val,
                y_val,
                sample_weight,
                params,
                n_estimators=self.n_estimators,
                verbose=verbose,
            ),
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
        )

        self.best_params = self.trials.best_trial["result"]["params"]
        self.best_n_estimators = self.trials.best_trial["result"]["best_n_estimators"]
        # TODO: recupere best_n_estimators ici
        if self.verbose:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = (
                "best_params_results_"
                + str(self.bst_name)
                + "_"
                + str(self.hyperopt_evals)
                + "_"
                + now
                + ".pickle"
            )

            with open(self.output_folder_path + filename, "wb") as f:
                pkl.dump(
                    {
                        "datetime": now,
                        "max_hyperopt_eval": self.hyperopt_evals,
                        "best_n_estimators": self.best_n_estimators,
                        "result": self.trials.best_trial["result"],
                    },
                    f,
                )
        return self.trials.best_trial["result"]

    def run(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        sample_weight,
        params=None,
        n_estimators=None,
        verbose=False,
    ):
        params = params or self.default_params
        if n_estimators is not None:
            self.n_estimators = n_estimators
        params = self.preprocess_params(params)
        start_time = time.time()
        bst, n_best_iteration = self.fit(
            params, X_train, y_train, (X_val, y_val), sample_weight, seed=None
        )
        fit_time = time.time() - start_time
        y_scores = self.predict(bst, X_val)
        evals_result = log_loss(y_val, y_scores)
        y_pred = np.argmax(y_scores, axis=1)

        results = {
            "loss": evals_result,
            "fit_time": fit_time,
            "best_n_estimators": n_best_iteration,
            "status": STATUS_FAIL if np.isnan(evals_result) else STATUS_OK,
            "params": params.copy(),
        }

        if self.learning_task_ == "binary-classification":
            roc_auc = roc_auc_score(y_val, y_scores[:, 1])
            roc_auc_weighted = roc_auc
            avg_precision_score = average_precision_score(y_val, y_scores[:, 1])
            avg_precision_score_weighted = avg_precision_score
            log_loss_ = log_loss(y_val, y_scores)
            accuracy = accuracy_score(y_val, y_pred)
            results.update(
                {
                    "roc_auc": roc_auc,
                    "roc_auc_weighted": roc_auc_weighted,
                    "avg_precision_score": avg_precision_score,
                    "avg_precision_score_weighted": avg_precision_score_weighted,
                    "log_loss": log_loss_,
                    "accuracy": accuracy,
                }
            )
        elif self.learning_task_ == "multiclass-classification":
            y_val_binary = LabelBinarizer().fit_transform(y_val)
            roc_auc = roc_auc_score(y_val, y_scores, multi_class="ovr", average="macro")
            roc_auc_weighted = roc_auc_score(
                y_val, y_scores, multi_class="ovr", average="weighted"
            )
            avg_precision_score = average_precision_score(y_val_binary, y_scores)
            avg_precision_score_weighted = average_precision_score(
                y_val_binary, y_scores, average="weighted"
            )
            log_loss_ = log_loss(y_val, y_scores)
            accuracy = accuracy_score(y_val, y_pred)
            results.update(
                {
                    "roc_auc": roc_auc,
                    "roc_auc_weighted": roc_auc_weighted,
                    "avg_precision_score": avg_precision_score,
                    "avg_precision_score_weighted": avg_precision_score_weighted,
                    "log_loss": log_loss_,
                    "accuracy": accuracy,
                }
            )
        # TODO: regression

        self.best_loss = min(self.best_loss, results["loss"])
        self.hyperopt_eval_num += 1
        self.random_state += 1  # change random state after a run
        results.update(
            {"hyperopt_eval_num": self.hyperopt_eval_num, "best_loss": self.best_loss}
        )

        if verbose:
            print(
                "[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}".format(
                    self.hyperopt_eval_num,
                    self.hyperopt_evals,
                    fit_time,
                    self.metric,
                    results["loss"],
                    self.best_loss,
                )
            )
        return results

    def fit(
        self,
        params,
        X_train,
        y_train,
        Xy_val,
        sample_weight,
        n_estimators=None,
        seed=None,
    ):
        raise NotImplementedError("Method fit is not implemented.")

    def predict(self, bst, X_test):
        raise NotImplementedError("Method predict is not implemented.")

    def preprocess_params(self, params):
        raise NotImplementedError("Method preprocess_params is not implemented.")


class RFExperiment(Experiment):
    def __init__(
        self,
        learning_task,
        n_estimators=100,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            "rf",
            n_estimators,
            max_hyperopt_evals,
            None,  # no categorical_feature since RF uses one-hot
            0,  # no early-stopping for RF
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here TODO: check for other parameters?
        self.space = {
            "max_features": hp.choice("max_features", [None, "sqrt"]),
            "min_samples_split": hp.choice("min_samples_split", [2, 6, 10])
         }
        # hard-coded default params here
        self.default_params = {"max_features": "sqrt", "min_samples_split": 2}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "sklearn-RandomForest"

    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update(
            {"n_estimators": self.n_estimators,
             "random_state": self.random_state,
             "max_depth": None,
             "min_samples_leaf": int(params["min_samples_split"]/2)}
        )
        return params_

    def fit(
        self,
        params,
        X_train,
        y_train,
        Xy_val,
        sample_weight,
        n_estimators=None,
        seed=None,
    ):
        # no categorical_features, use one-hot encoding with RandomForestClassfier
        #  X_val, y_val not used since no early stopping
        if seed is not None:
            params.update({"random_state": seed})
        if n_estimators is not None:
            params.update({"n_estimators": n_estimators})
        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf, None

    def predict(self, bst, X_test):
        if self.learning_task == "classification":
            preds = bst.predict_proba(X_test)
        else:
            preds = bst.predict(X_test)
        return preds


class HGBExperiment(Experiment):
    def __init__(
        self,
        learning_task,
        n_estimators=100,
        max_hyperopt_evals=50,
        categorical_features=None,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            "hgb",
            n_estimators,
            max_hyperopt_evals,
            categorical_features,
            0,  # no early stopping round for HGB
            random_state,
            output_folder_path,
        )

        # hard-coded params search space here
        self.space = {
            "learning_rate": hp.loguniform("learning_rate", -4, 1),
            "max_leaf_nodes": hp.qloguniform("num_leaves", 0, 7, 1),
            "min_samples_leaf": hp.qloguniform("min_samples_leaf", 0, 6, 1),
            "l2_regularization": hp.choice(
                "l2_regularization",
                [0, hp.loguniform("l2_regularization_positive", -16, 2)],
            ),
        }
        # hard-coded default params here
        self.default_params = {
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0,
        }
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "sklearn-HistGradientBoosting"

    def preprocess_params(self, params):
        params_ = params.copy()
        if self.learning_task_ == "binary-classification":
            params_.update({"loss": "binary_crossentropy"})
        elif self.learning_task_ == "multiclass-classification":
            params_.update({"loss": "categorical_crossentropy"})
        elif self.learning_task_ == "regression":
            pass  # TODO

        params_["max_leaf_nodes"] = max(int(params_["max_leaf_nodes"]), 2)
        params_.update(
            {"max_iter": self.n_estimators, "random_state": self.random_state, "max_depth": None}
        )
        return params_

    def fit(
        self,
        params,
        X_train,
        y_train,
        Xy_val,
        sample_weight,
        n_estimators=None,
        seed=None,
    ):
        # Xy_val not used
        if seed is not None:
            params.update({"random_state": seed})
        if n_estimators is not None:
            params.update({"max_iter": n_estimators})
        clf = HistGradientBoostingClassifier(**params, early_stopping="auto")
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf, None

    def predict(self, bst, X_test):
        if self.learning_task == "classification":
            preds = bst.predict_proba(X_test)
        else:
            preds = bst.predict(X_test)
        return preds


class LGBExperiment(Experiment):
    def __init__(
        self,
        learning_task,
        n_estimators=100,
        max_hyperopt_evals=50,
        categorical_features=None,
        early_stopping_round=5,
        random_state=0,
        use_gpu=False,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            "lgb",
            n_estimators,
            max_hyperopt_evals,
            categorical_features,
            early_stopping_round,
            random_state,
            output_folder_path,
            use_gpu
        )

        # hard-coded params search space here
        self.space = {
            "learning_rate": hp.loguniform("learning_rate", -7, 0),
            "num_leaves": hp.qloguniform("num_leaves", 0, 7, 1),
            "feature_fraction": hp.uniform("feature_fraction", 0.5, 1),
            "bagging_fraction": hp.uniform("bagging_fraction", 0.5, 1),
            "min_data_in_leaf": hp.qloguniform("min_data_in_leaf", 0, 6, 1),
            "min_sum_hessian_in_leaf": hp.loguniform("min_sum_hessian_in_leaf", -16, 5),
            "lambda_l1": hp.choice(
                "lambda_l1", [0, hp.loguniform("lambda_l1_positive", -16, 2)]
            ),
            "lambda_l2": hp.choice(
                "lambda_l2", [0, hp.loguniform("lambda_l2_positive", -16, 2)]
            ),
        }
        # hard-coded default params here
        self.default_params = {
            "learning_rate": 0.1,
            "num_leaves": 127,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "min_data_in_leaf": 20,
            "min_sum_hessian_in_leaf": 1e-3,
            "lambda_l1": 0,
            "lambda_l2": 0,
        }
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "LightGBM"

    def preprocess_params(self, params):
        params_ = params.copy()
        if self.learning_task_ == "binary-classification":
            params_.update(
                {
                    "objective": "binary",
                    "metric": "binary_logloss",
                }
            )
        elif self.learning_task_ == "multiclass-classification":
            params_.update(
                {
                    "objective": "multiclass",
                    "metric": "multiclass",
                }
            )
        elif self.learning_task == "regression":
            params_.update(
                {
                    "objective": "mean_squared_error",
                    "metric": "l2",
                }
            )
        params_["num_leaves"] = max(int(params_["num_leaves"]), 2)
        params_["min_data_in_leaf"] = int(params_["min_data_in_leaf"])

        params_.update(
            {
                "n_estimators": self.n_estimators,
                "bagging_freq": 1,
                "random_state": self.random_state,
                "verbose": -1,
            }
        )
        return params_

    def fit(
        self,
        params,
        X_train,
        y_train,
        Xy_val,
        sample_weight,
        n_estimators=None,
        seed=None,
    ):
        if seed is not None:
            params.update({"random_state": seed})
        if n_estimators is not None:
            params.update({"n_estimators": n_estimators})

        bst = lgb.LGBMClassifier(**params, device_type='cpu' if not self.use_gpu else 'gpu')
        bst.fit(
            X_train,
            y=y_train,
            categorical_feature="auto",  # We do not use `self.categorical_features` actually,
            sample_weight=sample_weight,
            eval_set=[Xy_val] if Xy_val is not None else None,
            early_stopping_rounds=self.early_stopping_round
            if Xy_val is not None
            else None,
        )
        return bst, bst.best_iteration_

    def predict(self, bst, X_test):
        if self.learning_task == "classification":
            preds = bst.predict_proba(X_test)
        else:
            preds = bst.predict(X_test)
        return preds


class XGBExperiment(Experiment):
    def __init__(
        self,
        learning_task,
        n_estimators=100,
        max_hyperopt_evals=50,
        early_stopping_round=5,
        random_state=0,
        use_gpu=False,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            "xgb",
            n_estimators,
            max_hyperopt_evals,
            None,  # no categorical_feature since XGB uses one-hot
            early_stopping_round,
            random_state,
            output_folder_path,
            use_gpu
        )

        # hard-coded params search space here
        self.space = {
            "eta": hp.loguniform("eta", -7, 0),
            "max_depth": hp.quniform("max_depth", 2, 10, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
            "min_child_weight": hp.loguniform("min_child_weight", -16, 5),
            "alpha": hp.choice("alpha", [0, hp.loguniform("alpha_positive", -16, 2)]),
            "lambda": hp.choice(
                "lambda", [0, hp.loguniform("lambda_positive", -16, 2)]
            ),
            "gamma": hp.choice("gamma", [0, hp.loguniform("gamma_positive", -16, 2)]),
        }
        # hard-coded default params here
        self.default_params = {
            "eta": 0.3,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "colsample_bylevel": 1.0,
            "min_child_weight": 1,
            "alpha": 0,
            "lambda": 1,
            "gamma": 0,
        }
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "XGBoost"

    def preprocess_params(self, params):
        params_ = params.copy()
        if self.learning_task_ == "binary-classification":
            params_.update(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                }
            )
        elif self.learning_task_ == "multiclass-classification":
            params_.update(
                {
                    "objective": "multi:softmax",
                    "eval_metric": "mlogloss",
                }
            )
        elif self.learning_task_ == "regression":
            params_.update(
                {
                    "objective": "reg:linear",
                    "eval_metric": "rmse",
                }
            )
        params_["max_depth"] = int(params_["max_depth"])
        params_.update(
            {
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "verbosity": 1,
            }
        )
        return params_

    def fit(
        self,
        params,
        X_train,
        y_train,
        Xy_val,
        sample_weight,
        n_estimators=None,
        seed=None,
    ):
        # no categorical_features, use one-hot encoding with XGBoost
        if seed is not None:
            params.update({"seed": seed})
        if n_estimators is not None:
            params.update({"n_estimators": n_estimators})
        bst = xgb.XGBClassifier(**params, use_label_encoder=False, n_jobs=-1,
                                tree_method='hist' if not self.use_gpu else 'gpu_hist')
        bst.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[Xy_val] if Xy_val is not None else None,
            early_stopping_rounds=self.early_stopping_round
            if Xy_val is not None
            else None,
            verbose=True,
        )
        return bst, bst.best_iteration

    def predict(self, bst, X_test):
        if self.learning_task == "classification":
            preds = bst.predict_proba(X_test)
        else:
            preds = bst.predict(X_test)
        return preds


class CABExperiment(Experiment):
    def __init__(
        self,
        learning_task,
        n_estimators=100,
        max_hyperopt_evals=50,
        categorical_features=None,
        early_stopping_round=5,
        random_state=0,
        use_gpu=False,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            "cab",
            n_estimators,
            max_hyperopt_evals,
            categorical_features,
            early_stopping_round,
            random_state,
            output_folder_path,
            use_gpu,
        )

        # hard-coded params search space here
        # TODO: check these params, and params in catboost paper
        self.space = {
            "depth": hp.choice("depth", [6]),
            # 'ctr_border_count': hp.choice('ctr_border_count', [16]),
            "border_count": hp.choice("border_count", [128]),  # max_bin
            # ctr_description
            "simple_ctr": hp.choice("simple_ctr", [["Borders", "Counter"]]),
            "combinations_ctr": hp.choice("combinations_ctr", [["Borders", "Counter"]]),
            "learning_rate": hp.loguniform("learning_rate", -5, 0),
            "random_strength": hp.choice("random_strength", [1, 20]),
            "one_hot_max_size": hp.choice("one_hot_max_size", [0, 25]),
            "l2_leaf_reg": hp.loguniform("l2_leaf_reg", 0, np.log(10)),
            "bagging_temperature": hp.uniform("bagging_temperature", 0, 1),
            "used_ram_limit": hp.choice("used_ram_limit", [100000000000]),
        }

        # if learning_task == 'classification':
        #     self.space.update({
        #         'gradient_iterations': hp.choice('gradient_iterations', [1, 10])
        #     })

        # hard-coded default params here
        self.default_params = {
            "learning_rate": 0.03,
            "depth": 6,
            "fold_len_multiplier": 2,
            "rsm": 1.0,
            "border_count": 128,
            # 'ctr_border_count': 16,
            "l2_leaf_reg": 3,
            "leaf_estimation_method": "Newton",
            # 'gradient_iterations': 10,
            "simple_ctr": ["Borders", "Counter"],
            "combinations_ctr": ["Borders", "Counter"],
            "used_ram_limit": 100000000000,
        }
        self.default_params = self.preprocess_params(self.default_params)
        self.title = "CatBoost"

    def preprocess_params(self, params):
        params_ = params.copy()
        if self.learning_task_ == "binary-classification":
            params_.update({"loss_function": "Logloss"})
        elif self.learning_task_ == "multiclass-classification":
            params_.update({"loss_function": "MultiClass"})
        elif self.learning_task_ == "regression":
            params_.update({"loss_function": "RMSE"})
        params_.update(
            {
                "n_estimators": self.n_estimators,
                "logging_level": "Silent",
                "allow_writing_files": False,
                "thread_count": -1,
                "random_seed": self.random_state,
            }
        )
        return params_

    def fit(
        self,
        params,
        X_train,
        y_train,
        Xy_val,
        sample_weight,
        n_estimators=None,
        seed=None,
    ):
        if seed is not None:
            params.update({"random_seed": seed})
        if n_estimators is not None:
            params.update({"n_estimators": n_estimators})
        bst = CatBoostClassifier(**params, task_type="GPU" if self.use_gpu else "CPU")
        bst.fit(
            X_train,
            y=y_train,
            sample_weight=sample_weight,
            cat_features=self.categorical_features,
            eval_set=[Xy_val] if Xy_val is not None else None,
            early_stopping_rounds=self.early_stopping_round
            if Xy_val is not None
            else None,
        )
        return bst, bst.best_iteration_

    def predict(self, bst, X_test):
        if self.learning_task == "classification":
            preds = np.array(bst.predict_proba(X_test))
        else:
            preds = np.array(bst.predict(X_test))
        return preds


class LogRegExperiment(Experiment):
    def __init__(
        self,
        learning_task,
        max_hyperopt_evals=50,
        random_state=0,
        output_folder_path="./",
    ):
        Experiment.__init__(
            self,
            learning_task,
            "logreg",
            max_hyperopt_evals,
            1,  # n_estimators not useful in log reg
            None,  # no categorical_feature since log reg uses one-hot
            0,  # no early stopping
            random_state,
            output_folder_path,
        )
        # TODO but this subclass seems not useful
        self.title = "sklearn-LogisticRegression"

    def run(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        sample_weight,
        params=None,
        n_estimators=None,
        verbose=False,
    ):
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            n_jobs=-1,
            solver="saga",
            verbose=0,
            max_iter=100,
        )
        start_time = time.time()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        fit_time = time.time() - start_time
        y_scores = clf.predict_proba(X_val)
        evals_result = log_loss(y_val, y_scores)
        y_pred = np.argmax(y_scores, axis=1)
        results = {
            "loss": evals_result,
            "fit_time": fit_time,
            # "params": params.copy(),  # TODO
        }
        roc_auc = roc_auc_score(y_val, y_scores[:, 1])
        roc_auc_weighted = roc_auc
        avg_precision_score = average_precision_score(y_val, y_scores[:, 1])
        avg_precision_score_weighted = avg_precision_score
        log_loss_ = log_loss(y_val, y_scores)
        accuracy = accuracy_score(y_val, y_pred)
        results.update(
            {
                "roc_auc": roc_auc,
                "roc_auc_weighted": roc_auc_weighted,
                "avg_precision_score": avg_precision_score,
                "avg_precision_score_weighted": avg_precision_score_weighted,
                "log_loss": log_loss_,
                "accuracy": accuracy,
            }
        )
        return results

    def optimize_params(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        sample_weight,
        max_evals=None,
        verbose=True,
    ):
        clf = LogisticRegressionCV(
            Cs=self.hyperopt_evals,
            penalty="l2",
            n_jobs=-1,
            solver="lbfgs",  # saga
            verbose=1,
            max_iter=500,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)  # sample_weight=sample_weight
        return clf
