from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import numpy as np
import os
import time

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost, Pool


class Experiment(object):
    """
    inspired from
    https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/experiment.py
    """
    def __init__(self, learning_task='classification', bst_name=None, n_estimators=100, hyperopt_evals=50,
                 output_folder_path='./'):
        self.learning_task, self.bst_name = learning_task, bst_name

        self.n_estimators, self.best_loss = n_estimators, np.inf
        self.best_n_estimators = None
        self.hyperopt_evals, self.hyperopt_eval_num = hyperopt_evals, 0
        self.output_folder_path = os.path.join(output_folder_path, '')  # TODO: write output
        self.default_params, self.best_params = None, None

        # to specify definitions in particular experiments
        self.title = None
        self.space = None
        self.trials = None

        if self.learning_task == 'classification':
            self.metric = 'logloss'
        elif self.learning_task == 'regression':
            self.metric = 'rmse'
        else:
            raise ValueError('Task type must be "classification" or "regression"')

    def optimize_params(self, X_train, y_train, X_val, y_val, sample_weight, max_evals=None, cat_cols=None, verbose=True):
        max_evals = max_evals or self.hyperopt_evals
        self.trials = Trials()
        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        _ = fmin(fn=lambda params: self.run(X_train, y_train, X_val, y_val, sample_weight, params, cat_cols, verbose=verbose),
                 space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=self.trials)

        self.best_params = self.trials.best_trial['result']['params']
        return self.trials.best_trial['result']

    def run(self, X_train, y_train, X_val, y_val, sample_weight, params=None, n_estimators=None, cat_cols=None, verbose=False):
        params = params or self.default_params
        n_estimators = n_estimators or self.n_estimators
        params = self.preprocess_params(params)
        evals_results, start_time = [], time.time()

        dtrain = self.convert_to_dataset(X_train.astype(float), y_train, cat_cols, sample_weight)
        dval = self.convert_to_dataset(X_val.astype(float), y_val, cat_cols, sample_weight=None)
        _, evals_results = self.fit(params, dtrain, dval, n_estimators)
        evals_result = np.mean(evals_results)
        eval_time = time.time() - start_time

        cv_result = {'loss': evals_result,
                     'eval_time': eval_time,
                     'status': STATUS_FAIL if np.isnan(evals_result) else STATUS_OK,
                     'params': params.copy()}
        self.best_loss = min(self.best_loss, cv_result['loss'])
        self.hyperopt_eval_num += 1
        cv_result.update({'hyperopt_eval_num': self.hyperopt_eval_num, 'best_loss': self.best_loss})

        if verbose:
            print('[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}'.format(
                self.hyperopt_eval_num, self.hyperopt_evals, eval_time,
                self.metric, cv_result['loss'], self.best_loss))
        return cv_result

    def fit(self, params, dtrain, dtest, n_estimators):
        raise NotImplementedError('Method fit is not implemented.')

    def predict(self, bst, X_test):
        raise NotImplementedError('Method predict is not implemented.')

    def preprocess_params(self, params):
        raise NotImplementedError('Method preprocess_params is not implemented.')

    def convert_to_dataset(self, data, label, cat_cols, sample_weight):
        raise NotImplementedError('Method convert_to_dataset is not implemented.')


class LGBExperiment(Experiment):

    def __init__(self, learning_task, n_estimators=100, max_hyperopt_evals=50,
                 output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'lgb', n_estimators, max_hyperopt_evals,
                            output_folder_path)

        # hard-coded params search space here
        self.space = {
            'learning_rate': hp.loguniform('learning_rate', -7, 0),
            'num_leaves': hp.qloguniform('num_leaves', 0, 7, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        }
        # hard-coded default params here
        self.default_params = {'learning_rate': 0.1, 'num_leaves': 127, 'feature_fraction': 1.0,
                               'bagging_fraction': 1.0, 'min_data_in_leaf': 100, 'min_sum_hessian_in_leaf': 10,
                               'lambda_l1': 0, 'lambda_l2': 0}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = 'LightGBM'

    def preprocess_params(self, params):
        params_ = params.copy()
        if self.learning_task == 'classification':
            params_.update({'objective': 'binary', 'metric': 'binary_logloss',
                            'bagging_freq': 1, 'verbose': -1})
        elif self.learning_task == "regression":
            params_.update({'objective': 'mean_squared_error', 'metric': 'l2',
                            'bagging_freq': 1, 'verbose': -1})
        params_['num_leaves'] = max(int(params_['num_leaves']), 2)
        params_['min_data_in_leaf'] = int(params_['min_data_in_leaf'])
        return params_

    def convert_to_dataset(self, data, label, cat_cols, sample_weight=None):
        return lgb.Dataset(data, label, weight=sample_weight, categorical_feature=cat_cols)

    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params.update({
            'data_random_seed': 1 + seed,
            'feature_fraction_seed': 2 + seed,
            'bagging_seed': 3 + seed,
            'drop_seed': 4 + seed,
        })
        evals_result = {}

        bst = lgb.train(params, dtrain, valid_sets=[dtest], valid_names=['val'], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)

        results = np.power(evals_result['val']['l2'], 0.5) if self.learning_task == 'regression' \
            else evals_result['val']['binary_logloss']
        return bst, results

    def predict(self, bst, X_test):
        preds = bst.predict(X_test)
        return preds


class XGBExperiment(Experiment):

    def __init__(self, learning_task, n_estimators=5000, max_hyperopt_evals=50, output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'xgb', n_estimators, max_hyperopt_evals,
                            output_folder_path)

        # hard-coded params search space here
        self.space = {
            'eta': hp.loguniform('eta', -7, 0),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)])
        }
        # hard-coded default params here
        self.default_params = {'eta': 0.3, 'max_depth': 6, 'subsample': 1.0,
                               'colsample_bytree': 1.0, 'colsample_bylevel': 1.0,
                               'min_child_weight': 1, 'alpha': 0, 'lambda': 1, 'gamma': 0}
        self.default_params = self.preprocess_params(self.default_params)
        self.title = 'XGBoost'

    def preprocess_params(self, params):
        if self.learning_task == "classification":
            params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 1})
        elif self.learning_task == "regression":
            params.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'verbosity': 1})
        params['max_depth'] = int(params['max_depth'])
        return params

    def convert_to_dataset(self, data, label, cat_cols=None, sample_weight=None):
        # In this way, categorical features are used as their numerical values
        # TODO but this is not the currect way, should use one-hot encoding indeed
        return xgb.DMatrix(data, label, weight=sample_weight)

    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params.update({"seed": seed})
        evals_result = {}
        bst = xgb.train(params, dtrain, evals=[(dtest, 'val')], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)

        results = evals_result['val']['rmse'] if self.learning_task == 'regression' \
            else evals_result['val']['logloss']
        return bst, results

    def predict(self, bst, X_test):
        preds = bst.predict(X_test)
        return preds


class CABExperiment(Experiment):

    def __init__(self, learning_task, n_estimators=5000, max_hyperopt_evals=50,
                 counters_sort_col=None, holdout_size=0,
                 train_path=None, test_path=None, cd_path=None, output_folder_path='./'):
        assert holdout_size == 0, 'For Catboost holdout_size must be equal to 0'
        Experiment.__init__(self, learning_task, 'cab', n_estimators, max_hyperopt_evals,
                            output_folder_path)

        # hard-coded params search space here
        self.space = {
            'depth': hp.choice('depth', [6]),
            # 'ctr_border_count': hp.choice('ctr_border_count', [16]),
            'border_count': hp.choice('border_count', [128]),  # max_bin
            # ctr_description
            'combinations_ctr': hp.choice('combinations_ctr', [['Borders', 'Counter']]),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'random_strength': hp.choice('random_strength', [1, 20]),
            'one_hot_max_size': hp.choice('one_hot_max_size', [0, 25]),
            'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
            'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
            'used_ram_limit': hp.choice('used_ram_limit', [100000000000]),
        }

        # if learning_task == 'classification':
        #     self.space.update({
        #         'gradient_iterations': hp.choice('gradient_iterations', [1, 10])
        #     })

        # hard-coded default params here
        self.default_params = {
            'learning_rate': 0.03,
            'depth': 6,
            'fold_len_multiplier': 2,
            'rsm': 1.0,
            'border_count': 128,
            # 'ctr_border_count': 16,
            'l2_leaf_reg': 3,
            'leaf_estimation_method': 'Newton',
            # 'gradient_iterations': 10,
            'combinations_ctr': ['Borders', 'Counter'],
            'used_ram_limit': 100000000000,
        }
        self.default_params = self.preprocess_params(self.default_params)
        self.title = 'CatBoost'

    def convert_to_dataset(self, data, label, cat_cols, sample_weight=None):
        return Pool(data, label, weight=sample_weight, cat_features=cat_cols)

    def preprocess_params(self, params):
        if self.learning_task == 'classification':
            params.update({'loss_function': 'Logloss', 'verbose': False, 'thread_count': 16, 'random_seed': 0})
        elif self.learning_task == 'regression':
            params.update({'loss_function': 'RMSE', 'verbose': False, 'thread_count': 16, 'random_seed': 0})
        return params

    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params.update({"iterations": n_estimators})
        params.update({"random_seed": seed})
        bst = CatBoost(params)
        bst.fit(dtrain, eval_set=dtest)
        # with open("test_error.tsv", "r") as f:
        #     results = np.array(map(lambda x: float(x.strip().split()[-1]), f.readlines()[1:]))
        results = bst.get_best_score()['validation']['Logloss']

        return bst, results

    def predict(self, bst, X_test):
        preds = np.array(bst.predict(X_test))
        if self.learning_task == 'classification':
            preds = np.power(1 + np.exp(-preds), -1)
        return preds


