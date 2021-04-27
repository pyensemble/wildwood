from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import numpy as np
import os
import time

import lightgbm as lgb


class Experiment(object):
    """
    inspired from
    https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/experiment.py
    """
    def __init__(self, learning_task='classification', bst_name=None, n_estimators=5000, hyperopt_evals=50,
                 compute_counters=True, counters_sort_col=None, holdout_size=0,
                 output_folder_path='./'):
        self.learning_task, self.bst_name = learning_task, bst_name
        self.compute_counters = compute_counters
        self.holdout_size = holdout_size
        self.counters_sort_col = counters_sort_col
        self.n_estimators, self.best_loss = n_estimators, np.inf
        self.best_n_estimators = None
        self.hyperopt_evals, self.hyperopt_eval_num = hyperopt_evals, 0
        self.output_folder_path = os.path.join(output_folder_path, '')
        self.default_params, self.best_params = None, None
        # to specify definitions in particular experiments
        self.title = None
        self.space = None

        if self.learning_task == 'classification':
            self.metric = 'logloss'
        elif self.learning_task == 'regression':
            self.metric = 'rmse'
        else:
            raise ValueError('Task type must be "classification" or "regression"')

    def optimize_params(self, cv_pairs, max_evals=None, verbose=True):
        max_evals = max_evals or self.hyperopt_evals
        self.trials = Trials()
        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        _ = fmin(fn=lambda params: self.run_cv(cv_pairs, params, verbose=verbose),
                 space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=self.trials)

        self.best_params = self.trials.best_trial['result']['params']
        self.best_n_estimators = self.trials.best_trial['result']['best_n_estimators']
        return self.trials.best_trial['result']

    def run_cv(self, cv_pairs, params=None, n_estimators=None, verbose=False, cat_cols=None):
        params = params or self.default_params
        n_estimators = n_estimators or self.n_estimators
        params = self.preprocess_params(params)
        evals_results, start_time = [], time.time()
        for (X_train, y_train), (X_test, y_test) in cv_pairs:
            dtrain = self.convert_to_dataset(X_train.astype(float), y_train, cat_cols)
            dtest = self.convert_to_dataset(X_test.astype(float), y_test, cat_cols)
            _, evals_result = self.fit(params, dtrain, dtest, n_estimators)
            evals_results.append(evals_result)
        mean_evals_results = np.mean(evals_results, axis=0)
        best_n_estimators = np.argmin(mean_evals_results) + 1
        eval_time = time.time() - start_time

        cv_result = {'loss': mean_evals_results[best_n_estimators - 1],
                     'best_n_estimators': best_n_estimators,
                     'eval_time': eval_time,
                     'status': STATUS_FAIL if np.isnan(mean_evals_results[best_n_estimators - 1]) else STATUS_OK,
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

    def convert_to_dataset(self, data, label, cat_cols=None):
        raise NotImplementedError('Method convert_to_dataset is not implemented.')


class LGBExperiment(Experiment):

    def __init__(self, learning_task, n_estimators=5000, max_hyperopt_evals=50,
                 counters_sort_col=None, holdout_size=0,
                 output_folder_path='./'):
        Experiment.__init__(self, learning_task, 'lgb', n_estimators, max_hyperopt_evals,
                            True, counters_sort_col, holdout_size,
                            output_folder_path)

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

    def convert_to_dataset(self, data, label, cat_cols=None):
        return lgb.Dataset(data, label)

    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params.update({
            'data_random_seed': 1 + seed,
            'feature_fraction_seed': 2 + seed,
            'bagging_seed': 3 + seed,
            'drop_seed': 4 + seed,
        })
        evals_result = {}
        bst = lgb.train(params, dtrain, valid_sets=[dtest], valid_names=['test'], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)

        results = np.power(evals_result['test']['l2'], 0.5) if self.learning_task == 'regression' \
            else evals_result['test']['binary_logloss']
        return bst, results

    def predict(self, bst, X_test):
        preds = bst.predict(X_test)
        return preds
