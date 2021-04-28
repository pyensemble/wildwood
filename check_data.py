# from wildwood.datasets._adult import load_adult
from wildwood.dataset import load_breastcancer
from experiments.experiment import LGBExperiment, XGBExperiment, CABExperiment

dataset = load_breastcancer()

print(dataset)

X_train, X_test, y_train, y_test = dataset.extract(42)

learning_task = 'classification'
n_estimators = 100

# exp = LGBExperiment(learning_task, n_estimators=n_estimators, max_hyperopt_evals=2)
# exp = XGBExperiment(learning_task, n_estimators=n_estimators, max_hyperopt_evals=2)
exp = CABExperiment(learning_task, n_estimators=n_estimators, max_hyperopt_evals=2)
print("Run default params exp")
default_cv_result = exp.run(X_train, y_train, X_test, y_test, sample_weight=None, cat_cols=dataset.categorical_features_, verbose=True)
print("Run train-val hyperopt exp")
tuned_cv_result = exp.optimize_params(X_train, y_train, X_test, y_test, max_evals=2,
                                      sample_weight=None, cat_cols=dataset.categorical_features_, verbose=True)
# TODO: add fitting time