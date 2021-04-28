# from wildwood.datasets._adult import load_adult
from wildwood.dataset import load_adult
from experiments.experiment import LGBExperiment

dataset = load_adult()

print(dataset)

X_train, X_test, y_train, y_test = dataset.extract(42)



learning_task = 'classification'
n_estimators = 100

exp = LGBExperiment(learning_task, n_estimators=n_estimators, max_hyperopt_evals=2)
print("run default params")
default_cv_result = exp.run(X_train, y_train, X_test, y_test, verbose=True)
print("run train-val hyperopt")
tuned_cv_result = exp.optimize_params(X_train, y_train, X_test, y_test, max_evals=2, verbose=True)
# TODO: add fitting time