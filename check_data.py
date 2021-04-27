from wildwood.datasets._adult import load_adult

dataset = load_adult()

print(dataset)
print(dataset.df_raw)

X_train, X_test, y_train, y_test = dataset.extract(42)

# print(dataset.columns_)
from experiments.experiment import LGBExperiment

learning_task = 'classification'
n_estimators = 100

exp = LGBExperiment(learning_task, n_estimators=n_estimators, max_hyperopt_evals=2)
default_cv_result = exp.run_cv([((X_train, y_train), (X_test, y_test))])  # TODO: shoud be [((X_train, y_train), (X_val, y_val))]
tuned_cv_result = exp.optimize_params([((X_train, y_train), (X_test, y_test))], max_evals=2)
# TODO: add fitting time

