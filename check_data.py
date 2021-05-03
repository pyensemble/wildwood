import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
)

# from wildwood.datasets._adult import load_adult
from wildwood.dataset import load_adult, load_bank

from experiments.experiment import (
    LogRegExperiment,
    RFExperiment,
    HGBExperiment,
    LGBExperiment,
    XGBExperiment,
    CABExperiment,
)


def get_train_sample_weights(labels, n_classes):
    # TODO: maybe put this function in _utils.py?
    def get_class_proportions(data):
        return np.bincount(data.astype(int)) / len(data)

    return (1 / (n_classes * get_class_proportions(labels)))[labels]


data_extraction = {
    "LogisticRegression": {
        "one_hot_encode": True,
        "standardize": False,
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

"""
It starts here
"""

dataset = load_bank()  # TODO: make this as argument
clf_name = "LGBMClassifier"
# "LGBMClassifier", "XGBClassifier", "CatBoostClassifier",
#  "RandomForestClassifier", "HistGradientBoostingClassifier", "LogisticRegression"
learning_task = "binary-classification"
n_estimators = 100
max_hyperopt_eval = 10
random_state_seed = 42
# TODO: check it is able to use all processes on GPU machine

random_states = {
    "data_extract_random_state": random_state_seed,
    "train_val_split_random_state": 1 + random_state_seed,
    "expe_random_state": 2 + random_state_seed,
}

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


experiment_setting = {
    "LogisticRegression": LogRegExperiment(
        learning_task,
        max_hyperopt_evals=max_hyperopt_eval,
        random_state=random_states["expe_random_state"],
    ),
    "RandomForestClassifier": RFExperiment(
        learning_task,
        n_estimators=n_estimators,
        max_hyperopt_evals=max_hyperopt_eval,
        random_state=random_states["expe_random_state"],
    ),
    "HistGradientBoostingClassifier": HGBExperiment(
        learning_task,
        n_estimators=n_estimators,
        max_hyperopt_evals=max_hyperopt_eval,
        categorical_features=dataset.categorical_columns,
        random_state=random_states["expe_random_state"],
    ),
    "XGBClassifier": XGBExperiment(
        learning_task,
        n_estimators=n_estimators,
        max_hyperopt_evals=max_hyperopt_eval,
        random_state=random_states["expe_random_state"],
    ),
    "LGBMClassifier": LGBExperiment(
        learning_task,
        n_estimators=n_estimators,
        max_hyperopt_evals=max_hyperopt_eval,
        categorical_features=dataset.categorical_columns,
        random_state=random_states["expe_random_state"],
    ),
    "CatBoostClassifier": CABExperiment(
        learning_task,
        n_estimators=n_estimators,
        max_hyperopt_evals=max_hyperopt_eval,
        categorical_features=dataset.categorical_columns,
        random_state=random_states["expe_random_state"],
    ),
}

exp = experiment_setting[clf_name]
sample_weights = get_train_sample_weights(y_tr, dataset.n_classes_)

if clf_name == "LogisticRegression":
    print("Run default params exp...")
    default_params_result = exp.run(
        X_tr, y_tr, X_val, y_val, sample_weight=sample_weights, verbose=True
    )

    print("Run train-val hyperopt exp...")
    tuned_clf = exp.optimize_params(
        X_train,
        y_train,
        X_val,  # not used
        y_val,  # not used
        max_evals=max_hyperopt_eval,
        sample_weight=get_train_sample_weights(y_train, dataset.n_classes_),
        verbose=True,
    )
    y_scores = tuned_clf.predict_proba(X_test)
    logloss = log_loss(y_test, y_scores)
    y_pred = np.argmax(y_scores, axis=1)
    print("logloss on test=", logloss)
    print("Done.")
else:
    print("Run default params exp...")
    default_params_result = exp.run(
        X_tr, y_tr, X_val, y_val, sample_weight=sample_weights, verbose=True
    )

    print("Run train-val hyperopt exp...")
    tuned_cv_result = exp.optimize_params(
        X_tr,
        y_tr,
        X_val,
        y_val,
        max_evals=max_hyperopt_eval,
        sample_weight=sample_weights,
        verbose=True,
    )

    print("Run fitting with tuned params...")
    model = exp.fit(
        tuned_cv_result["params"],
        X_train,
        y_train,
        sample_weight=get_train_sample_weights(y_train, dataset.n_classes_),
    )

    y_scores = exp.predict(model, X_test)
    logloss = log_loss(y_test, y_scores)
    y_pred = np.argmax(y_scores, axis=1)
    print("logloss on test=", logloss)
    print("Done.")

# TODO: add other metrics on test data, put them into dataframe
# TODO: add fitting time
# TODO: exhaustive search for the number of trees in the interval [1, 5000] (??)
