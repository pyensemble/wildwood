from itertools import product

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from wildwood.dataset import load_bank, load_car, load_adult, load_boston
from wildwood import ForestRegressor, ForestClassifier


dataset = load_car()
dataset.one_hot_encode = False
dataset.test_size = 1.0 / 5
random_state = 42
X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)

n_estimators = 1
aggregation = False
class_weight = "balanced"
# class_weight = None
n_jobs = -1
max_features = None
random_state = 42
dirichlet = 0.0
categorical_features = dataset.categorical_features_

multiclass = "multinomial"
clf = ForestClassifier(
    n_estimators=n_estimators,
    n_jobs=n_jobs,
    multiclass=multiclass,
    aggregation=aggregation,
    max_features=max_features,
    class_weight=class_weight,
    categorical_features=categorical_features,
    random_state=random_state,
    dirichlet=dirichlet,
)
clf.fit(X_train, y_train)
y_scores_train = clf.predict_proba(X_train)
y_pred_train = clf.predict(X_train)
y_scores_test = clf.predict_proba(X_test)
y_pred_test = clf.predict(X_test)
print("LL(train):", log_loss(y_train, y_scores_train))
print("LL(test):", log_loss(y_test, y_scores_test))
# print(classification_report(y_train, y_pred_train))
# print(classification_report(y_test, y_pred_test))

multiclass = "ovr"
clf = ForestClassifier(
    n_estimators=n_estimators,
    n_jobs=n_jobs,
    multiclass=multiclass,
    aggregation=aggregation,
    max_features=max_features,
    class_weight=class_weight,
    categorical_features=categorical_features,
    random_state=random_state,
    dirichlet=dirichlet,
)
clf.fit(X_train, y_train)
y_scores_train = clf.predict_proba(X_train)
y_pred_train = clf.predict(X_train)
y_scores_test = clf.predict_proba(X_test)
y_pred_test = clf.predict(X_test)
print("LL(train):", log_loss(y_train, y_scores_train))
print("LL(test):", log_loss(y_test, y_scores_test))

# # lloss1 = log_loss(y_train, y_scores)
# print(classification_report(y_train, y_pred_train))
# print(classification_report(y_test, y_pred_test))
