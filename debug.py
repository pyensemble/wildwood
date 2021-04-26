
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score, classification_report

from wildwood.dataset import load_car, load_bank
from wildwood import ForestClassifier


dataset = load_car()
dataset.one_hot_encode = False
dataset.standardize = False
random_state = 42
X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
y_test_binary = LabelBinarizer().fit_transform(y_test)


n_estimators = 1

clf = ForestClassifier(
    n_estimators=n_estimators,
    n_jobs=1,
    multiclass="multinomial",
    aggregation=False,
    max_features=None,
    class_weight="balanced",
    random_state=random_state,
)

clf.fit(X_train, y_train)

print(clf)
print(classification_report(y_train, clf.predict(X_train)))
print(classification_report(y_test, clf.predict(X_test)))


clf = ForestClassifier(
    n_estimators=10,
    n_jobs=1,
    multiclass="ovr",
    aggregation=False,
    max_features=None,
    class_weight="balanced",
    random_state=random_state,
)
clf.fit(X_train, y_train)

print(clf)
print(classification_report(y_train, clf.predict(X_train)))
print(classification_report(y_test, clf.predict(X_test)))


clf = ForestClassifier(
    n_estimators=10,
    n_jobs=-1,
    # multiclass="multinomial",
    aggregation=False,
    max_features=None,
    class_weight="balanced",
    # categorical_features=dataset.categorical_features_,
    random_state=random_state,
    dirichlet=0.0
)
clf.fit(X_train, y_train)
y_scores_train = clf.predict_proba(X_train)
y_scores_test = clf.predict_proba(X_test)

avg_prec_train = average_precision_score(y_train, y_scores_train[:, 1])
avg_prec_test = average_precision_score(y_test, y_scores_test[:, 1])

print(clf)
print("avg_prec_train:", avg_prec_train)
print("avg_prec_test:", avg_prec_test)

# print(classification_report(y_train, clf.predict(X_train)))
# print(classification_report(y_test, clf.predict(X_test)))


clf = ForestClassifier(
    n_estimators=10,
    n_jobs=-1,
    # multiclass="ovr",
    aggregation=False,
    max_features=None,
    class_weight="balanced",
    categorical_features=dataset.categorical_features_,
    random_state=random_state,
    dirichlet=0.0
)

clf.fit(X_train, y_train)
y_scores_train = clf.predict_proba(X_train)
y_scores_test = clf.predict_proba(X_test)


avg_prec_train = average_precision_score(y_train, y_scores_train[:, 1])
avg_prec_test = average_precision_score(y_test, y_scores_test[:, 1])

print(clf)
print("avg_prec_train:", avg_prec_train)
print("avg_prec_test:", avg_prec_test)



# print(clf)
# print(classification_report(y_train, clf.predict(X_train)))
# print(classification_report(y_test, clf.predict(X_test)))
