
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score, classification_report

from wildwood.dataset import load_car
from wildwood import ForestClassifier


dataset = load_car()
dataset.one_hot_encode = False
dataset.standardize = False
random_state = 42
X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
y_test_binary = LabelBinarizer().fit_transform(y_test)


clf = ForestClassifier(
    n_estimators=10,
    n_jobs=-1,
    multiclass="multinomial",
    aggregation=False,
    max_features=None,
    class_weight="balanced",
    categorical_features=dataset.categorical_features_,
    random_state=random_state,
)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))


clf = ForestClassifier(
    n_estimators=10,
    n_jobs=-1,
    multiclass="ovr",
    aggregation=False,
    max_features=None,
    class_weight="balanced",
    categorical_features=dataset.categorical_features_,
    random_state=random_state,
)

clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
