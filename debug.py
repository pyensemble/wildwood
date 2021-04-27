from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score, classification_report, log_loss,\
    roc_auc_score
from wildwood.dataset import load_bank, load_car, load_adult
from wildwood import ForestClassifier


dataset = load_adult()

# dataset.test_size = 1.0 / 5
# dataset.one_hot_encode = False
# dataset.standardize = False


n_estimators = 10
aggregation = False
class_weight = "balanced"
n_jobs = -1
max_features = None
random_state = 42
dirichlet = 0.0
step = 1.0


def run(multiclass, categorical_features, one_hot_encode):
    dataset.one_hot_encode = one_hot_encode
    X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)

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
        step=step
    )
    clf.fit(X_train, y_train)
    y_scores_train = clf.predict_proba(X_train)
    y_scores_test = clf.predict_proba(X_test)
    print("-" * 64)
    print(
        "multiclass:",
        multiclass,
        "categorical_features:",
        categorical_features,
        "one_hot_encode:",
        one_hot_encode,
    )
    # print(classification_report(y_train, clf.predict(X_train)))
    # print(classification_report(y_test, clf.predict(X_test)))
    print(
        "LL(train):",
        log_loss(y_train, y_scores_train),
        "LL(test):",
        log_loss(y_test, y_scores_test),
        # "AUC(train):",
        # roc_auc_score(y_train, y_scores_train[:, 1]),
        # "AUC(test):",
        # roc_auc_score(y_test, y_scores_test[:, 1]),
    )


multiclass = "multinomial"
categorical_features = None
one_hot_encode = True
run(multiclass, categorical_features, one_hot_encode)

multiclass = "multinomial"
categorical_features = None
one_hot_encode = False
run(multiclass, categorical_features, one_hot_encode)

multiclass = "multinomial"
categorical_features = dataset.categorical_features_
one_hot_encode = False
run(multiclass, categorical_features, one_hot_encode)
#
# multiclass = "ovr"
# categorical_features = None
# one_hot_encode = True
# run(multiclass, categorical_features, one_hot_encode)
#
# multiclass = "ovr"
# categorical_features = None
# one_hot_encode = False
# run(multiclass, categorical_features, one_hot_encode)
#
# multiclass = "ovr"
# categorical_features = dataset.categorical_features_
# one_hot_encode = False
# run(multiclass, categorical_features, one_hot_encode)
