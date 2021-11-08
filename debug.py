# @pytest.mark.parametrize("n_estimators", [2])
# @pytest.mark.parametrize("aggregation", (True,))
# @pytest.mark.parametrize("class_weight", (None, "balanced"))
# @pytest.mark.parametrize("dirichlet", (1e-7,))
# @pytest.mark.parametrize("n_jobs", (-1,))
# @pytest.mark.parametrize("max_features", ("auto",))
# @pytest.mark.parametrize("random_state", (42,))
# @pytest.mark.parametrize("step", (1.0,))
# @pytest.mark.parametrize("multiclass", ("multinomial", "ovr"))
# @pytest.mark.parametrize("cat_split_strategy", ("binary",))
# @pytest.mark.parametrize(
#     "dataset_name, one_hot_encode, use_categoricals",
#     [
#         ("adult", False, False),
#         ("adult", False, True),
#         ("adult", True, False),
#         ("iris", False, False),
#     ],
# )

import pickle as pkl
from sklearn.model_selection import train_test_split
from wildwood import ForestClassifier
from wildwood.datasets import load_adult


n_estimators = 2
aggregation = True
class_weight = "balanced"
dirichlet = 1e-4
n_jobs = -1
max_features = "auto"
random_state = 42
step = 1.0
multiclass = "multinomial"
cat_split_strategy = "binary"


X, y = load_adult(raw=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


clf1 = ForestClassifier(
    n_estimators=n_estimators,
    n_jobs=n_jobs,
    multiclass=multiclass,
    cat_split_strategy=cat_split_strategy,
    aggregation=aggregation,
    max_features=max_features,
    class_weight=class_weight,
    random_state=random_state,
    dirichlet=dirichlet,
    step=step,
)

clf1.fit(X_train, y_train)


filename = "forest_classifier_on_iris.pkl"
with open(filename, "wb") as f:
    pkl.dump(clf1, f)

with open(filename, "rb") as f:
    clf2 = pkl.load(f)


# os.remove(filename)
#
# assert_forests_equal(clf1, clf2, is_classifier=True)
#
# y_pred1 = clf1.predict_proba(X_test)
# y_pred2 = clf2.predict_proba(X_test)
# assert np.all(y_pred1 == y_pred2)
#
# y_pred1 = clf1.predict(X_test)
# y_pred2 = clf2.predict(X_test)
# assert np.all(y_pred1 == y_pred2)
#
# apply1 = clf1.apply(X_test)
# apply2 = clf2.apply(X_test)
# assert np.all(apply1 == apply2)