from wildwood.datasets import (
    load_adult,
    load_kddcup99,
    load_covtype,
    load_epsilon_catboost,
    load_kick,
    load_amazon,
    load_cardio,
    load_higgs,
)

from time import time
import logging
from sklearn.datasets import make_circles
from wildwood import ForestClassifier

logging.info("JIT compiling...")
tic = time()
X, y = make_circles(n_samples=10, noise=0.2, factor=0.5, random_state=1)
clf = ForestClassifier()
clf.fit(X, y)
y_scores = clf.predict_proba(X)
toc = time()
logging.info("Spent {time} compiling.".format(time=toc - tic))

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, classification_report


random_state = 42
# X, y = load_covtype(raw=True)

X, y = load_adult(raw=True)

dataset = "adult"

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

n_estimators = 100
n_jobs = -1
class_weight = "balanced"
categorical_features = [True] * X_train.shape[1]

clf = ForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state,
    class_weight=class_weight,
    n_jobs=n_jobs,
    categorical_features=categorical_features,
    verbose=False,
)
tic = time()
clf.fit(X_train, y_train)
toc = time()
fit_time = toc - tic
logging.info(f"fit took {fit_time} on {dataset}.")

tic = time()
y_scores = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)
toc = time()
fit_time = toc - tic
logging.info(f"predict_proba took {fit_time} on {dataset}.")

log_loss_adult = log_loss(y_test, y_scores)
logging.info(f"log-loss on {dataset} is {log_loss_adult}.")

print(clf.is_categorical_)
