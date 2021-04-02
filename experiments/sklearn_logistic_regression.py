from time import time
import argparse

import datasets

from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Moons")
parser.add_argument('--normalize-intervals', action="store_true", default=False)
parser.add_argument('--one-hot-categoricals', action="store_true", default=False)
parser.add_argument('--dataset-path', type=str, default="data")
parser.add_argument('--dataset-subsample', type=int, default=100000)
parser.add_argument('--penalty', type=str, default='l2')
parser.add_argument('--dual', action="store_true", default=False)
parser.add_argument('--random-state', type=int, default=0)
parser.add_argument('--n-jobs', type=int, default=-1)
parser.add_argument('--solver', type=str, default='lbfgs')# use Saga ?
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--max-iter', type=int, default=100)


args = parser.parse_args()


print("Running sklearn Logistic Regression with training set {}".format(args.dataset))
print ("")
print("with arguments")
print(args)
print ("")

dataset = datasets.load_dataset(args, as_pandas=True)

if dataset.task != "classification":
    print("The loaded dataset is not for classification ... exiting")
    exit()
dataset.info()
#sample_weights = dataset.get_train_sample_weights()

print("Training Scikit Learn Logistic regression classifier ...")
### Using class_weights="balanced" instead of sample_weights to dodge LogisticRegression bug when n_jobs > 1 and dataset is too big
clf = LogisticRegression(penalty=args.penalty, dual = args.dual, n_jobs=args.n_jobs, solver=args.solver, verbose=args.verbose, max_iter=args.max_iter, class_weight="balanced")


tic = time()
clf.fit(dataset.data_train, dataset.target_train)#, sample_weight=sample_weights)
toc = time()

print(f"fitted in {toc - tic:.3f}s")

datasets.evaluate_classifier(clf, dataset.data_test, dataset.target_test, binary=dataset.binary)

