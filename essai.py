import numpy as np

from numpy.random import randint
from joblib import Parallel
from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.utils import check_random_state
from numba import njit, _helperlib

n = 10
n_jobs = 4
random_state = 42

out = np.empty((n_jobs, n))


@njit
def f(out, n, random_state, idx):
    # np.random.seed(random_state)
    np.random.seed(random_state)
    for i in range(n):
        out[idx, i] = randint(100)


def fit(random_state):
    np.random.seed(random_state)
    random_states = randint(np.iinfo(np.intp).max, size=n_jobs)
    trees = Parallel(n_jobs=16, **_joblib_parallel_args(prefer="threads"),)(
        delayed(f)(out, n, random_state, idx)
        for idx, random_state in zip(range(n_jobs), random_states)
    )


fit(random_state)
print(out)

fit(random_state)
print(out)

fit(random_state)
print(out)


rng = check_random_state(0)

print(rng.randint(np.iinfo(np.intp).max, size=10))