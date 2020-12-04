
from sklearn.datasets import make_circles

n_samples = 150
random_state = 42

X, y = make_circles(n_samples=n_samples, noise=0.2, random_state=random_state)

