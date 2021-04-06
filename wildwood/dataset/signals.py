# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
"""
This modules provide simple 1D noisy signals to check WildWood's forest for
regression. These examples of signals were originally made by David L. Donoho.
"""
import numpy as np
from numpy import sin, sign, pi, abs, sqrt
from numba import vectorize
from numba import float32, float64


__all__ = ["get_signal", "make_regression"]


@vectorize([float32(float32), float64(float64)], nopython=True)
def _heavisine(x):
    """Computes the "heavisine" signal.

    Parameters
    ----------
    x : ndarray
        Inputs values of shape (n_samples,) with dtype float32 or float64

    Returns
    -------
    output : ndarray
        The value of the signal at given inputs with shape (n_samples,)

    Notes
    -----
    Inputs are supposed to belong to [0, 1]
    """
    return 4 * sin(4 * pi * x) - sign(x - 0.3) - sign(0.72 - x)


@vectorize([float32(float32), float64(float64)], nopython=True)
def _bumps(x):
    """Computes the "bumps" signal.

    Parameters
    ----------
    x : ndarray
        Inputs values of shape (n_samples,) with dtype float32 or float64

    Returns
    -------
    output : ndarray
        The value of the signal at given inputs with shape (n_samples,)

    Notes
    -----
    Inputs are supposed to belong to [0, 1]
    """
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wth = np.array(
        [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005],
    )
    y = 0
    for j in range(pos.shape[0]):
        y += hgt[j] / ((1 + (abs(x - pos[j]) / wth[j])) ** 4)
    return y


@vectorize([float32(float32), float64(float64)], nopython=True)
def _blocks(x):
    """Computes the "blocks" signal.

    Parameters
    ----------
    x : ndarray
        Inputs values of shape (n_samples,) with dtype float32 or float64

    Returns
    -------
    output : ndarray
        The value of the signal at given inputs with shape (n_samples,)

    Notes
    -----
    Inputs are supposed to belong to [0, 1]
    """
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    y = 2.0
    for j in range(pos.shape[0]):
        y += (1 + sign(x - pos[j])) * (hgt[j] / 2)
    return y


@vectorize([float32(float32), float64(float64)], nopython=True)
def _doppler(x):
    """Computes the "doppler" signal.

    Parameters
    ----------
    x : ndarray
        Inputs values of shape (n_samples,) with dtype float32 or float64

    Returns
    -------
    output : ndarray
        The value of the signal at given inputs with shape (n_samples,)

    Notes
    -----
    Inputs are supposed to belong to [0, 1]
    """
    return sqrt(x * (1 - x)) * sin((2 * pi * 1.05) / (x + 0.05)) + 0.5


def get_signal(x, signal="heavisine"):
    """Computes a signal at the given inputs.

    Parameters
    ----------
    x : ndarray
        Inputs values of shape (n_samples,) with dtype float32 or float64

    signal : {"heavisine", "bumps", "blocks", "doppler"}, default="heavisine"
        Type of signal

    Returns
    -------
    output : ndarray
        The value of the signal at given inputs with shape (n_samples,)

    Notes
    -----
    Inputs are supposed to belong to [0, 1]
    """
    if signal == "heavisine":
        y = _heavisine(x)
    elif signal == "bumps":
        y = _bumps(x)
    elif signal == "blocks":
        y = _blocks(x)
    elif signal == "doppler":
        y = _doppler(x)
    else:
        y = _heavisine(x)
    y_min = y.min()
    y_max = y.max()
    return (y - y_min) / (y_max - y_min)


def make_regression(n_samples=5000, signal="heavisine", noise=0.03, random_state=None):
    """

    Parameters
    ----------
    n_samples : int, default=5000
        Number of desired samples

    signal : {"heavisine", "bumps", "blocks", "doppler"}, default="heavisine"
        Type of signal

    noise : float, default=0.03
        Standard-deviation of the Gaussian noise applied to the signal

    random_state : int
        Seed for numpy's random state

    Returns
    -------
    output : tuple
        Tuple containing (X, y) of shape (n_samples, 1) and (n_samples) corresponding
        to the input features and the output labels.
    """
    if random_state is not None:
        np.random.seed(random_state)
    X = np.random.uniform(size=n_samples)
    X = np.sort(X)
    y = get_signal(X, signal)
    X = X.reshape(n_samples, 1)
    y += noise * np.random.randn(n_samples)
    return X, y
